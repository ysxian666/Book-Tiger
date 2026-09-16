"""DuckDB-based preprocessing for the Amazon Reviews 2023 Books category.

The raw files used by this project are JSON-lines and are tens of GB.  The
implementation therefore keeps the main lifecycle in DuckDB and only writes
Parquet artifacts consumed by downstream Python code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from tiger_rec.config import Config
from tiger_rec.utils import human_int, write_json


def _escape_sql_path(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("'", "''")


class AmazonBooksPreprocessor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out = cfg.paths.out
        self.out.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(database=str(self.out / "preprocess.duckdb"))
        self.conn.execute(f"PRAGMA threads={int(cfg.data.duckdb_threads)}")
        self.conn.execute(f"PRAGMA memory_limit='{cfg.data.duckdb_memory_limit}'")
        self.conn.execute(f"PRAGMA temp_directory='{_escape_sql_path(self.out / 'duckdb_tmp')}'")

    def close(self) -> None:
        self.conn.close()

    def _copy(self, query: str, filename: str) -> None:
        target = _escape_sql_path(self.out / filename)
        sql = (
            f"COPY ({query}) TO '{target}' "
            f"(FORMAT PARQUET, COMPRESSION '{self.cfg.data.parquet_compression}')"
        )
        self.conn.execute(sql)

    def _copy_table(self, table: str, filename: str) -> None:
        self._copy(f"SELECT * FROM {table}", filename)

    def _create_reviews(self) -> None:
        review_path = _escape_sql_path(self.cfg.paths.review_path)
        cfg = self.cfg.data
        limit = f"LIMIT {int(cfg.max_reviews)}" if cfg.max_reviews > 0 else ""
        verified_filter = "AND verified_purchase = TRUE" if cfg.require_verified_purchase else ""
        self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE reviews AS
            SELECT rating, user_id, item_id, timestamp, review_text, verified_purchase
            FROM (
                SELECT * EXCLUDE (rn)
                FROM (
                    SELECT
                        CAST(rating AS DOUBLE) AS rating,
                        CAST(user_id AS VARCHAR) AS user_id,
                        COALESCE(NULLIF(CAST(parent_asin AS VARCHAR), ''), CAST(asin AS VARCHAR)) AS item_id,
                        CAST(timestamp AS BIGINT) AS timestamp,
                        COALESCE(CAST(text AS VARCHAR), '') AS review_text,
                        COALESCE(CAST(verified_purchase AS BOOLEAN), FALSE) AS verified_purchase,
                        row_number() OVER (
                            PARTITION BY CAST(user_id AS VARCHAR),
                                         COALESCE(NULLIF(CAST(parent_asin AS VARCHAR), ''), CAST(asin AS VARCHAR))
                            ORDER BY CAST(timestamp AS BIGINT) DESC
                        ) AS rn
                    FROM read_json_auto(
                        '{review_path}',
                        format='newline_delimited',
                        ignore_errors=true,
                        union_by_name=true
                    )
                    WHERE user_id IS NOT NULL
                      AND timestamp IS NOT NULL
                      AND COALESCE(parent_asin, asin) IS NOT NULL
                      AND CAST(rating AS DOUBLE) >= {float(cfg.min_rating)}
                      {verified_filter}
                    {limit}
                )
                WHERE rn = 1
            )
            """
        )

    def _kcore_once(self) -> int:
        cfg = self.cfg.data
        self.conn.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE interactions_next AS
            WITH user_ok AS (
                SELECT user_id FROM interactions
                GROUP BY user_id HAVING COUNT(*) >= {int(cfg.min_user_interactions)}
            ), item_ok AS (
                SELECT item_id FROM interactions
                GROUP BY item_id HAVING COUNT(*) >= {int(cfg.min_item_interactions)}
            )
            SELECT i.*
            FROM interactions i
            INNER JOIN user_ok u USING (user_id)
            INNER JOIN item_ok t USING (item_id)
            """
        )
        old_count = self.conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        new_count = self.conn.execute("SELECT COUNT(*) FROM interactions_next").fetchone()[0]
        self.conn.execute("DROP TABLE interactions")
        self.conn.execute("ALTER TABLE interactions_next RENAME TO interactions")
        return int(new_count if new_count != old_count else 0)

    def _apply_kcore(self) -> None:
        self.conn.execute("CREATE OR REPLACE TABLE interactions AS SELECT * FROM reviews")
        previous = -1
        for iteration in range(self.cfg.data.kcore_iterations):
            current = self.conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            changed = self._kcore_once()
            new_count = self.conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            print(
                f"[k-core] iteration={iteration + 1} "
                f"rows={human_int(int(new_count))} removed={human_int(int(current - new_count))}"
            )
            if previous == new_count or changed == 0:
                break
            previous = int(new_count)

    def _apply_catalog_caps(self) -> None:
        cfg = self.cfg.data
        if cfg.max_items > 0:
            self.conn.execute(
                f"""
                CREATE OR REPLACE TABLE keep_items AS
                SELECT item_id FROM interactions
                GROUP BY item_id
                ORDER BY COUNT(*) DESC, item_id
                LIMIT {int(cfg.max_items)}
                """
            )
            self.conn.execute("DELETE FROM interactions WHERE item_id NOT IN (SELECT item_id FROM keep_items)")
        if cfg.max_users > 0:
            self.conn.execute(
                f"""
                CREATE OR REPLACE TABLE keep_users AS
                SELECT user_id FROM interactions
                GROUP BY user_id
                ORDER BY COUNT(*) DESC, user_id
                LIMIT {int(cfg.max_users)}
                """
            )
            self.conn.execute("DELETE FROM interactions WHERE user_id NOT IN (SELECT user_id FROM keep_users)")
        if cfg.max_items > 0 or cfg.max_users > 0:
            for _ in range(self.cfg.data.kcore_iterations):
                changed = self._kcore_once()
                if changed == 0:
                    break

    def _filter_short_sequences(self) -> None:
        keep = max(3, int(self.cfg.data.min_sequence_length))
        self.conn.execute(
            f"""
            DELETE FROM interactions
            WHERE user_id IN (
                SELECT user_id FROM interactions
                GROUP BY user_id HAVING COUNT(*) < {keep}
            )
            """
        )

    def _create_splits(self) -> None:
        # Last item = test, second-last = validation, all earlier = train.
        self.conn.execute(
            """
            CREATE OR REPLACE TABLE ordered AS
            SELECT
                *,
                row_number() OVER (PARTITION BY user_id ORDER BY timestamp, item_id) AS seq_pos,
                COUNT(*) OVER (PARTITION BY user_id) AS seq_len
            FROM interactions
            """
        )
        self.conn.execute("CREATE OR REPLACE TABLE train_interactions AS SELECT * FROM ordered WHERE seq_pos <= seq_len - 2")
        self.conn.execute("CREATE OR REPLACE TABLE valid_interactions AS SELECT * FROM ordered WHERE seq_pos = seq_len - 1")
        self.conn.execute("CREATE OR REPLACE TABLE test_interactions AS SELECT * FROM ordered WHERE seq_pos = seq_len")
        self.conn.execute(
            """
            CREATE OR REPLACE TABLE user_sequences AS
            SELECT
                user_id,
                array_agg(item_id ORDER BY seq_pos) AS item_seq,
                array_agg(rating ORDER BY seq_pos) AS rating_seq,
                array_agg(timestamp ORDER BY seq_pos) AS timestamp_seq
            FROM train_interactions
            GROUP BY user_id
            """
        )

    def _create_item_stats(self) -> None:
        self.conn.execute(
            """
            CREATE OR REPLACE TABLE item_stats AS
            WITH train_counts AS (
                SELECT item_id, COUNT(*) AS train_count
                FROM train_interactions GROUP BY item_id
            ), valid_counts AS (
                SELECT item_id, COUNT(*) AS valid_count
                FROM valid_interactions GROUP BY item_id
            ), test_counts AS (
                SELECT item_id, COUNT(*) AS test_count
                FROM test_interactions GROUP BY item_id
            )
            SELECT
                i.item_id,
                COALESCE(t.train_count, 0) AS train_count,
                COALESCE(v.valid_count, 0) AS valid_count,
                COALESCE(s.test_count, 0) AS test_count
            FROM (SELECT DISTINCT item_id FROM interactions) i
            LEFT JOIN train_counts t USING (item_id)
            LEFT JOIN valid_counts v USING (item_id)
            LEFT JOIN test_counts s USING (item_id)
            """
        )

    def _create_catalog(self) -> None:
        meta_path = _escape_sql_path(self.cfg.paths.meta_path)
        limit = f"LIMIT {int(self.cfg.data.max_meta_records)}" if self.cfg.data.max_meta_records > 0 else ""
        self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE raw_meta AS
            SELECT * EXCLUDE (rn) FROM (
                SELECT
                    CAST(parent_asin AS VARCHAR) AS item_id,
                    CAST(title AS VARCHAR) AS title,
                    CAST(subtitle AS VARCHAR) AS subtitle,
                    CAST(author AS VARCHAR) AS author,
                    CAST(main_category AS VARCHAR) AS main_category,
                    CAST(categories AS VARCHAR) AS categories,
                    CAST(features AS VARCHAR) AS features,
                    CAST(description AS VARCHAR) AS description,
                    CAST(details AS VARCHAR) AS details,
                    CAST(store AS VARCHAR) AS store,
                    TRY_CAST(price AS DOUBLE) AS price,
                    TRY_CAST(average_rating AS DOUBLE) AS average_rating,
                    TRY_CAST(rating_number AS BIGINT) AS rating_number,
                    CAST(bought_together AS VARCHAR) AS bought_together,
                    row_number() OVER (
                        PARTITION BY CAST(parent_asin AS VARCHAR)
                        ORDER BY TRY_CAST(rating_number AS BIGINT) DESC NULLS LAST
                    ) AS rn
                FROM read_json_auto(
                    '{meta_path}',
                    format='newline_delimited',
                    ignore_errors=true,
                    union_by_name=true
                )
                WHERE parent_asin IS NOT NULL
                {limit}
            )
            WHERE rn = 1
            """
        )
        self.conn.execute(
            """
            CREATE OR REPLACE TABLE catalog AS
            SELECT
                s.item_id,
                m.title,
                m.subtitle,
                m.author,
                m.main_category,
                m.categories,
                m.features,
                m.description,
                m.details,
                m.store,
                m.price,
                m.average_rating,
                m.rating_number,
                m.bought_together,
                s.train_count,
                s.valid_count,
                s.test_count
            FROM item_stats s
            LEFT JOIN raw_meta m USING (item_id)
            """
        )

    def run(self) -> dict[str, Any]:
        print(f"[data] review={self.cfg.paths.review_path}")
        print(f"[data] meta={self.cfg.paths.meta_path}")
        self._create_reviews()
        self._apply_kcore()
        self._apply_catalog_caps()
        self._filter_short_sequences()
        self._create_splits()
        self._create_item_stats()
        self._create_catalog()

        files = {
            "interactions": "interactions.parquet",
            "train_interactions": "train_interactions.parquet",
            "valid_interactions": "valid_interactions.parquet",
            "test_interactions": "test_interactions.parquet",
            "user_sequences": "user_sequences.parquet",
            "item_stats": "item_stats.parquet",
            "catalog": "catalog.parquet",
        }
        for table, filename in files.items():
            print(f"[data] writing {filename}")
            self._copy_table(table, filename)

        rows = self.conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM interactions),
                (SELECT COUNT(DISTINCT user_id) FROM interactions),
                (SELECT COUNT(DISTINCT item_id) FROM interactions),
                (SELECT COUNT(*) FROM train_interactions),
                (SELECT COUNT(*) FROM valid_interactions),
                (SELECT COUNT(*) FROM test_interactions),
                (SELECT COUNT(*) FROM catalog)
            """
        ).fetchone()
        stats = {
            "interactions": int(rows[0]),
            "users": int(rows[1]),
            "items": int(rows[2]),
            "train_interactions": int(rows[3]),
            "valid_interactions": int(rows[4]),
            "test_interactions": int(rows[5]),
            "catalog_items": int(rows[6]),
            "split": "leave_one_out: test=last, valid=second_last, train=earlier",
            "raw_review_path": self.cfg.paths.review_path,
            "raw_meta_path": self.cfg.paths.meta_path,
        }
        write_json(self.out / "data_stats.json", stats)
        print(f"[data] done: {stats}")
        return stats


def run_preprocessing(cfg: Config) -> dict[str, Any]:
    processor = AmazonBooksPreprocessor(cfg)
    try:
        return processor.run()
    finally:
        processor.close()
