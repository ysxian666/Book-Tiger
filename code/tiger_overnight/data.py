"""Amazon Reviews 2023 Books preprocessing for the overnight protocol.

The pipeline deliberately never reads the first N rows.  It scans the complete
official Books JSONL stream, keeps all interactions belonging to a deterministic
user sample, applies repeated k-core filtering, builds leave-one-out splits, and
chooses the top item cap using train interactions only.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import duckdb

from .config import data_path, ensure_run_dirs


def _sql_path(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("'", "''")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _scalar(conn: duckdb.DuckDBPyConnection, query: str) -> Any:
    row = conn.execute(query).fetchone()
    return None if row is None else row[0]


class BooksOvernightPreprocessor:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        ensure_run_dirs(config)
        self.data_cfg = config["data"]
        self.data_dir = Path(config["paths"]["data_dir"])
        self.cache_dir = Path(config["paths"]["cache_dir"]) / "duckdb"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(database=":memory:")
        self.conn.execute(f"PRAGMA threads={int(self.data_cfg['duckdb_threads'])}")
        self.conn.execute(f"PRAGMA memory_limit='{self.data_cfg['duckdb_memory_limit']}'")
        self.conn.execute(f"PRAGMA temp_directory='{_sql_path(self.cache_dir)}'")
        self.user_sample_stats: dict[str, Any] = {}

    def close(self) -> None:
        self.conn.close()

    def _export(self, table: str, filename: str) -> None:
        target = _sql_path(self.data_dir / filename)
        compression = str(self.data_cfg.get("parquet_compression", "zstd"))
        self.conn.execute(
            f"COPY (SELECT * FROM {table}) TO '{target}' "
            f"(FORMAT PARQUET, COMPRESSION '{compression}')"
        )

    def _candidate_reviews(self, threshold: int | None) -> None:
        path = _sql_path(self.config["paths"]["review_path"])
        verified = "AND COALESCE(verified_purchase, FALSE)" if self.data_cfg.get("require_verified_purchase", False) else ""
        hash_filter = "" if threshold is None else f"AND HASH(CAST(user_id AS VARCHAR)) % 10000 < {int(threshold)}"
        self.conn.execute("DROP TABLE IF EXISTS candidate_reviews")
        self.conn.execute(
            f"""
            CREATE TABLE candidate_reviews AS
            SELECT * EXCLUDE (rn)
            FROM (
                SELECT
                    CAST(user_id AS VARCHAR) AS user_id,
                    COALESCE(NULLIF(CAST(parent_asin AS VARCHAR), ''), CAST(asin AS VARCHAR)) AS item_id,
                    CAST(timestamp AS BIGINT) AS timestamp,
                    CAST(rating AS DOUBLE) AS rating,
                    row_number() OVER (
                        PARTITION BY CAST(user_id AS VARCHAR),
                            COALESCE(NULLIF(CAST(parent_asin AS VARCHAR), ''), CAST(asin AS VARCHAR))
                        ORDER BY CAST(timestamp AS BIGINT) DESC
                    ) AS rn
                FROM read_json_auto(
                    '{path}',
                    format='newline_delimited',
                    ignore_errors=true,
                    union_by_name=true
                )
                WHERE user_id IS NOT NULL
                  AND timestamp IS NOT NULL
                  AND COALESCE(parent_asin, asin) IS NOT NULL
                  AND CAST(rating AS DOUBLE) >= {float(self.data_cfg['min_rating'])}
                  {hash_filter}
                  {verified}
            )
            WHERE rn = 1
            """
        )

    def _select_user_sample(self) -> None:
        target = int(self.data_cfg["target_interactions"])
        max_users = int(self.data_cfg["user_sample_max"])
        source_is_5core = bool(self.data_cfg.get("source_is_5core", False))
        attempts: list[dict[str, Any]] = []
        raw_users = 0
        raw_interactions = 0

        if source_is_5core:
            threshold: int | None = int(round(10000 * 0.20))
            for _ in range(8):
                self._candidate_reviews(threshold)
                users = int(_scalar(self.conn, "SELECT COUNT(DISTINCT user_id) FROM candidate_reviews") or 0)
                rows = int(_scalar(self.conn, "SELECT COUNT(*) FROM candidate_reviews") or 0)
                attempts.append({"threshold": threshold, "users": users, "interactions": rows})
                if users >= int(self.data_cfg["user_sample_min"]) and rows >= int(target * 1.25):
                    break
                threshold = min(10000, int(threshold) + 1500)
            else:
                raise RuntimeError(f"could not obtain a sufficiently large 5-core user sample: {attempts}")
            raw_users, raw_interactions = users, rows
            self.conn.execute(
                """
                CREATE TABLE interactions AS
                SELECT user_id, item_id, timestamp, rating
                FROM candidate_reviews
                """
            )
        else:
            print("[data] materializing full raw review stream for global 5-core filtering", flush=True)
            self._candidate_reviews(None)
            raw_users = int(_scalar(self.conn, "SELECT COUNT(DISTINCT user_id) FROM candidate_reviews") or 0)
            raw_interactions = int(_scalar(self.conn, "SELECT COUNT(*) FROM candidate_reviews") or 0)
            self.conn.execute(
                """
                CREATE TABLE interactions AS
                SELECT user_id, item_id, timestamp, rating
                FROM candidate_reviews
                """
            )
            self._apply_kcore()
            attempts.append({
                "threshold": None,
                "raw_users": raw_users,
                "raw_interactions": raw_interactions,
                "five_core_users": int(_scalar(self.conn, "SELECT COUNT(DISTINCT user_id) FROM interactions") or 0),
                "five_core_interactions": int(_scalar(self.conn, "SELECT COUNT(*) FROM interactions") or 0),
            })

        users = int(_scalar(self.conn, "SELECT COUNT(DISTINCT user_id) FROM interactions") or 0)
        rows = int(_scalar(self.conn, "SELECT COUNT(*) FROM interactions") or 0)
        if users < int(self.data_cfg["user_sample_min"]) or rows < int(target * 0.9):
            raise RuntimeError(f"5-core source is too small after filtering: users={users} interactions={rows}")

        self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE selected_users AS
            SELECT user_id
            FROM (
                SELECT user_id, row_number() OVER (ORDER BY HASH(user_id), user_id) AS rn
                FROM (SELECT DISTINCT user_id FROM interactions)
            )
            WHERE rn <= {max_users}
            """
        )
        self.conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE interactions_sampled AS
            SELECT i.user_id, i.item_id, i.timestamp, i.rating
            FROM interactions i
            INNER JOIN selected_users u USING (user_id)
            """
        )
        self.conn.execute("DROP TABLE interactions")
        self.conn.execute("ALTER TABLE interactions_sampled RENAME TO interactions")
        selected_users = int(_scalar(self.conn, "SELECT COUNT(DISTINCT user_id) FROM interactions") or 0)
        selected_interactions = int(_scalar(self.conn, "SELECT COUNT(*) FROM interactions") or 0)
        self.user_sample_stats = {
            "strategy": "global_5core_then_deterministic_user_sample" if not source_is_5core else "source_5core_hash_user_sample",
            "source_is_5core": source_is_5core,
            "attempts": attempts,
            "raw_users": raw_users,
            "raw_interactions": raw_interactions,
            "five_core_users": users,
            "five_core_interactions": rows,
            "selected_users_cap": max_users,
            "selected_users": selected_users,
            "selected_interactions": selected_interactions,
        }
        _write_json(self.data_dir / "sample_config.json", {
            "source": self.config["source"],
            "review_path": self.config["paths"]["review_path"],
            "meta_path": self.config["paths"]["meta_path"],
            "data": self.data_cfg,
            "seed": int(self.config["seed"]),
        })
        _write_json(self.data_dir / "user_sampling_stats.json", self.user_sample_stats)

    def _kcore_once(self) -> int:
        min_user = int(self.data_cfg["min_user_interactions"])
        min_item = int(self.data_cfg["min_item_interactions"])
        self.conn.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE interactions_next AS
            WITH user_ok AS (
                SELECT user_id FROM interactions GROUP BY user_id HAVING COUNT(*) >= {min_user}
            ), item_ok AS (
                SELECT item_id FROM interactions GROUP BY item_id HAVING COUNT(*) >= {min_item}
            )
            SELECT i.*
            FROM interactions i
            INNER JOIN user_ok u USING (user_id)
            INNER JOIN item_ok t USING (item_id)
            """
        )
        before = int(_scalar(self.conn, "SELECT COUNT(*) FROM interactions") or 0)
        after = int(_scalar(self.conn, "SELECT COUNT(*) FROM interactions_next") or 0)
        self.conn.execute("DROP TABLE interactions")
        self.conn.execute("ALTER TABLE interactions_next RENAME TO interactions")
        return before - after

    def _apply_kcore(self) -> None:
        removed = 1
        rounds = 0
        while removed > 0 and rounds < 12:
            rounds += 1
            removed = self._kcore_once()

    def _split_tables(self) -> None:
        min_sequence = int(self.data_cfg["min_sequence_length"])
        self.conn.execute(
            """
            CREATE OR REPLACE TABLE ranked_interactions AS
            SELECT
                user_id, item_id, timestamp, rating,
                row_number() OVER (PARTITION BY user_id ORDER BY timestamp DESC, item_id DESC) AS reverse_rank,
                COUNT(*) OVER (PARTITION BY user_id) AS user_count
            FROM interactions
            """
        )
        self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE eligible_interactions AS
            SELECT user_id, item_id, timestamp, rating, reverse_rank
            FROM ranked_interactions
            WHERE user_count >= {min_sequence}
            """
        )
        self.conn.execute(
            """
            CREATE OR REPLACE TABLE test_interactions AS
            SELECT user_id, item_id, timestamp, rating FROM eligible_interactions WHERE reverse_rank = 1
            """
        )
        self.conn.execute(
            """
            CREATE OR REPLACE TABLE valid_interactions AS
            SELECT user_id, item_id, timestamp, rating FROM eligible_interactions WHERE reverse_rank = 2
            """
        )
        self.conn.execute(
            """
            CREATE OR REPLACE TABLE train_interactions AS
            SELECT user_id, item_id, timestamp, rating FROM eligible_interactions WHERE reverse_rank > 2
            """
        )

    def _apply_item_cap(self) -> None:
        top_items = int(self.data_cfg["top_items"])
        self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE top_items AS
            SELECT item_id, train_count
            FROM (
                SELECT item_id, COUNT(*) AS train_count,
                       row_number() OVER (ORDER BY COUNT(*) DESC, item_id) AS item_rank
                FROM train_interactions
                GROUP BY item_id
            )
            WHERE item_rank <= {top_items}
            """
        )
        self.conn.execute(
            """
            CREATE OR REPLACE TEMP TABLE interactions_filtered AS
            SELECT i.user_id, i.item_id, i.timestamp, i.rating
            FROM interactions i
            INNER JOIN top_items t USING (item_id)
            """
        )
        self.conn.execute("DROP TABLE interactions")
        self.conn.execute("ALTER TABLE interactions_filtered RENAME TO interactions")
        self._apply_kcore()
        self._split_tables()
        _write_json(self.data_dir / "item_cap_stats.json", {
            "rule": "top_items_by_train_count_only",
            "top_items": top_items,
            "selected_items": int(_scalar(self.conn, "SELECT COUNT(*) FROM top_items") or 0),
            "train_interactions_used_for_ranking": int(_scalar(self.conn, "SELECT COUNT(*) FROM train_interactions") or 0),
            "valid_or_test_used_for_ranking": False,
        })

    def _create_user_sequences(self) -> None:
        max_history = max(20, int(self.config["generator"]["max_history_items"]))
        self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE user_sequences AS
            WITH joined AS (
                SELECT
                    t.user_id,
                    list(t.item_id ORDER BY t.timestamp) AS history_item_ids,
                    list(t.timestamp ORDER BY t.timestamp) AS history_timestamps,
                    min(t.timestamp) AS first_timestamp,
                    max(t.timestamp) AS last_timestamp,
                    count(*) AS history_length
                FROM train_interactions t
                GROUP BY t.user_id
            )
            SELECT
                user_id,
                list_slice(history_item_ids, greatest(1, length(history_item_ids) - {max_history} + 1), length(history_item_ids)) AS history_item_ids,
                list_slice(history_timestamps, greatest(1, length(history_timestamps) - {max_history} + 1), length(history_timestamps)) AS history_timestamps,
                history_length,
                first_timestamp,
                last_timestamp,
                v.item_id AS valid_item_id,
                s.item_id AS test_item_id
            FROM joined
            INNER JOIN valid_interactions v USING (user_id)
            INNER JOIN test_interactions s USING (user_id)
            """
        )

    def _create_item_stats(self) -> None:
        quantile = 0.8
        self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE item_stats AS
            WITH counts AS (
                SELECT
                    t.item_id,
                    COUNT(*)::BIGINT AS train_count,
                    COALESCE(v.valid_count, 0)::BIGINT AS valid_count,
                    COALESCE(s.test_count, 0)::BIGINT AS test_count
                FROM train_interactions t
                LEFT JOIN (
                    SELECT item_id, COUNT(*) AS valid_count FROM valid_interactions GROUP BY item_id
                ) v USING (item_id)
                LEFT JOIN (
                    SELECT item_id, COUNT(*) AS test_count FROM test_interactions GROUP BY item_id
                ) s USING (item_id)
                GROUP BY t.item_id, v.valid_count, s.test_count
            )
            SELECT
                *,
                ntile(5) OVER (ORDER BY train_count) AS frequency_quintile,
                CASE
                    WHEN train_count BETWEEN 1 AND 5 THEN 'few_shot_item'
                    WHEN train_count <= quantile_cont(train_count, {quantile}) OVER () THEN 'long_tail_item'
                    ELSE 'head_item'
                END AS frequency_group
            FROM counts
            """
        )

    def _create_catalog(self) -> None:
        meta_path = _sql_path(self.config["paths"]["meta_path"])
        self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE raw_meta AS
            SELECT
                item_id, title, subtitle, author, main_category,
                categories, features, description, details, store,
                price, average_rating, rating_number, bought_together
            FROM (
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
                  AND CAST(parent_asin AS VARCHAR) IN (SELECT item_id FROM top_items)
            )
            WHERE rn = 1
            """
        )
        self.conn.execute(
            """
            CREATE OR REPLACE TABLE catalog AS
            SELECT
                s.item_id, m.title, m.subtitle, m.author, m.main_category,
                m.categories, m.features, m.description, m.details, m.store,
                m.price, m.average_rating, m.rating_number, m.bought_together,
                s.train_count, s.valid_count, s.test_count,
                COALESCE(
                    NULLIF(TRIM(COALESCE(m.title, '')), ''),
                    NULLIF(TRIM(COALESCE(m.description, '')), ''),
                    NULLIF(TRIM(COALESCE(m.categories, '')), '')
                ) AS content_text,
                CASE WHEN m.item_id IS NULL THEN FALSE ELSE TRUE END AS metadata_present
            FROM item_stats s
            LEFT JOIN raw_meta m USING (item_id)
            ORDER BY s.item_id
            """
        )
        # Aggregate coverage is persisted in final_data_stats.json and metadata_audit.json.

    def _write_stats(self) -> dict[str, Any]:
        counts = self.conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM interactions),
                (SELECT COUNT(DISTINCT user_id) FROM interactions),
                (SELECT COUNT(DISTINCT item_id) FROM interactions),
                (SELECT COUNT(*) FROM train_interactions),
                (SELECT COUNT(*) FROM valid_interactions),
                (SELECT COUNT(*) FROM test_interactions),
                (SELECT COUNT(*) FROM catalog),
                (SELECT COUNT(*) FROM user_sequences)
            """
        ).fetchone()
        stats = {
            "interactions": int(counts[0]),
            "users": int(counts[1]),
            "items": int(counts[2]),
            "train_interactions": int(counts[3]),
            "valid_interactions": int(counts[4]),
            "test_interactions": int(counts[5]),
            "catalog_items": int(counts[6]),
            "eligible_users": int(counts[7]),
            "target_interactions": int(self.data_cfg["target_interactions"]),
            "top_items": int(self.data_cfg["top_items"]),
            "split": "leave_one_out: test=last, valid=second_last, train=earlier",
            "item_cap": "train-only frequency",
            "metadata_coverage": float(_scalar(self.conn, "SELECT COUNT(*) FROM catalog WHERE metadata_present") or 0) / max(int(counts[6]), 1),
            "content_coverage": float(_scalar(self.conn, "SELECT COUNT(*) FROM catalog WHERE content_text IS NOT NULL AND content_text <> ''") or 0) / max(int(counts[6]), 1),
        }
        _write_json(self.data_dir / "final_data_stats.json", stats)
        split_stats = {
            "users_train": int(_scalar(self.conn, "SELECT COUNT(DISTINCT user_id) FROM train_interactions") or 0),
            "users_valid": int(_scalar(self.conn, "SELECT COUNT(DISTINCT user_id) FROM valid_interactions") or 0),
            "users_test": int(_scalar(self.conn, "SELECT COUNT(DISTINCT user_id) FROM test_interactions") or 0),
            "train_interactions": stats["train_interactions"],
            "valid_interactions": stats["valid_interactions"],
            "test_interactions": stats["test_interactions"],
        }
        _write_json(self.data_dir / "split_stats.json", split_stats)
        return stats

    def run(self, force: bool = False) -> dict[str, Any]:
        final_stats_path = self.data_dir / "final_data_stats.json"
        if final_stats_path.exists() and not force:
            with final_stats_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        print("[data] scanning full review stream and selecting users", flush=True)
        self._select_user_sample()
        print(f"[data] sampled users={self.user_sample_stats['selected_users']:,} interactions={self.user_sample_stats['selected_interactions']:,}", flush=True)
        self._apply_kcore()
        print(f"[data] k-core interactions={_scalar(self.conn, 'SELECT COUNT(*) FROM interactions'):,}", flush=True)
        self._split_tables()
        self._apply_item_cap()
        print(f"[data] item cap interactions={_scalar(self.conn, 'SELECT COUNT(*) FROM interactions'):,}", flush=True)
        self._create_item_stats()
        print("[data] reading metadata for top items", flush=True)
        self._create_catalog()
        self._create_user_sequences()
        for table, filename in (
            ("interactions", "interactions.parquet"),
            ("train_interactions", "train_interactions.parquet"),
            ("valid_interactions", "valid_interactions.parquet"),
            ("test_interactions", "test_interactions.parquet"),
            ("user_sequences", "user_sequences.parquet"),
            ("item_stats", "item_stats.parquet"),
            ("catalog", "catalog.parquet"),
            ("top_items", "top_items.parquet"),
        ):
            self._export(table, filename)
        stats = self._write_stats()
        if int(stats["interactions"]) < int(self.data_cfg["target_interactions"]):
            raise RuntimeError(
                f"final interactions {stats['interactions']} are below target {self.data_cfg['target_interactions']}; "
                "increase data.user_sample_max or target_interactions is impossible for this source."
            )
        minimum_coverage = float(self.data_cfg["min_metadata_coverage"])
        audit = {
            "metadata_coverage": stats["metadata_coverage"],
            "content_coverage": stats["content_coverage"],
            "require_metadata_coverage": minimum_coverage,
            "passed": stats["metadata_coverage"] >= minimum_coverage,
        }
        _write_json(self.data_dir / "metadata_audit.json", audit)
        if not audit["passed"]:
            raise RuntimeError(
                f"metadata coverage {stats['metadata_coverage']:.4f} is below required {minimum_coverage:.4f}"
            )
        return stats

    def cleanup(self) -> None:
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir, ignore_errors=True)


def preprocess_books(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    processor = BooksOvernightPreprocessor(config)
    try:
        return processor.run(force=force)
    finally:
        processor.close()
        processor.cleanup()


def parquet_path(config: dict[str, Any], name: str) -> str:
    return str(data_path(config, name))
