from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1] / "code" / "scripts" / "link_or_download_data.py"), run_name="__main__")
