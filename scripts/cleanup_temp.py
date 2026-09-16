from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1] / "code" / "scripts" / "cleanup_temp.py"), run_name="__main__")
