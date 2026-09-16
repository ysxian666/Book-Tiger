from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1] / "code" / "scripts" / "run_overnight.py"), run_name="__main__")
