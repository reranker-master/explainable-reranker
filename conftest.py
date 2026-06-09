import os
from pathlib import Path

def pytest_configure(config):
    cache_dir = Path(__file__).parent / ".cache" / "pycache"
    os.environ.setdefault("PYTHONPYCACHEPREFIX", str(cache_dir))
