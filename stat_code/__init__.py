# stat_code/__init__.py

from importlib import import_module
from pathlib import Path

REGISTRY = {}

def register(test_cls):
    REGISTRY[test_cls.name] = test_cls()

    return test_cls

# 自動載入 stat_code 底下所有檔案
for file in Path(__file__).parent.glob('*.py'):
    if file.stem not in ("__init__", "base"):
        import_module(f"stat_code.{file.stem}")