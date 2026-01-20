# stat_code/__init__.py

from importlib import import_module
from pathlib import Path

SV_REGISTRY = {}

def sv_register(test_cls):
    SV_REGISTRY[test_cls.name] = test_cls()

    return test_cls

# 自動載入 stat_code 底下所有檔案
for file in Path(__file__).parent.glob('*.py'):
    if file.stem not in ("__init__", "sv_base"):
        import_module(f"survival_analysis.{file.stem}")