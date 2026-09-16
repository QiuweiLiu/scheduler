"""R2 B04-B10 model package(r2b)。

导入时确保 scripts_behavior_r2_v5(含既有 r2 包)在 sys.path:
    r2b/__init__.py -> r2_v5/r2b -> r2_v5 -> scripts_behavior_r2_v5
"""
import os
import sys

_R2_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _R2_PARENT not in sys.path:
    sys.path.insert(0, _R2_PARENT)
