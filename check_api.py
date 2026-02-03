import sys
sys.path.insert(0, 'src/core/python/src')
from resolve_core import TargetSpec
t = TargetSpec()
print('TargetSpec attributes:', dir(t))
