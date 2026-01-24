import shutil
import glob
import os
import subprocess

src = 'C:/Users/Gilles Colling/Documents/dev/libtorch/lib'
dst = 'C:/Users/Gilles Colling/Documents/dev/resolve-core/build/tests/Release'

# Copy DLLs
for dll in glob.glob(os.path.join(src, '*.dll')):
    shutil.copy(dll, dst)
    print(f'Copied {os.path.basename(dll)}')

print('\nRunning tests...\n')
exe = os.path.join(dst, 'resolve_tests.exe')
result = subprocess.run([exe], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print('STDERR:', result.stderr)
    print(f'\nExit code: {result.returncode}')
