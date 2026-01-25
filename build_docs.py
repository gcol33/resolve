#!/usr/bin/env python3
"""
Documentation builder for RESOLVE.

Thin wrapper around the global build script at ~/.python/build_docs.py

Usage:
    python build_docs.py              # Build docs
    python build_docs.py --serve      # Build and serve locally
    python build_docs.py --api        # Also generate API docs
"""

import sys
from pathlib import Path

# Use global build script if available
global_script = Path.home() / ".python" / "build_docs.py"

if global_script.exists():
    sys.path.insert(0, str(global_script.parent))
    from build_docs import main
    main()
else:
    print(f"Global build script not found at {global_script}")
    print("Install it or run: python ~/.python/build_docs.py")
    sys.exit(1)
