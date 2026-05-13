# CMake generated Testfile for 
# Source directory: C:/Users/Gilles Colling/Documents/dev/RESOLVE/src/core/tests
# Build directory: C:/Users/Gilles Colling/Documents/dev/RESOLVE/src/core/build_asan/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(resolve_tests "C:/Users/Gilles Colling/Documents/dev/RESOLVE/src/core/build_asan/tests/resolve_tests.exe")
set_tests_properties(resolve_tests PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Gilles Colling/Documents/dev/RESOLVE/src/core/tests/CMakeLists.txt;44;add_test;C:/Users/Gilles Colling/Documents/dev/RESOLVE/src/core/tests/CMakeLists.txt;0;")
subdirs("../_deps/catch2-build")
