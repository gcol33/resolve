// crash_smoke - end-to-end verification of the Windows crash handler
// (issues #18 / #19). Not a Catch2 test: it deliberately faults, so it must run
// as its own short-lived process and be checked from outside via its exit code.
// Driven by src/core/dev_notes/crash_smoke_drive.ps1.
//
// Modes (argv[1]):
//   clean          - install handler, exit 0 normally (no fault).
//   fault-midrun   - install handler, do NOT signal work complete, then trigger
//                    an access violation. The handler must TerminateProcess with
//                    the fault's NTSTATUS (0xC0000005) -- fast, no JIT-debugger
//                    hang (issue #19).
//   fault-teardown - install handler, signal work complete, then trigger an
//                    access violation. The handler must TerminateProcess with
//                    the shutdown code 0 -- the benign teardown path (issue #18).

#include "resolve/process.hpp"

#include <cstdio>
#include <cstring>

namespace {

[[noreturn]] void trigger_access_violation() {
    // volatile so the optimizer cannot elide the null write. Raises
    // STATUS_ACCESS_VIOLATION (0xC0000005), an unhandled SEH exception that the
    // installed filter intercepts.
    volatile int* p = nullptr;
    *p = 1;
    // Unreachable, but keep the compiler from proving [[noreturn]] violated.
    std::fputs("unreachable\n", stderr);
    for (;;) {}
}

}  // namespace

int main(int argc, char* argv[]) {
    const char* mode = (argc > 1) ? argv[1] : "clean";

    // Shutdown code 0: the teardown-after-success path exits 0.
    resolve::install_crash_handler(0);

    if (std::strcmp(mode, "clean") == 0) {
        resolve::signal_work_complete();
        std::puts("clean exit");
        return 0;
    }

    if (std::strcmp(mode, "fault-teardown") == 0) {
        resolve::signal_work_complete();
        trigger_access_violation();
    }

    // "fault-midrun" (default for any other arg): no signal_work_complete.
    trigger_access_violation();
}
