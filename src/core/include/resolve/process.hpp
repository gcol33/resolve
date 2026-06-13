#pragma once

namespace resolve {

// Limit libtorch's intra-op (OpenMP / c10 thread pool) and inter-op
// (std::thread launch pool) parallelism.
//
// intraop_threads <= 0 leaves the intra-op pool at libtorch's default;
// interop_threads <= 0 leaves the inter-op pool at its default. A positive
// value pins the corresponding pool to that thread count.
//
// On Windows the worker threads in these pools are joined during process exit,
// inside libtorch's static-destructor / DLL_PROCESS_DETACH teardown. Under the
// Rscript.exe launcher that join races with the rest of the C runtime teardown
// and faults with an access violation after the script has already finished and
// flushed its output (issue #18). Pinning both pools to 1 at load leaves no
// worker threads to join, removing the race at its source. Harmless on other
// platforms; the engine's heavy parallelism lives in CUDA kernels and batched
// tensor ops, not the host thread pools, for the thin-client (metrics / small
// CPU) workloads that run through the R bindings.
//
// Best-effort: the inter-op thread count can only be changed before that pool
// starts (libtorch throws otherwise), so call at startup (R .onLoad, module
// import, CLI entry). A rejected change is swallowed rather than propagated --
// the pinning is a defensive measure, not a correctness requirement -- so this
// never throws.
void set_thread_pools(int intraop_threads, int interop_threads = -1);

// Windows-only process-crash hardening (a no-op on every other platform).
//
// Converts an otherwise-unhandled native exception (access violation, in-page
// error, ...) into an immediate, deterministic process termination instead of:
//   * a Windows Error Reporting / JIT-debugger (AeDebug, vsjitdebugger)
//     handshake that hangs a headless training worker forever, holding the GPU
//     and stalling the batch (issue #19), or
//   * a post-main libtorch static-destructor access violation that crashes the
//     Rscript.exe launcher at teardown (issue #18).
//
// Installs, best-effort and Windows-only:
//   * SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX |
//     SEM_NOOPENFILEERRORBOX) -- suppress the critical-error / crash dialogs;
//   * WerSetFlags(WER_FAULT_REPORTING_NO_UI) -- suppress the WER UI;
//   * SetUnhandledExceptionFilter(filter) -- the filter calls TerminateProcess
//     (which bypasses AeDebug entirely) with the exit code chosen by
//     crash_exit_code(): the shutdown code once signal_work_complete() has run
//     (the work finished; a later fault is a teardown artifact -- issue #18),
//     or a non-zero failure code derived from the exception otherwise (a
//     mid-run crash the orchestrator must record -- issue #19).
//
// No minidump is written: the canonical #19 trigger is a faulting storage
// device, so touching the disk from inside the fault handler could itself hang.
// The fault's NTSTATUS is carried out through the process exit code instead.
//
// Idempotent: only the first call installs the handler; later calls only update
// the shutdown exit code.
void install_crash_handler(int shutdown_exit_code = 0);

// Mark that all real engine work for this process is complete. After this call
// install_crash_handler's filter treats any native fault as a benign teardown
// artifact and exits with the shutdown code rather than a failure code. Wired
// to the normal-shutdown hook of each binding: R's on-exit finalizer, Python's
// atexit, the CLI's end of main. Safe to call when no handler is installed.
void signal_work_complete();

// Pure decision helper behind install_crash_handler's filter, factored out so
// the exit-code policy is unit-testable without actually faulting a process.
//
//   work_complete == true  -> shutdown_exit_code (cast to unsigned), the clean
//                             teardown-after-success path (issue #18);
//   work_complete == false -> a guaranteed non-zero code: the NTSTATUS in
//                             exception_code when it is non-zero (so the
//                             orchestrator records the real fault, e.g.
//                             0xC0000005 / 0xC0000006), else a fixed non-zero
//                             fallback (issue #19).
unsigned int crash_exit_code(
    bool work_complete,
    int shutdown_exit_code,
    unsigned long exception_code
) noexcept;

} // namespace resolve
