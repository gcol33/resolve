#include "resolve/process.hpp"

#include <atomic>

#include <ATen/Parallel.h>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

namespace resolve {

namespace {

// Crash-handler policy, read by the Windows handlers and flipped by
// signal_work_complete. Atomics because a handler can run on any thread.
std::atomic<bool> g_handler_installed{false};
std::atomic<bool> g_work_complete{false};
std::atomic<int>  g_shutdown_exit_code{0};

}  // namespace

unsigned int crash_exit_code(
    bool work_complete,
    int shutdown_exit_code,
    unsigned long exception_code
) noexcept {
    if (work_complete) {
        return static_cast<unsigned int>(shutdown_exit_code);
    }
    if (exception_code != 0UL) {
        return static_cast<unsigned int>(exception_code);
    }
    // No usable NTSTATUS but a real mid-run fault: never report success.
    return 3U;
}

void set_thread_pools(int intraop_threads, int interop_threads) {
    // Best-effort: the thread pinning is a defensive measure against the
    // teardown-join race (issue #18), not a correctness requirement, so a
    // libtorch rejection (e.g. the inter-op pool already started) is swallowed
    // rather than propagated to the binding's load path.
    if (intraop_threads > 0) {
        try {
            at::set_num_threads(intraop_threads);
        } catch (...) {
        }
    }
    if (interop_threads > 0) {
        try {
            at::set_num_interop_threads(interop_threads);
        } catch (...) {
        }
    }
}

#if defined(_WIN32)

namespace {

// True for an access-violation-class NTSTATUS (top nibble 0xC: access violation
// 0xC0000005, in-page error 0xC0000006, illegal instruction, stack overflow,
// ...). Deliberately excludes the C++ EH exception (0xE06D7363) and debugger
// events (0x8xxxxxxx) so normal control flow and breakpoints are never mistaken
// for a crash.
bool is_fatal_ntstatus(unsigned long code) {
    return (code & 0xF0000000UL) == 0xC0000000UL;
}

unsigned long exception_code_of(EXCEPTION_POINTERS* info) {
    return (info != nullptr && info->ExceptionRecord != nullptr)
               ? info->ExceptionRecord->ExceptionCode
               : 0UL;
}

// Last-resort handler for a mid-run fault while the process is otherwise live
// (issue #19). Reached only if no frame-based handler claimed the exception.
LONG WINAPI resolve_unhandled_exception_filter(EXCEPTION_POINTERS* info) {
    const unsigned int code = crash_exit_code(
        g_work_complete.load(std::memory_order_acquire),
        g_shutdown_exit_code.load(std::memory_order_acquire),
        exception_code_of(info));
    // TerminateProcess never consults the AeDebug post-mortem-debugger key, so a
    // headless run cannot hang waiting for a JIT debugger to attach (issue #19).
    ::TerminateProcess(::GetCurrentProcess(), code);
    return EXCEPTION_EXECUTE_HANDLER;  // unreached if TerminateProcess succeeds
}

// Vectored handler: fires for every exception, including faults during the
// post-main C-runtime teardown / DLL_PROCESS_DETACH, where the unhandled-
// exception filter above is no longer invoked -- that teardown window is the
// suspected point of the libtorch static-destructor access violation that
// crashes the Rscript.exe launcher (issue #18). It acts ONLY once work is
// complete (so a live mid-run fault still flows to the normal handlers / the
// #19 path) and ONLY on an access-violation-class fault (so C++ exceptions and
// breakpoints pass through untouched). In that narrow case the fault is a
// benign teardown artifact, so exit with the shutdown code.
LONG WINAPI resolve_vectored_handler(EXCEPTION_POINTERS* info) {
    if (g_work_complete.load(std::memory_order_acquire) &&
        is_fatal_ntstatus(exception_code_of(info))) {
        ::TerminateProcess(
            ::GetCurrentProcess(),
            static_cast<UINT>(g_shutdown_exit_code.load(std::memory_order_acquire)));
    }
    return EXCEPTION_CONTINUE_SEARCH;  // let normal dispatch proceed otherwise
}

}  // namespace

void install_crash_handler(int shutdown_exit_code) {
    g_shutdown_exit_code.store(shutdown_exit_code, std::memory_order_release);

    if (g_handler_installed.exchange(true, std::memory_order_acq_rel)) {
        return;  // already installed; only the shutdown code was refreshed
    }

    // Suppress the OS critical-error / general-protection-fault dialogs so a
    // crash cannot block on a modal message box in a headless run.
    ::SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX |
                   SEM_NOOPENFILEERRORBOX);

    // Suppress the Windows Error Reporting UI. WerSetFlags lives in kernel32 on
    // current Windows; resolve it dynamically to avoid a hard link dependency
    // on wer.lib in the resolve_c build.
    if (HMODULE k32 = ::GetModuleHandleW(L"kernel32.dll")) {
        using WerSetFlagsFn = HRESULT(WINAPI*)(DWORD);
        // Route through void*: GetProcAddress returns FARPROC, and a direct
        // reinterpret_cast between two unrelated function-pointer types is what
        // GCC's -Wcast-function-type reports. Windows guarantees function
        // pointers and void* are interconvertible, so the two-step cast is the
        // portable spelling of the same conversion.
        if (auto wer_set_flags = reinterpret_cast<WerSetFlagsFn>(
                reinterpret_cast<void*>(::GetProcAddress(k32, "WerSetFlags")))) {
            constexpr DWORD kWerFaultReportingNoUi = 0x0020;  // WER_FAULT_REPORTING_NO_UI
            wer_set_flags(kWerFaultReportingNoUi);
        }
    }

    ::SetUnhandledExceptionFilter(&resolve_unhandled_exception_filter);

    // First-in-line vectored handler so a teardown-window fault (issue #18),
    // which the unhandled-exception filter above never sees, is still caught.
    ::AddVectoredExceptionHandler(1UL, &resolve_vectored_handler);
}

#else  // !_WIN32

void install_crash_handler(int shutdown_exit_code) {
    // No JIT-debugger / WER teardown pathology off Windows; record the code so
    // crash_exit_code stays consistent if anything queries it.
    g_shutdown_exit_code.store(shutdown_exit_code, std::memory_order_release);
    g_handler_installed.store(true, std::memory_order_release);
}

#endif  // _WIN32

void signal_work_complete() {
    g_work_complete.store(true, std::memory_order_release);
}

}  // namespace resolve
