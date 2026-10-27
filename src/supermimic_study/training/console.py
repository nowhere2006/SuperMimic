"""Tee Python and native-library stdout/stderr into a per-run log on macOS/Linux."""
from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import sys
import threading


def _flush():
    sys.stdout.flush()
    sys.stderr.flush()
    # Native MuJoCo/Warp warnings may be buffered by libc rather than Python.
    ctypes.CDLL(None).fflush(None)


@contextmanager
def capture_console(path):
    if os.name != "posix":
        raise RuntimeError("native console capture currently supports macOS/Linux")
    _flush()
    terminal_out, terminal_err = os.dup(1),os.dup(2)
    read_fd,write_fd = os.pipe()
    failures = []
    log = Path(path).open("ab",buffering=0)

    def drain():
        try:
            while True:
                chunk = os.read(read_fd,65536)
                if not chunk:
                    break
                if not failures:
                    try:
                        log.write(chunk)
                    except OSError as exc:
                        failures.append(exc)  # Keep draining the pipe to avoid deadlock.
                try:
                    os.write(terminal_out,chunk)
                except BrokenPipeError:
                    pass  # A closed terminal must not discard the file log.
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=drain,daemon=True)
    thread.start()
    os.dup2(write_fd,1)
    os.dup2(write_fd,2)
    os.close(write_fd)
    try:
        yield
    finally:
        _flush()
        os.dup2(terminal_out,1)
        os.dup2(terminal_err,2)
        thread.join()
        os.close(read_fd)
        os.close(terminal_out)
        os.close(terminal_err)
        log.close()
        if failures:
            raise RuntimeError("console log failed") from failures[0]
