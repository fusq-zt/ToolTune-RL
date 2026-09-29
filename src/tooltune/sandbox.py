"""Isolated Python execution: namespaces, calculator policy and kernel limits.

There is no fallback to unsandboxed execution. A broken sandbox is infrastructure
failure, never a model error. Only read-only OS files enter the namespace.
"""

import os
from pathlib import Path
import signal
import shutil
import subprocess
from tooltune.tools.python_policy import _WRAPPER

BWRAP = os.environ.get("TOOLTUNE_BWRAP") or shutil.which("bwrap") or ""


def command():
    if not BWRAP or not Path(BWRAP).is_file():
        raise RuntimeError("Install bubblewrap or set TOOLTUNE_BWRAP on Linux")
    args = [
        BWRAP,
        "--unshare-all",
        "--new-session",
        "--die-with-parent",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/lib",
        "/lib",
        "--dir",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/work",
        "--chdir",
        "/work",
        "--clearenv",
        "--setenv",
        "PATH",
        "/usr/bin",
        "--setenv",
        "LANG",
        "C.UTF-8",
    ]
    if Path("/lib64").exists():
        args.extend(["--ro-bind", "/lib64", "/lib64"])
    return args


PRELUDE = """
import resource, sys
resource.setrlimit(resource.RLIMIT_NPROC, (8, 8))
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
class BoundedWriter:
    def __init__(self, sink): self.sink, self.n = sink, 0
    def write(self, s):
        self.n += len(s)
        if self.n > 16000: raise RuntimeError("OutputLimitExceeded")
        return self.sink.write(s)
    def flush(self): self.sink.flush()
sys.stdout = BoundedWriter(sys.stdout)
sys.stderr = BoundedWriter(sys.stderr)
"""


def preflight():
    probe = (
        "import os,socket; "
        'assert not os.path.exists("/root"); '
        'assert not os.path.exists("/home"); '
        'assert not os.environ.get("HF_TOKEN"); '
        'assert all(name == "lo" for _,name in socket.if_nameindex()); '
        'print("isolated: only loopback, no host files or environment")'
    )
    run = subprocess.run(
        command() + ["/usr/bin/python3", "-I", "-S", "-c", probe],
        capture_output=True,
        text=True,
        timeout=8,
    )
    if run.returncode != 0:
        raise RuntimeError("Sandbox preflight failed: " + run.stderr)
    return {
        "isolated_network": run.stdout.strip(),
        "workspace_hidden": True,
        "proc_unmounted": True,
    }


def execute(code):
    if not Path(BWRAP).exists():
        raise RuntimeError("Required bubblewrap is unavailable")
    args = command() + ["/usr/bin/python3", "-I", "-S", "-c", PRELUDE + _WRAPPER]
    process = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(code, timeout=4)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return {"ok": False, "output": "PYTHON_ERROR: timeout", "status": "timeout"}
    if "bwrap:" in stderr:
        raise RuntimeError("Sandbox infrastructure error: " + stderr[:1000])
    return {
        "ok": process.returncode == 0,
        "output": (stdout if process.returncode == 0 else "PYTHON_ERROR: " + stderr)[
            :8000
        ],
        "status": "ok" if process.returncode == 0 else "execution_error",
    }
