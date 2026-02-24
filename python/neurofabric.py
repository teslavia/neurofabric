"""
neurofabric.py — Zero-dependency ctypes binding for libnf_c_api

Usage:
    from neurofabric import Engine, Session

    engine = Engine(n_threads=4)
    session = Session(engine, "model.nfir")
    session.step()
    print(f"{session.last_step_us():.1f} us")
"""

import ctypes
import os
import pathlib
import sys


def _find_library():
    """Locate nf_c_api shared library."""
    env = os.environ.get("NF_LIB_PATH")
    if env and os.path.isfile(env):
        return env

    here = pathlib.Path(__file__).resolve().parent
    if sys.platform == "darwin":
        name = "nf_c_api.dylib"
    elif sys.platform == "win32":
        name = "nf_c_api.dll"
    else:
        name = "nf_c_api.so"

    candidates = [
        here / ".." / "build" / "lib" / name,
        here / ".." / "build-rel" / "lib" / name,
        here / ".." / "build-debug" / "lib" / name,
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())

    raise FileNotFoundError(
        f"Cannot find {name}. Set NF_LIB_PATH or build the project first."
    )


class Engine:
    """Wraps nf_engine_t — owns a PipelineEngine + mock provider."""

    def __init__(self, n_threads=0, lib_path=None):
        path = lib_path or _find_library()
        self._lib = ctypes.CDLL(path)
        self._setup_prototypes()
        self._handle = self._lib.nf_create_engine(n_threads)
        if not self._handle:
            raise RuntimeError("nf_create_engine returned NULL")

    def _setup_prototypes(self):
        L = self._lib
        L.nf_create_engine.argtypes = [ctypes.c_uint32]
        L.nf_create_engine.restype = ctypes.c_void_p
        L.nf_destroy_engine.argtypes = [ctypes.c_void_p]
        L.nf_destroy_engine.restype = None

        L.nf_create_session.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.nf_create_session.restype = ctypes.c_void_p
        L.nf_destroy_session.argtypes = [ctypes.c_void_p]
        L.nf_destroy_session.restype = None

        L.nf_session_set_input.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64
        ]
        L.nf_session_set_input.restype = ctypes.c_int

        L.nf_session_step.argtypes = [ctypes.c_void_p]
        L.nf_session_step.restype = ctypes.c_int

        L.nf_session_get_output.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64
        ]
        L.nf_session_get_output.restype = ctypes.c_int

        L.nf_session_get_last_step_us.argtypes = [ctypes.c_void_p]
        L.nf_session_get_last_step_us.restype = ctypes.c_double

        L.nf_session_num_tensors.argtypes = [ctypes.c_void_p]
        L.nf_session_num_tensors.restype = ctypes.c_uint32

        L.nf_session_num_nodes.argtypes = [ctypes.c_void_p]
        L.nf_session_num_nodes.restype = ctypes.c_uint32

        L.nf_session_set_push_constants.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint32
        ]
        L.nf_session_set_push_constants.restype = ctypes.c_int

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.nf_destroy_engine(self._handle)
            self._handle = None


class Session:
    """Wraps nf_session_t — pre-compiled execution plan."""

    def __init__(self, engine, nfir_path):
        self._engine = engine
        self._lib = engine._lib
        path_bytes = str(nfir_path).encode("utf-8")
        self._handle = self._lib.nf_create_session(engine._handle, path_bytes)
        if not self._handle:
            raise RuntimeError(f"nf_create_session failed for {nfir_path}")

    def set_input(self, tensor_id, data):
        """Set input tensor data (bytes or buffer protocol object)."""
        buf = bytes(data)
        rc = self._lib.nf_session_set_input(
            self._handle, tensor_id, buf, len(buf)
        )
        if rc != 0:
            raise RuntimeError(f"nf_session_set_input failed: {rc}")

    def step(self):
        """Execute one inference step. Returns status code."""
        rc = self._lib.nf_session_step(self._handle)
        if rc != 0:
            raise RuntimeError(f"nf_session_step failed: {rc}")
        return rc

    def get_output(self, tensor_id, size):
        """Read output tensor data. Returns bytes."""
        buf = (ctypes.c_uint8 * size)()
        rc = self._lib.nf_session_get_output(
            self._handle, tensor_id, buf, size
        )
        if rc != 0:
            raise RuntimeError(f"nf_session_get_output failed: {rc}")
        return bytes(buf)

    def last_step_us(self):
        """Microseconds elapsed during last step()."""
        return self._lib.nf_session_get_last_step_us(self._handle)

    def num_tensors(self):
        return self._lib.nf_session_num_tensors(self._handle)

    def num_nodes(self):
        return self._lib.nf_session_num_nodes(self._handle)

    def set_push_constants(self, node_name, data):
        """Inject push constants into a named node (max 64 bytes)."""
        name_bytes = node_name.encode("utf-8") if isinstance(node_name, str) else node_name
        buf = bytes(data)
        rc = self._lib.nf_session_set_push_constants(
            self._handle, name_bytes, buf, len(buf))
        if rc != 0:
            raise RuntimeError(f"set_push_constants failed: {rc}")

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.nf_destroy_session(self._handle)
            self._handle = None


class Generator:
    """High-level text generation API using nf_generate CLI."""

    def __init__(self, model_path, fp16=False, paged=True):
        self.model_path = str(model_path)
        self.fp16 = fp16
        self.paged = paged
        self._bin = self._find_binary()

    def _find_binary(self):
        here = pathlib.Path(__file__).resolve().parent
        candidates = [
            here / ".." / "build" / "bin" / "nf_generate",
            here / ".." / "build-rel" / "bin" / "nf_generate",
        ]
        for c in candidates:
            if c.exists():
                return str(c.resolve())
        return "nf_generate"

    def _build_cmd(self, prompt, max_tokens=128, temperature=0.8):
        cmd = [self._bin, self.model_path, prompt,
               "--max-tokens", str(max_tokens),
               "--temperature", str(temperature)]
        if self.fp16:
            cmd.append("--fp16")
        if self.paged:
            cmd.append("--paged")
        return cmd

    def generate(self, prompt, max_tokens=128, temperature=0.8):
        """Generate text from a prompt. Returns the generated string."""
        import subprocess
        cmd = self._build_cmd(prompt, max_tokens, temperature)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.stdout

    def chat(self, messages, max_tokens=128, temperature=0.8, stream=False):
        """Chat-style generation. messages: list of dicts with 'role'/'content'.
        If stream=True, returns an iterator of tokens."""
        import subprocess
        prompt = messages[-1]["content"] if messages else ""
        cmd = self._build_cmd(prompt, max_tokens, temperature)
        cmd.extend(["--chat", "--chat-format", "chatml"])
        if stream:
            return self._stream(cmd)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.stdout

    def _stream(self, cmd):
        """Yield tokens as they are generated."""
        import subprocess
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            for line in proc.stdout:
                yield line
        finally:
            proc.terminate()
            proc.wait()
