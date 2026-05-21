"""Optional wandb shim.

When ``wandb`` is installed, this module re-exports it transparently along
with the ``trace_tree`` submodule that ``moralsim`` relies on.

When ``wandb`` is NOT installed, it exposes a no-op stub API that supports
exactly the call surface used by ``moralsim``. This lets ``debug=true`` runs
execute without requiring the optional ``wandb`` dependency.

Usage:

    from moralsim.utils.wandb_compat import wandb, trace_tree, HAS_WANDB
"""

from __future__ import annotations

try:
    import wandb as _wandb
    from wandb.sdk.data_types import trace_tree as _trace_tree

    HAS_WANDB = True
    wandb = _wandb
    trace_tree = _trace_tree

except Exception:
    import uuid as _uuid

    HAS_WANDB = False

    class _NoopSpan:
        def __init__(self, name: str = "", start_time_ms: float = 0.0) -> None:
            self.name = name
            self.start_time_ms = start_time_ms
            self.end_time_ms = 0.0
            self.status_code = None
            self.status_message = None
            self.child_spans: list = []

        def add_named_result(self, inputs=None, outputs=None):
            return None

    class _NoopTrace:
        def __init__(
            self,
            name: str = "",
            kind=None,
            start_time_ms: float = 0.0,
            end_time_ms: float = 0.0,
            status_code=None,
            status_message=None,
            metadata=None,
            inputs=None,
            outputs=None,
        ) -> None:
            self.name = name
            self._span = _NoopSpan(name=name, start_time_ms=start_time_ms)
            self._model_dict: dict = {}

        def add_child(self, child) -> None:
            self._span.child_spans.append(child)

    class _NoopSpanKind:
        AGENT = "agent"
        CHAIN = "chain"
        LLM = "llm"

    class _NoopWBTraceTree:
        def __init__(self, *_a, **_kw) -> None:
            pass

    class _NoopRun:
        def __init__(self) -> None:
            self.id = _uuid.uuid4().hex[:8]
            self.name = f"local_{self.id}"

        def log_artifact(self, *_a, **_kw):
            return None

    class _NoopArtifact:
        def __init__(self, *_a, **_kw) -> None:
            pass

        def add_dir(self, *_a, **_kw):
            return None

        def add_file(self, *_a, **_kw):
            return None

    class _NoopWandb:
        Artifact = _NoopArtifact

        def __init__(self) -> None:
            self.run = _NoopRun()

        def init(self, **_kw):
            self.run = _NoopRun()
            return self.run

        def log(self, *_a, **_kw):
            return None

        def save(self, *_a, **_kw):
            return None

    class _NoopTraceTreeModule:
        Trace = _NoopTrace
        SpanKind = _NoopSpanKind
        WBTraceTree = _NoopWBTraceTree

    wandb = _NoopWandb()
    trace_tree = _NoopTraceTreeModule()
