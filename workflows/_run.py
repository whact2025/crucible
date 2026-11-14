# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Workflow run helper: open a corpus run folder and tee stdio.

Every workflow's ``main()`` should wrap its body in
``open_run_with_capture(corpus, name)``. The helper:

1. Opens ``corpus.open_run(name)`` (a fresh ``runs/<ts>_<name>_<tag>/``).
2. Tees ``sys.stdout`` and ``sys.stderr`` into
   ``<run_dir>/stdout.log`` and ``<run_dir>/stderr.log`` for the
   duration of the workflow. Original streams continue to write to
   the terminal — operators see live output AND the run folder
   captures it for postmortem.

What gets captured:

- Python's ``logging`` output (the ``logging`` module writes to
  ``sys.stderr`` by default — adapter retry / classification
  warnings land here, as do any ``logger.info`` / ``logger.error``
  calls workflows make).
- Anything else that writes to ``sys.stdout`` / ``sys.stderr`` from
  Python code in this process. Workflows themselves do NOT call
  ``print()``: the final summary is written as
  ``<run_dir>/summary.json`` and the terminal status is recorded in
  ``<run_dir>/result.json`` via ``handle.finalize(...)``.

What does NOT get captured (by design — these have their own files):

- Build / verify subprocess output → immutable
  ``<run_dir>/{build,verify}_<timestamp>_<tag>.log`` attempts plus atomic
  ``build.log`` / ``verify.log`` latest views written by ``_build_verify``.
- Bench subprocess output → ``<run_dir>/bench_<id>.log`` written by
  ``assay.bench``.
- Backend SDK transcripts + per-invocation debug →
  ``<run_dir>/sdk_<seq>_<backend>.json`` written by
  ``crucible.backends._sdk_debug.SdkDebugSink``, with complete redacted result
  diagnostics under ``<run_dir>/sdk_artifacts/``. Both are diagnostic-only;
  typed workflow records remain semantic parse authority.
- SDK CLI subprocess output that the SDK spawns and never surfaces
  to Python — that bypasses ``sys.stdout`` entirely. Wrap the whole
  workflow invocation in shell ``tee`` if you need that level of
  capture.

The tee uses Python-level stream substitution (``sys.stdout =
TeeStream(...)``), not ``os.dup2`` on fd 1/2. This is deliberate:
fd-level redirection would also swallow subprocess inherited stdio,
losing live terminal feedback. The Python-level swap preserves
terminal output while still capturing the workflow's own writes.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import secrets
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Callable, Iterator

from assay.corpus.runs import RunResult


logger = logging.getLogger(__name__)
_PROVENANCE_ROOT_UNSET = object()
_FAILURE_TERMINAL_RESERVED_SUMMARY_FIELDS = frozenset(
    {
        "status",
        "workflow",
        "cost_usd",
        "error_kind",
        "error_message",
        "partial_summary",
    }
)


@dataclass(frozen=True, slots=True)
class FailureTerminal:
    """Typed exception terminal selected after nested cleanup has completed.

    ``open_run_with_capture`` continues to own the exception identity, error
    fields, measured duration, and cost callback.  A composing workflow may
    narrow only the detail status and add non-conflicting summary evidence.
    This lets a parent preserve an ``incomplete_accounting`` terminal after
    child cleanup without racing the generic exception publisher.
    """

    status: str
    summary_fields: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.status, str)
            or not self.status
            or self.status != self.status.strip()
            or len(self.status) > 128
        ):
            raise ValueError(
                "FailureTerminal.status must be a non-empty canonical string"
            )
        # This selector runs only while an exception is propagating.  Letting
        # it publish a success/no-work status would make the typed outcome
        # contradict the still-raised exception and could authorize dependent
        # work.  Import lazily so _run remains usable during module startup.
        from ._outcome import TerminalClass, classify_terminal_status

        if classify_terminal_status(self.status) in {
            TerminalClass.SUCCEEDED,
            TerminalClass.NO_WORK,
        }:
            raise ValueError(
                "FailureTerminal.status must classify as a non-success terminal"
            )
        if not isinstance(self.summary_fields, Mapping):
            raise TypeError("FailureTerminal.summary_fields must be a mapping")
        if any(not isinstance(key, str) or not key for key in self.summary_fields):
            raise TypeError(
                "FailureTerminal.summary_fields keys must be non-empty strings"
            )
        conflicts = sorted(
            set(self.summary_fields) & _FAILURE_TERMINAL_RESERVED_SUMMARY_FIELDS
        )
        if conflicts:
            raise ValueError(
                "FailureTerminal.summary_fields cannot replace generic "
                f"terminal fields: {conflicts!r}"
            )


class _TeeStream:
    """A file-like that fans writes out to several underlying streams.

    Used to replace ``sys.stdout`` and ``sys.stderr`` so the workflow's
    output reaches both the terminal and a log file.
    """

    def __init__(self, *streams: IO[str]) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        written = 0
        for s in self._streams:
            try:
                n = s.write(data)
                if isinstance(n, int):
                    written = n
            except Exception:  # noqa: BLE001 — one stream failing must not break others
                pass
        # Return the length of data so callers tracking writes don't
        # see zero (which can confuse some loggers' buffer accounting).
        return written or len(data)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:  # noqa: BLE001
                pass

    def isatty(self) -> bool:
        # Conservative: claim non-TTY so callers don't try ANSI escapes
        # destined for the log file. The original stdout/stderr that
        # the terminal user reads is still a TTY independently.
        return False

    def fileno(self) -> int:
        # Some libraries (e.g. ``rich``) ask for a fileno; refusing
        # is the safe answer — they fall back to non-fd-aware paths.
        raise OSError(
            "TeeStream has no single file descriptor; "
            "Python-level callers should write via .write() instead"
        )

    @property
    def encoding(self) -> str:
        # Prefer the encoding of the first underlying stream that
        # exposes one; default to utf-8.
        for s in self._streams:
            enc = getattr(s, "encoding", None)
            if enc:
                return enc
        return "utf-8"

    def writable(self) -> bool:
        return True


class _LiveStderrHandler(logging.Handler):
    """Logging handler that resolves ``sys.stderr`` on each emit.

    Standard ``logging.StreamHandler`` captures the stream reference
    at construction time, so a handler created before
    ``capture_to_run_dir`` swaps ``sys.stderr`` keeps writing to the
    original. This subclass reads the current ``sys.stderr`` per
    record, so it always lands in whichever tee is active.

    Nested captures install one handler each, but the innermost
    ``_TeeStream`` already fans a write through every active capture.
    A private marker on the shared ``LogRecord`` therefore lets the
    first live handler emit once while sibling live handlers skip it.
    Other handler classes neither consult nor inherit this marker.
    """

    _EMITTED_MARKER = "_crucible_live_stderr_emitted"
    _marker_lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        # ``Handler.handle`` serializes one handler instance, not sibling
        # instances installed by nested captures. Claim the record under a
        # shared lock before formatting or writing so two threads dispatching
        # the same record cannot both emit it.
        with self._marker_lock:
            if record.__dict__.get(self._EMITTED_MARKER, False):
                return
            record.__dict__[self._EMITTED_MARKER] = True
        try:
            msg = self.format(record)
            stream = sys.stderr
            stream.write(msg + "\n")
            try:
                stream.flush()
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            self.handleError(record)


def write_atomic_bytes(path: Path, data: bytes) -> None:
    """Durably replace a private workflow artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (
        f".tmp_{path.name}_{os.getpid()}_{secrets.token_hex(4)}"
    )
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some filesystems do not support fsync on directory handles.
            pass
    finally:
        temporary.unlink(missing_ok=True)


def write_atomic_json(path: Path, payload: object) -> None:
    """Durably replace a standalone private JSON document.

    Run-owned summaries should use :meth:`RunHandle.write_summary`.  This
    helper is for workflows, such as DeepSWE, that intentionally manage a
    run directory without a ``RunHandle``.  The temporary file is created in
    the destination directory with mode ``0600``, fsynced, atomically renamed,
    and followed by a best-effort directory fsync.
    """

    data = json.dumps(payload, indent=2, default=str).encode("utf-8")
    write_atomic_bytes(Path(path), data)


@contextlib.contextmanager
def capture_to_run_dir(run_dir: Path) -> Iterator[None]:
    """Tee sys.stdout and sys.stderr into ``<run_dir>/{stdout,stderr}.log``.

    Also installs a root-logger handler at INFO level so workflow
    ``logger.info(...)`` / ``logger.error(...)`` calls land in
    ``stderr.log`` (via the tee). The handler resolves ``sys.stderr``
    per record, so it correctly follows the swapped stream.

    Original streams + logger config are restored on exit.
    Line-buffered so ``tail -f <run_dir>/stdout.log`` shows live
    output. Each entry truncates the log — one run = one folder.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Prompts, transcripts, source excerpts, and subprocess output can carry
    # credentials or proprietary code.  Run folders are private by default.
    run_dir.chmod(0o700)
    out_path = run_dir / "stdout.log"
    err_path = run_dir / "stderr.log"
    # ``buffering=1`` = line buffering (text mode) — flushes on every
    # newline so the log is live-tail-able.
    out_fh = open(out_path, "w", encoding="utf-8", buffering=1)
    err_fh = open(err_path, "w", encoding="utf-8", buffering=1)
    out_path.chmod(0o600)
    err_path.chmod(0o600)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = _TeeStream(orig_stdout, out_fh)  # type: ignore[assignment]
    sys.stderr = _TeeStream(orig_stderr, err_fh)  # type: ignore[assignment]

    # Hook the root logger so workflow logging.info / .error reach
    # stderr.log. Level snapshot is restored on exit.
    root = logging.getLogger()
    saved_level = root.level
    handler = _LiveStderrHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)

    try:
        yield
    finally:
        # Restore BEFORE closing the files so any error during close
        # still has somewhere to print.
        try:
            root.removeHandler(handler)
            root.setLevel(saved_level)
        except Exception:  # noqa: BLE001
            pass
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        try:
            out_fh.flush()
            out_fh.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            err_fh.flush()
            err_fh.close()
        except Exception:  # noqa: BLE001
            pass


@contextlib.contextmanager
def open_run_with_capture(
    corpus,
    workflow_name: str,
    *,
    provenance_root: Path | str | None | object = _PROVENANCE_ROOT_UNSET,
    failure_cost_usd: Callable[[], float] | None = None,
    failure_terminal: (Callable[[BaseException], FailureTerminal | None] | None) = None,
):
    """Open ``corpus.open_run(workflow_name)`` and capture stdio.

    Yields the ``RunHandle`` so callers can write events, finalize
    with a ``RunResult``, and pass ``handle.run_dir`` to the backend
    adapter as ``log_dir`` for per-invocation SDK debug dumps. A parent
    workflow that accumulates durable child costs may provide
    ``failure_cost_usd`` so an exception after child completion retains that
    known lower bound in its terminal envelope. ``failure_terminal`` runs only
    after nested context managers and rollback callbacks have unwound. It may
    select a typed failure status and additive summary evidence without
    replacing the original exception identity or generic terminal fields.

    Typical workflow usage::

        def main(argv):
            args = _parse_args(argv)
            corpus = Corpus(args.spec)
            spec = corpus.latest_spec()
            if spec is None:
                raise SystemExit(...)

            with open_run_with_capture(corpus, "breach") as handle:
                backend = ClaudeBackend(
                    model=args.model, log_dir=handle.run_dir,
                )
                ...
                # Final summary lands as a structured artifact, not stdout.
                handle.write_summary(summary)
                handle.finalize(RunResult(status=..., cost_usd=..., duration_s=...))
            return rc
    """
    # Reconcile only runs whose exact PID/start-token owner disappeared.  This
    # is safe in the presence of concurrent live workflows and closes the
    # SIGKILL/OOM observability gap before a new run starts.
    from . import _outcome

    _outcome.reconcile_stale_runs(corpus)
    starting_spec = corpus.latest_spec()
    handle = corpus.open_run(workflow_name)
    handle.run_dir.chmod(0o700)
    begin_kwargs = {}
    if provenance_root is not _PROVENANCE_ROOT_UNSET:
        begin_kwargs["provenance_root"] = provenance_root
    _outcome.begin_run(
        handle.run_dir,
        workflow=workflow_name,
        spec=starting_spec,
        **begin_kwargs,
    )
    # Own finalization here rather than delegating exception fallback to
    # RunHandle.__exit__.  A composing workflow may need a typed debt terminal;
    # if that terminal cannot be persisted, absence is safer than an invented
    # exact-looking ``failed/$0`` result.
    body_failed = False
    body_exception: BaseException | None = None
    exception_terminal_published = False
    try:
        capture_stack = contextlib.ExitStack()
        try:
            capture_stack.enter_context(capture_to_run_dir(handle.run_dir))
            yield handle
            capture_stack.close()
        except BaseException as exc:
            try:
                capture_stack.close()
            except BaseException as capture_cleanup_exc:  # noqa: BLE001
                exc.add_note(
                    "secondary capture cleanup failure: "
                    f"{type(capture_cleanup_exc).__name__}: "
                    f"{str(capture_cleanup_exc)[:1000]}"
                )
            body_failed = True
            body_exception = exc
            # Publish one coherent terminal envelope before RunHandle's
            # outer context commits result.json. Preserve an earlier
            # checkpoint as diagnostics, but never pair a stale success
            # summary with a failed terminal result.
            if not handle.finalized:
                failure_cost = 0.0
                cost_accounting_error = None
                if failure_cost_usd is not None:
                    try:
                        supplied_cost = failure_cost_usd()
                        if not isinstance(supplied_cost, (int, float)) or isinstance(
                            supplied_cost, bool
                        ):
                            raise TypeError("failure cost must be a real number")
                        failure_cost = float(supplied_cost)
                        if not math.isfinite(failure_cost) or failure_cost < 0.0:
                            raise ValueError(
                                "failure cost must be finite and non-negative"
                            )
                    except BaseException as cost_exc:  # noqa: BLE001
                        failure_cost = 0.0
                        cost_accounting_error = (
                            f"{type(cost_exc).__name__}: {str(cost_exc)[:1000]}"
                        )
                        logger.exception(
                            "could not resolve failure cost for %s",
                            workflow_name,
                        )
                terminal_status = "failed"
                terminal_summary_fields: dict[str, object] = {}
                failure_terminal_error = None
                if failure_terminal is not None:
                    try:
                        supplied_terminal = failure_terminal(exc)
                        if supplied_terminal is not None and not isinstance(
                            supplied_terminal,
                            FailureTerminal,
                        ):
                            raise TypeError(
                                "failure_terminal must return FailureTerminal or None"
                            )
                        if supplied_terminal is not None:
                            terminal_status = supplied_terminal.status
                            terminal_summary_fields = dict(
                                supplied_terminal.summary_fields
                            )
                    except BaseException as terminal_exc:  # noqa: BLE001
                        # The callback exists specifically because the
                        # caller may own paid child evidence. Failure to
                        # resolve it cannot safely degrade to an exact-
                        # looking ordinary failure.
                        terminal_status = "incomplete_accounting"
                        failure_terminal_error = (
                            f"{type(terminal_exc).__name__}: {str(terminal_exc)[:1000]}"
                        )
                        terminal_summary_fields = {
                            "cost_usd_is_lower_bound": True,
                            "accounting_debt": True,
                            "accounting_debt_by_gate": {
                                "failure_terminal": [
                                    "cost_complete",
                                    "token_usage_complete",
                                ],
                            },
                            "failure_terminal_error": (failure_terminal_error),
                        }
                        logger.exception(
                            "could not resolve failure terminal for %s",
                            workflow_name,
                        )
                if cost_accounting_error is not None:
                    # The numeric zero is only a lower bound when the
                    # workflow's cost ledger cannot be read.  This
                    # takes precedence over any exact-looking caller
                    # terminal and composes with its existing debt.
                    terminal_status = "incomplete_accounting"
                    debt_by_gate: dict[str, set[str]] = {}
                    supplied_debt = terminal_summary_fields.get(
                        "accounting_debt_by_gate"
                    )
                    if isinstance(supplied_debt, Mapping):
                        for gate, dimensions in supplied_debt.items():
                            if isinstance(gate, str) and isinstance(
                                dimensions, (list, tuple, set)
                            ):
                                debt_by_gate[gate] = {
                                    str(dimension) for dimension in dimensions
                                }
                    debt_by_gate["failure_cost_usd"] = {
                        "cost_complete",
                        "token_usage_complete",
                    }
                    terminal_summary_fields.update(
                        {
                            "cost_usd_is_lower_bound": True,
                            "accounting_debt": True,
                            "accounting_debt_by_gate": {
                                gate: sorted(dimensions)
                                for gate, dimensions in sorted(debt_by_gate.items())
                            },
                        }
                    )
                previous = None
                summary_path = handle.run_dir / "summary.json"
                if summary_path.is_file():
                    try:
                        previous = json.loads(summary_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        previous = {"unreadable": True}
                terminal_error_kind = type(exc).__name__
                terminal_error_message = str(exc)[:2000]
                failure_summary = {
                    "status": terminal_status,
                    "workflow": workflow_name,
                    "cost_usd": failure_cost,
                    "error_kind": terminal_error_kind,
                    "error_message": terminal_error_message,
                    **terminal_summary_fields,
                }
                if cost_accounting_error is not None:
                    failure_summary["cost_accounting_error"] = cost_accounting_error
                if previous is not None:
                    failure_summary["partial_summary"] = previous
                summary_error: BaseException | None = None
                for _attempt in range(2):
                    try:
                        handle.write_summary(failure_summary)
                    except BaseException as publication_exc:  # noqa: BLE001
                        summary_error = publication_exc
                        continue
                    summary_error = None
                    break
                if summary_error is not None:
                    logger.exception(
                        "could not persist failure summary for %s",
                        workflow_name,
                        exc_info=(
                            type(summary_error),
                            summary_error,
                            summary_error.__traceback__,
                        ),
                    )
                result_error: BaseException | None = None
                for _attempt in range(2):
                    try:
                        handle.finalize_elapsed(
                            RunResult(
                                status=terminal_status,
                                cost_usd=failure_cost,
                                error_kind=terminal_error_kind,
                                error_message=terminal_error_message,
                            )
                        )
                    except BaseException as publication_exc:  # noqa: BLE001
                        result_error = publication_exc
                        continue
                    result_error = None
                    break
                if result_error is not None and not handle.finalized:
                    # A test double or filesystem seam may fail the
                    # measured helper before it reaches create-once.
                    # Fall back to the same typed terminal with one
                    # locally measured duration; never ask RunHandle's
                    # generic __exit__ to invent an ordinary failure.
                    measured_duration_s = handle.elapsed_s
                    for _attempt in range(2):
                        try:
                            handle.finalize(
                                RunResult(
                                    status=terminal_status,
                                    cost_usd=failure_cost,
                                    duration_s=measured_duration_s,
                                    error_kind=terminal_error_kind,
                                    error_message=terminal_error_message,
                                )
                            )
                        except BaseException as publication_exc:  # noqa: BLE001
                            result_error = publication_exc
                            continue
                        result_error = None
                        break
                if result_error is None and handle.finalized:
                    exception_terminal_published = True
                if result_error is not None and not handle.finalized:
                    logger.exception(
                        "could not persist failure result for %s",
                        workflow_name,
                        exc_info=(
                            type(result_error),
                            result_error,
                            result_error.__traceback__,
                        ),
                    )
            raise
        else:
            if not handle.finalized:
                handle.finalize_elapsed(RunResult(status="completed"))
    finally:
        # Project only a durably published RunResult.  On an exceptional path,
        # never replace a persistently unpublishable typed terminal with a
        # contradictory generic outcome.
        if handle.finalized:
            try:
                if body_failed and not exception_terminal_published:
                    if body_exception is None:
                        raise RuntimeError(
                            "failed run lost its exception identity"
                        )
                    _outcome._finalize_uncommitted_result_as_incomplete(
                        handle.run_dir,
                        cause=body_exception,
                    )
                else:
                    latest = corpus.latest_spec()
                    _outcome.finalize_run(
                        handle.run_dir,
                        spec_version=(
                            latest.version if latest is not None else None
                        ),
                        trust_summary=not body_failed,
                    )
            except BaseException as outcome_error:  # noqa: BLE001
                if not body_failed:
                    raise
                if body_exception is not None:
                    body_exception.add_note(
                        "secondary typed-outcome publication failure: "
                        f"{type(outcome_error).__name__}: "
                        f"{str(outcome_error)[:1000]}"
                    )
                logger.exception(
                    "could not publish typed outcome for %s", workflow_name
                )


def record_handled_terminal(
    corpus,
    workflow_name: str,
    *,
    status: str,
    message: str,
    error_kind: str | None = None,
    extra: dict | None = None,
    duration_s: float | None = None,
    provenance_root: Path | str | None | object = _PROVENANCE_ROOT_UNSET,
) -> None:
    """Persist a summary/result pair for a handled preflight terminal."""

    open_kwargs = {}
    if provenance_root is not _PROVENANCE_ROOT_UNSET:
        open_kwargs["provenance_root"] = provenance_root
    with open_run_with_capture(
        corpus,
        workflow_name,
        **open_kwargs,
    ) as handle:
        terminal_duration_s = (
            handle.elapsed_s if duration_s is None else max(0.0, duration_s)
        )
        summary = {
            "status": status,
            "workflow": workflow_name,
            "message": message,
            "duration_s": terminal_duration_s,
        }
        if error_kind is not None:
            summary["error_kind"] = error_kind
        if extra:
            summary.update(extra)
        handle.write_summary(summary)
        handle.finalize(
            RunResult(
                status=status,
                duration_s=terminal_duration_s,
                error_kind=error_kind,
                error_message=message,
            )
        )


__all__ = [
    "FailureTerminal",
    "capture_to_run_dir",
    "open_run_with_capture",
    "record_handled_terminal",
    "write_atomic_bytes",
    "write_atomic_json",
    "exec_ctx_now",
    "read_assay_version",
    "register_subprocess_cleanup",
]


_CLEANUP_REGISTERED = False


def register_subprocess_cleanup(*, grace_seconds: float = 5.0) -> None:
    """Register handlers that terminate descendant processes on exit.

    This is a best-effort process-exit backstop for workflow-owned
    descendants, including selected backend CLI processes. Normal invocation
    cancellation and mutation settlement retain their own ownership and
    proof contracts; this helper does not establish quiescence authority.

    This helper registers two cleanup paths:

    1. ``atexit`` — runs on normal process exit (returns from
       ``main()``, ``SystemExit``, unhandled exception
       propagating to the interpreter).
    2. Handlers for ``SIGTERM``, ``SIGINT``, and ``SIGHUP`` where the process
       has a default or ignored handler. Cleanup then re-raises the signal
       so the process exits with the conventional status (128 + signum).
       Existing custom handlers remain intact.

    Cleanup walks ``psutil.Process(os.getpid()).children(recursive=True)``,
    sends SIGTERM to each, waits up to ``grace_seconds``, then
    SIGKILLs survivors. Cleanup exceptions are swallowed during teardown;
    missing optional ``psutil`` support leaves this backstop inactive.

    ``SIGKILL`` bypasses both paths. Detached or already-reparented
    processes also lie outside this descendant snapshot. Exact workflow
    settlement must account for its admitted process and container boundary.

    **Idempotent**: safe to call multiple times; subsequent calls are
    no-ops. ``psutil`` supplies descendant enumeration and bounded waits.
    """
    global _CLEANUP_REGISTERED
    if _CLEANUP_REGISTERED:
        return
    _CLEANUP_REGISTERED = True

    import atexit
    import os
    import signal

    def _kill_descendants() -> None:
        try:
            import psutil
        except ImportError:
            return
        try:
            me = psutil.Process(os.getpid())
        except psutil.NoSuchProcess:
            return
        try:
            children = me.children(recursive=True)
        except psutil.NoSuchProcess:
            return
        if not children:
            return
        for c in children:
            try:
                c.terminate()
            except psutil.NoSuchProcess:
                pass
            except Exception:  # noqa: BLE001 — best-effort
                pass
        try:
            _gone, alive = psutil.wait_procs(
                children, timeout=grace_seconds,
            )
        except Exception:  # noqa: BLE001
            alive = children
        for c in alive:
            try:
                c.kill()
            except psutil.NoSuchProcess:
                pass
            except Exception:  # noqa: BLE001
                pass

    atexit.register(_kill_descendants)

    # Signal handler: run cleanup, restore default handler,
    # re-raise so the process exits with the conventional status
    # code (128 + signum) and the shell sees the right reason.
    def _handler(signum, _frame):
        _kill_descendants()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    # Only override if the user hasn't already set a custom handler.
    # SIGTERM + SIGINT are the two we care about. SIGHUP is also
    # relevant for terminal-detach scenarios — include it.
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            current = signal.getsignal(sig)
            if current in (signal.SIG_DFL, signal.SIG_IGN, None):
                signal.signal(sig, _handler)
        except (ValueError, OSError):
            # ValueError when not in main thread; OSError on systems
            # without the signal. Both are fine to skip.
            pass


# ----------------------------------------------------------------------
# Shared helpers for ExecutionContext stamping
# ----------------------------------------------------------------------
#
# Workflows share these helpers when stamping ExecutionContext values for
# records such as lifecycle events.


def read_assay_version() -> str:
    """Return the installed ``assay`` package version, or ``"0.0.0"``.

    Best-effort: returns the literal string when ``importlib.metadata``
    can't resolve the package (e.g., in an editable-install layout
    where setuptools hasn't laid down the metadata).
    """
    try:
        from importlib.metadata import version

        return version("assay")
    except Exception:  # noqa: BLE001
        return "0.0.0"


def exec_ctx_now(workflow: str, run_id: str | None = None):
    """Construct an ``ExecutionContext`` snapshot for the current process.

    ``workflow``: short identifier of the calling workflow
    (``"architect"``, ``"breach"``, ``"goal"``, etc.). Stored on
    ``ExecutionContext.recipe`` so downstream postmortem tooling can
    trace which workflow produced an artifact.

    Imports ``ExecutionContext`` lazily so this module's top-level
    has no assay dependency (avoids a circular-import path when assay
    types are extended).
    """
    import datetime as _datetime
    import os
    import platform
    import socket

    from assay import ExecutionContext

    return ExecutionContext(
        timestamp=_datetime.datetime.now(
            _datetime.timezone.utc
        ).isoformat(timespec="seconds"),
        machine=socket.gethostname(),
        cwd=os.getcwd(),
        toolchain={
            "assay": read_assay_version(),
            "python": platform.python_version(),
        },
        backend=None,
        model=None,
        recipe=workflow,
        recipe_commit=None,
        gpu_model=None,
        run_id=run_id,
    )
