# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Phase-aware exact pytest observer with a parent-owned supervisor.

This module intentionally has no Crucible imports. The outer runner embeds it
into the target Python environment. A parent supervisor owns the reserved
outer exit and checks it against the worker process outcome. The optional
failure envelope is only a bounded diagnostic hint: its pytest hook shares an
interpreter and report channel with generated test code, so consumers must not
treat it as authoritative provenance or acceptance evidence. The same
limitation applies to the generated test's reported semantics generally; this
adapter proves neither that its oracle is valid nor resistance to deliberately
hostile in-process harness tampering.
"""

# This source is embedded verbatim into benchmark environments. Keep it on
# the Python 3.6 syntax/runtime subset even though Crucible itself requires a
# newer interpreter; several supported SWE-bench testbeds still use 3.6.
import base64
import dis
import json
import os
import re
import subprocess
import sys


PASSED = 70
FAILED = 71
NOT_OBSERVED = 72
RUNNER_ERROR = 73

_FAILURE_SCHEMA = "crucible.pytest.failure.v1"
_MAX_RESULT_BYTES = 4096
_MAX_PATH_BYTES = 512
_MAX_SYMBOL_BYTES = 256
_MAX_LINE = 10_000_000
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_DOTTED_IDENTIFIER_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z"
)
_CODE_NAME_RE = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]*|<[A-Za-z_][A-Za-z0-9_ ]*>)\Z"
)

_expected_node_id = ""
_observed_nodes = set()
_status = None
_collection_failed = False
_collection_skipped = False
_result_fd = -1
_phase_failures = {}


def _same_node(observed, expected):
    observed_relative = observed[2:] if observed.startswith("./") else observed
    expected_relative = expected[2:] if expected.startswith("./") else expected
    return observed == expected or observed_relative == expected_relative


def _merge(status):
    global _status
    priority = {
        "passed": 0,
        "not_observed": 1,
        "failed": 2,
        "runner_error": 3,
    }
    if _status is None or priority[status] > priority[_status]:
        _status = status


def _bounded_text(value, *, limit):
    if type(value) is not str or not value:
        return None
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        return None
    if len(encoded) > limit or any(
        ord(character) < 0x20 or ord(character) == 0x7F
        for character in value
    ):
        return None
    return value


def _expected_test_identity():
    parts = _expected_node_id.split("::")
    if len(parts) != 2:
        return None
    path = _bounded_text(parts[0], limit=_MAX_PATH_BYTES)
    function = _bounded_text(parts[1], limit=_MAX_SYMBOL_BYTES)
    if (
        path is None
        or function is None
        or _IDENTIFIER_RE.fullmatch(function) is None
        or not function.startswith("test_")
    ):
        return None
    return path, function


def _safe_frame_path(filename):
    value = _bounded_text(filename, limit=_MAX_PATH_BYTES)
    if value is None:
        return None
    absolute = os.path.realpath(os.path.abspath(value))
    try:
        relative = os.path.relpath(absolute, os.getcwd())
    except (OSError, ValueError):
        relative = absolute
    if relative != os.pardir and not relative.startswith(os.pardir + os.sep):
        value = relative
    else:
        value = absolute
    return _bounded_text(value, limit=_MAX_PATH_BYTES)


def _same_file(filename, expected_path):
    if type(filename) is not str:
        return False
    try:
        return os.path.realpath(os.path.abspath(filename)) == os.path.realpath(
            os.path.abspath(expected_path)
        )
    except (OSError, ValueError):
        return False


def _traceback_frames(traceback):
    frames = []
    current = traceback
    while current is not None and len(frames) <= 1024:
        frames.append(current)
        current = current.tb_next
    if current is not None:
        return []
    return frames


def _direct_load_subject(frame, instructions, instruction_index):
    """Resolve only the named value loaded immediately before ``LOAD_ATTR``.

    Python 3.6-3.9 do not expose ``AttributeError.obj`` / ``.name``.  Their
    bytecode still proves a simple ``Mode.DEFAULT`` access: ``LOAD_GLOBAL``
    (or another named load) immediately followed by ``LOAD_ATTR``.  Reject
    calls, subscripts, chained attributes, and every other stack-producing
    expression because reconstructing those would require untrusted source or
    exception-message parsing.
    """

    if instruction_index <= 0:
        return None
    load = instructions[instruction_index - 1]
    name = load.argval
    if type(name) is not str or _IDENTIFIER_RE.fullmatch(name) is None:
        return None
    if load.opname in {"LOAD_FAST", "LOAD_DEREF", "LOAD_CLASSDEREF"}:
        return frame.f_locals.get(name)
    if load.opname == "LOAD_GLOBAL":
        if name in frame.f_globals:
            return frame.f_globals[name]
        return frame.f_builtins.get(name)
    if load.opname == "LOAD_NAME":
        if name in frame.f_locals:
            return frame.f_locals[name]
        if name in frame.f_globals:
            return frame.f_globals[name]
        return frame.f_builtins.get(name)
    return None


def _direct_class_attribute(
    exception,
    frames,
    *,
    expected_path,
    expected_function,
):
    """Return only bytecode-proven ``Type.missing`` information.

    ``AttributeError`` text is intentionally never parsed.  CPython records
    the missing owner and name as structured exception attributes.  The
    failing test frame must additionally be stopped on the matching
    ``LOAD_ATTR`` instruction, which proves that the owner/name pair was
    referenced directly by the exact generated test rather than raised by a
    product function called from that test.
    """

    if type(exception) is not AttributeError:
        return None

    test_frame = None
    for traceback in frames:
        code = traceback.tb_frame.f_code
        if (
            code.co_name == expected_function
            and _same_file(code.co_filename, expected_path)
        ):
            test_frame = traceback
    if test_frame is None:
        return None

    instructions = list(dis.get_instructions(test_frame.tb_frame.f_code))
    instruction_index = next(
        (
            index
            for index, candidate in enumerate(instructions)
            if candidate.offset == test_frame.tb_lasti
        ),
        None,
    )
    instruction = (
        instructions[instruction_index]
        if instruction_index is not None
        else None
    )
    attribute = getattr(exception, "name", None)
    if attribute is None and instruction is not None:
        attribute = instruction.argval
    attribute = _bounded_text(attribute, limit=_MAX_SYMBOL_BYTES)
    if (
        instruction is None
        or instruction.opname != "LOAD_ATTR"
        or attribute is None
        or _IDENTIFIER_RE.fullmatch(attribute) is None
        or instruction.argval != attribute
    ):
        return None
    subject = getattr(exception, "obj", None)
    if subject is None:
        subject = _direct_load_subject(
            test_frame.tb_frame,
            instructions,
            instruction_index,
        )
    if not isinstance(subject, type):
        return None
    positions = getattr(instruction, "positions", None)
    if positions is not None:
        start_line = positions.lineno
        end_line = positions.end_lineno
        if (
            type(start_line) is not int
            or type(end_line) is not int
            or not start_line <= test_frame.tb_lineno <= end_line
        ):
            return None

    try:
        subject_module_value = type.__getattribute__(subject, "__module__")
        subject_name_value = type.__getattribute__(subject, "__qualname__")
    except (AttributeError, TypeError):
        return None
    subject_module = _bounded_text(
        subject_module_value, limit=_MAX_SYMBOL_BYTES
    )
    subject_name = _bounded_text(
        subject_name_value, limit=_MAX_SYMBOL_BYTES
    )
    if (
        subject_module is None
        or subject_name is None
        or _DOTTED_IDENTIFIER_RE.fullmatch(subject_module) is None
        or _DOTTED_IDENTIFIER_RE.fullmatch(subject_name) is None
    ):
        return None
    return {
        "subject_kind": "type",
        "subject_module": subject_module,
        "subject_name": subject_name,
        "attribute": attribute,
        "direct_source_reference": True,
    }


def _failure_metadata(item, call):
    phase = call.when
    if phase not in {"call", "setup", "teardown"} or call.excinfo is None:
        return None
    if not _same_node(str(item.nodeid), _expected_node_id):
        return None
    identity = _expected_test_identity()
    if identity is None:
        return None
    expected_path, expected_function = identity

    exception = call.excinfo.value
    exception_type = _bounded_text(
        type(exception).__name__, limit=_MAX_SYMBOL_BYTES
    )
    if exception_type is None or _IDENTIFIER_RE.fullmatch(exception_type) is None:
        return None
    frames = _traceback_frames(call.excinfo.tb)
    if not frames:
        return None
    terminal = frames[-1]
    terminal_code = terminal.tb_frame.f_code
    terminal_path = _safe_frame_path(terminal_code.co_filename)
    terminal_function = _bounded_text(
        terminal_code.co_name, limit=_MAX_SYMBOL_BYTES
    )
    terminal_line = terminal.tb_lineno
    if (
        terminal_path is None
        or terminal_function is None
        or _CODE_NAME_RE.fullmatch(terminal_function) is None
        or type(terminal_line) is not int
        or not 0 < terminal_line <= _MAX_LINE
    ):
        return None

    exception_record = {"type": exception_type}
    attribute_record = _direct_class_attribute(
        exception,
        frames,
        expected_path=expected_path,
        expected_function=expected_function,
    )
    if attribute_record is not None:
        exception_record.update(attribute_record)
    return {
        "schema": _FAILURE_SCHEMA,
        "phase": phase,
        "test": {
            "path": expected_path,
            "function": expected_function,
        },
        "frame": {
            "path": terminal_path,
            "function": terminal_function,
            "line": terminal_line,
        },
        "exception": exception_record,
    }


def pytest_runtest_makereport(item, call):
    """Capture a bounded exception record before pytest renders a traceback."""

    failure = _failure_metadata(item, call)
    if failure is not None:
        _phase_failures.setdefault(call.when, failure)


def pytest_collectreport(report):
    global _collection_failed, _collection_skipped
    if report.failed:
        _collection_failed = True
    elif report.skipped:
        _collection_skipped = True


def pytest_runtest_logreport(report):
    node_id = str(report.nodeid)
    _observed_nodes.add(node_id)
    if not _same_node(node_id, _expected_node_id):
        _merge("runner_error")
        return

    was_xfail = hasattr(report, "wasxfail")
    if report.when == "setup":
        if report.failed:
            _merge("runner_error")
        elif report.skipped:
            _merge("not_observed")
        return
    if report.when == "call":
        if was_xfail or report.skipped:
            _merge("not_observed")
        elif report.failed:
            _merge("failed")
        elif report.passed:
            _merge("passed")
        else:
            _merge("runner_error")
        return
    if report.when == "teardown":
        if report.failed:
            _merge("runner_error")
        elif report.skipped:
            _merge("not_observed")


def _session_status(exitstatus):
    if _collection_failed or len(_observed_nodes) > 1:
        return "runner_error"
    # ``not_observed`` is reserved for an expected node that pytest actually
    # collected and then skipped/xfail'd.  Exit 4/5 with no runtest report is
    # a missing canonical node (or equivalent collection failure), not an
    # allowed safety skip. Keeping that distinction in the runner prevents
    # callers from treating mere file existence as successful authoring.
    if not _observed_nodes:
        return "runner_error"
    if _status == "passed":
        return "passed" if int(exitstatus) == 0 else "runner_error"
    if _status == "failed":
        return "failed" if int(exitstatus) == 1 else "runner_error"
    if _status == "not_observed" or _collection_skipped:
        return "not_observed"
    if int(exitstatus) in {4, 5}:
        return "not_observed"
    return "runner_error"


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    global _result_fd
    if _result_fd < 0:
        return
    status = _session_status(int(exitstatus))
    failure = None
    if status == "failed":
        failure = _phase_failures.get("call")
    elif status == "runner_error":
        failure = _phase_failures.get("setup") or _phase_failures.get("teardown")
    payload = status.encode("ascii")
    if failure is not None:
        candidate = json.dumps(
            {"failure": failure, "status": status},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        if len(candidate) <= _MAX_RESULT_BYTES:
            payload = candidate
    try:
        os.write(_result_fd, payload)
    finally:
        os.close(_result_fd)
        _result_fd = -1


def worker_exact(result_fd, expected_node_id, pytest_args):
    """Run pytest in the child and emit a bounded hook report."""

    global _expected_node_id, _result_fd
    # ``pass_fds`` necessarily makes this descriptor survive the worker exec.
    # Restore close-on-exec before importing project code so a target-spawned
    # subprocess cannot keep the one-use report channel alive.
    os.set_inheritable(result_fd, False)
    _expected_node_id = expected_node_id
    _result_fd = result_fd
    import pytest

    return int(pytest.main(pytest_args, plugins=[sys.modules[__name__]]))


def supervise_exact(
    encoded_source,
    expected_node_id,
    pytest_args,
):
    """Own the final outcome outside the process that imports project code."""

    read_fd, write_fd = os.pipe()
    worker = (
        "import base64,sys,types;"
        "m=types.ModuleType('_crucible_exact_pytest_worker');"
        "m.__file__='<crucible_exact_pytest_worker>';"
        "sys.modules[m.__name__]=m;"
        f"exec(compile(base64.b64decode('{encoded_source}'),m.__file__,'exec'),m.__dict__);"
        "raise SystemExit(m.worker_exact(int(sys.argv[1]),sys.argv[2],sys.argv[3:]))"
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-I",
                "-c",
                worker,
                str(write_fd),
                expected_node_id,
                *pytest_args,
            ],
            pass_fds=(write_fd,),
            # Target tests can fork or spawn a long-lived child. Do not let
            # that child inherit the supervisor's outer capture pipes and
            # delay the supervisor after the pytest worker has exited.
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except BaseException:
        completed = None
    finally:
        os.close(write_fd)

    # The worker has exited, so its report is already buffered. Do not wait
    # for EOF: project code may have forked a process that inherited the pipe
    # descriptor and keeps it open indefinitely.
    os.set_blocking(read_fd, False)
    chunks = []
    size = 0
    try:
        while size <= _MAX_RESULT_BYTES:
            chunk = os.read(read_fd, min(1024, _MAX_RESULT_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
    except BlockingIOError:
        pass
    os.close(read_fd)
    payload = b"".join(chunks)
    status = payload.decode("ascii", errors="replace")
    failure_output = b""
    if payload.startswith(b"{") and len(payload) <= _MAX_RESULT_BYTES:
        try:
            record = json.loads(payload)
            if (
                type(record) is not dict
                or set(record) != {"failure", "status"}
                or record["status"] not in {
                    "passed", "failed", "not_observed", "runner_error"
                }
                or type(record["failure"]) is not dict
            ):
                raise ValueError("invalid observer record")
            # Re-encode the inner value canonically.  The outer process will
            # apply the independent schema validator before accepting it.
            failure_output = json.dumps(
                record["failure"],
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            if len(failure_output) > _MAX_RESULT_BYTES:
                raise ValueError("observer failure record exceeded cap")
            status = record["status"]
        except (UnicodeError, ValueError, TypeError, json.JSONDecodeError):
            status = "runner_error"
            failure_output = b""
    returncode = completed.returncode if completed is not None else None
    consistent = (
        (status == "passed" and returncode == 0)
        or (status == "failed" and returncode == 1)
        or (
            status == "not_observed"
            and returncode in {0, 1, 4, 5}
        )
        or status == "runner_error"
    )
    if not consistent:
        status = "runner_error"
        failure_output = b""
    code = {
        "passed": PASSED,
        "failed": FAILED,
        "not_observed": NOT_OBSERVED,
        "runner_error": RUNNER_ERROR,
    }.get(status, RUNNER_ERROR)
    if failure_output and status in {"failed", "runner_error"}:
        try:
            os.write(1, failure_output)
        except OSError:
            pass
    os._exit(code)
