# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Driver of the Codex official-SDK kind: the worker route behind its locked worker.

The backend is built exactly as a workflow builds it, with ``resolve_for_role``
on the column's shipped routing file, and nothing is overlaid. Everything from
routing down to the tool runs: the adapter with the inherited controller, its
worker client (the probe of the installed runtime, the process group of a
request, the bounded reads, the validation of the terminal payload), the
locked interpreter of ``backends/codex_worker`` with the real worker module
(its request validation, its configuration overrides, its projection of the
tool items) and the pinned official SDK with its generated types and its
client of the app-server.

What is replaced is the vendor's command-line tool, which the SDK would start
from the runtime its dependency bundles, and the provider behind it. Two
pieces do that:

- the launch interposer (:mod:`..peers.launch`) at the two places where the
  worker client starts the locked interpreter: ``subprocess`` (the probe) and
  ``asyncio`` (the request), both as that module sees them. Every launch of
  the locked interpreter runs under the worker harness, which installs the
  network guard before anything of the worker environment is imported;
- the stage (:mod:`..peers.codex_worker`), which the request launch runs in
  front of the worker module: it puts the scripted peer of the cell
  (:mod:`..peers.codex_app_server`) where the SDK looks for its bundled
  runtime.

The peer is the one the app-server column uses and speaks the same dialect,
so every notification of a cell is validated by the vendor's generated types
inside the worker: one the SDK cannot validate never reaches the worker.

The seam canary of the column fails unless the SDK in the locked worker
started the peer: the peer's parent is a worker that the harness journalled
(the adapter's own interpreter, isolated, without bytecode, under the guard,
on the stage), the stage recorded that it replaced the bundled runtime by
the peer of the cell, the session was opened by the pinned SDK under its own
name and version, and the sign-in file was found as a private copy.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

from crucible.backends import codex_official_worker as worker_client
from crucible.backends.config import load_routing, resolve_for_role

from .. import columns as matrix_columns
from ..columns import CANARY_CREDENTIAL, COLUMN_PRODUCT, Column
from ..peers import codex_worker as stage
from ..peers import launch
from ..provider import LoopbackProvider, SeenRequest, answered_from, refused_from
from ..script import ProviderScript
from .base import (
    GUARD_POLICY,
    Cancellation,
    Lens,
    MatrixSetupError,
    close_when,
    register,
)
from .codex_legacy import (
    LOGIN_WORD,
    CodexAppServerDialect,
    Peer,
    private_home,
    write_peer,
)

_WORKER_PROJECT = launch.REPOSITORY / "backends" / "codex_worker"
# What the worker client starts for a request: ``-m`` this module, no argument.
_WORKER_MODULE = "crucible_codex_worker"
# The two distributions the worker's lock pins together: the SDK and the
# runtime it bundles.
_SDK = "openai-codex"
_RUNTIME = "openai-codex-cli-bin"
# How the pinned SDK names itself in its first request
# (``openai_codex.client.CodexConfig.client_name``); the worker sets no other.
_SDK_CLIENT = "codex_python_sdk"
# The two launch seams of the worker client: module globals it calls through.
(_SUBPROCESS_SEAM,) = (
    name for module, name in launch.SUBPROCESS_SEAMS if module == worker_client.__name__
)
(_ASYNCIO_SEAM,) = (
    name for module, name in launch.ASYNCIO_SEAMS if module == worker_client.__name__
)
_STAGE = Path(os.path.abspath(stage.__file__))
INSTALL = "uv sync --frozen --project backends/codex_worker"


@functools.cache
def pinned_version(distribution: str) -> str:
    """The version of a distribution the worker's lock pins: read, never typed."""

    lock = tomllib.loads((_WORKER_PROJECT / "uv.lock").read_text(encoding="utf-8"))
    versions = [
        str(package.get("version"))
        for package in lock.get("package", ())
        if package.get("name") == distribution
    ]
    if len(versions) != 1:
        raise MatrixSetupError(
            f"the Codex worker lock pins {distribution!r} {len(versions)} times"
        )
    return versions[0]


def staged(journal: launch.Journal) -> list[Mapping[str, Any]]:
    """What the stage recorded in the journal of a cell."""

    path = journal.directory / stage.STAGED
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


class OfficialDialect(CodexAppServerDialect):
    """The app-server as the pinned official SDK starts it inside the locked worker."""

    def __init__(
        self,
        *,
        client: str = "",
        journal: launch.Journal | None = None,
        interpreter: str | None = None,
        launcher: str | None = None,
    ) -> None:
        super().__init__(client=client)
        # The journal of the cell's worker launches, the interpreter the
        # adapter says it starts and the peer of the cell; without them
        # nothing can pass the seam.
        self._journal = journal
        self._interpreter = interpreter
        self._launcher = launcher

    def launch_problems(
        self, session: Mapping[str, Any], before: Sequence[str], seen: SeenRequest
    ) -> list[str]:
        launches = [] if self._journal is None else self._journal.launches()
        worker = next(
            (item for item in launches if item.get("pid") == session.get("ppid")), None
        )
        if worker is None:
            return [
                f"the peer ran with parent {session.get('ppid')}, which is no worker "
                "launch the harness journalled"
            ]
        problems: list[str] = []
        if worker.get("pid") == os.getpid() or worker.get("executable") != self._interpreter:
            problems.append(
                f"the worker that started the peer ran {worker.get('executable')!r} as "
                f"process {worker.get('pid')}: not the adapter's locked interpreter "
                f"{self._interpreter!r} in a process of its own"
            )
        if (worker.get("form"), worker.get("target")) != ("script", str(_STAGE)):
            problems.append("the worker that started the peer was not launched on the stage")
        if not (worker.get("isolated") and worker.get("no_bytecode") and worker.get("guard")):
            problems.append(
                "the worker that started the peer did not run isolated, without "
                "bytecode and under the network guard"
            )
        records = [] if self._journal is None else staged(self._journal)
        record = next((item for item in records if item.get("pid") == worker.get("pid")), None)
        if record is None or record.get("executable") != self._launcher:
            problems.append(
                "the stage did not record that it put the peer of the cell where the "
                "SDK looks for its bundled runtime"
            )
        elif record.get("module") != _WORKER_MODULE or not str(record.get("bundled")).startswith(
            str(Path(str(self._interpreter)).parents[1])
        ):
            problems.append(
                "the stage did not run the worker module of the locked environment in "
                "the place of its bundled runtime"
            )
        methods = [str(item.get("method")) for item in seen.evidence.get("requests", ())]
        if "initialized" not in methods:
            problems.append("the client did not confirm the handshake as the SDK does")
        return problems


@dataclass
class _Bound:
    backend: Any
    provider: LoopbackProvider
    log_dirs: tuple[Path, ...]
    peer: Peer
    journal: launch.Journal
    secrets: tuple[str, ...] = (LOGIN_WORD,)
    tool_path: Literal["function_call", "mcp_bridge", "control_channel"] = "mcp_bridge"

    def cancel_when(self, marker: str) -> Cancellation:
        return close_when(self.provider, self.backend, marker)

    def children(self) -> tuple[int, ...]:
        # Every worker of the cell and every process a worker's SDK started.
        return tuple(sorted({*self.journal.pids(), *self.peer.pids()}))


class CodexOfficialDriver:
    """The shipped official-SDK route, its worker's tool answered by the scripted peer."""

    kind = matrix_columns.CODEX_OFFICIAL_KIND
    dialect: str = CodexAppServerDialect.name
    fidelity: Literal["wire", "worker_protocol"] = "wire"
    bypassed_layers: tuple[str, ...] = (
        "the -m entry of the worker (the worker harness and the stage run the same "
        "module as __main__, in the locked interpreter with the launch's own options)",
        "the runtime the official SDK bundles (the stage puts a scripted peer where "
        "the SDK looks for it; its tool loop and the usage counters of its thread "
        "are emulated)",
        "the provider (the peer takes each model response from scripted envelopes)",
    )
    # Module globals of the worker client, replaced through the launch
    # interposer for as long as a backend is open.
    private_seams: tuple[str, ...] = (
        f"{worker_client.__name__}.{_SUBPROCESS_SEAM}",
        f"{worker_client.__name__}.{_ASYNCIO_SEAM}",
    )
    # One locked worker is started for the call, and it starts one scripted
    # tool, which opens its session before the request of the turn: about a
    # second on an idle machine, between five and six where the test shares
    # one hardware thread with busy loops.
    startup_seconds: float = 10.0
    faults: tuple[str, ...] = ()

    @property
    def inexpressible(self) -> Mapping[str, str]:
        """Script steps this kind's wire cannot carry: the inventory's rows only."""

        from .. import cells  # noqa: PLC0415 - cells imports the drivers

        return {
            f"{row['scenario']}/{row['step']}": row["reason"]
            for row in cells.load_inventory().not_expressible
            if row["kind"] == self.kind
        }

    @property
    def envelopes(self) -> tuple[str, ...]:
        return CodexAppServerDialect().envelope_names()

    def answers(self, script: ProviderScript) -> tuple[str, ...]:
        return answered_from(CodexAppServerDialect(), script)

    def refuses(self) -> tuple[str, ...]:
        return refused_from(CodexAppServerDialect())

    @property
    def interpreter(self) -> Path:
        """Where the worker environment puts its interpreter."""

        return _WORKER_PROJECT / ".venv" / "bin" / "python"

    def _stage(self, peer: Path, journal: launch.Journal) -> Any:
        """The request launch of the worker module runs on the stage; all else is guarded.

        The decision is the interposer's own; a request launch is then started
        on the stage in the place of the worker module, which the stage runs.
        """

        interposer = launch.Interposer(lambda: journal)

        def wrap(
            argv: Sequence[str], env: Mapping[str, str] | None
        ) -> tuple[tuple[str, ...], dict[str, str]] | None:
            if not interposer.decides(argv):
                return None
            options, target = launch.split(argv)
            if target != ("-m", _WORKER_MODULE):
                return interposer.command(argv, env)
            environment = dict(os.environ if env is None else env)
            environment[stage.SETTINGS_ENV] = json.dumps(
                {
                    "module": _WORKER_MODULE,
                    "executable": str(peer),
                    "journal": str(journal.directory),
                },
                sort_keys=True,
            )
            return interposer.command((argv[0], *options, str(_STAGE)), environment)

        return interposer, wrap

    @contextlib.contextmanager
    def _interposed(self, peer: Path, journal: launch.Journal) -> Iterator[None]:
        interposer, wrap = self._stage(peer, journal)
        with contextlib.ExitStack() as seams:
            seams.enter_context(launch.at_subprocess(worker_client, interposer, _SUBPROCESS_SEAM))
            seams.enter_context(launch.at_asyncio(worker_client, wrap, _ASYNCIO_SEAM))
            yield

    def _probe(self, journal: launch.Journal) -> None:
        """The adapter's own probe of the installed worker, under the guard."""

        # The probe is cached per interpreter; it must run for this journal.
        worker_client.qualify_worker_runtime.cache_clear()
        try:
            with self._interposed(self.interpreter, journal):
                worker_client.qualify_worker_runtime(str(self.interpreter))
        except (
            worker_client.CodexOfficialSDKUnavailable,
            worker_client.CodexOfficialSDKQualificationError,
        ) as exc:
            raise MatrixSetupError(
                f"kind {self.kind}: the locked worker environment is not the qualified "
                f"one ({exc}); install it with `{INSTALL}`"
            ) from exc
        finally:
            worker_client.qualify_worker_runtime.cache_clear()
        violations = journal.violations()
        if violations or not journal.launches():
            raise MatrixSetupError(
                f"kind {self.kind}: the probe of the worker did not run cleanly under "
                f"the harness: {'; '.join(violations) or 'no launch was journalled'}"
            )

    def preflight(self) -> None:
        if sys.platform != "linux":
            raise MatrixSetupError(f"kind {self.kind} runs on Linux only")
        if not self.interpreter.is_file():
            raise MatrixSetupError(
                f"kind {self.kind} needs the locked worker environment, and its "
                f"interpreter ({'/'.join(self.interpreter.parts[-5:])}) does not exist; "
                f"install it with `{INSTALL}`"
            )
        if not launch.is_locked_interpreter(self.interpreter):
            raise MatrixSetupError(
                f"kind {self.kind}: the worker interpreter lies outside every worker "
                "environment: the launch interposer would not guard it"
            )
        with tempfile.TemporaryDirectory(prefix="matrix-preflight-") as directory:
            self._probe(launch.Journal(Path(directory)))

    def overlay(self, column: Column, env: Mapping[str, str]) -> Mapping[str, str]:
        return {}

    def _dialect(self, journal: launch.Journal, launcher: Path) -> CodexAppServerDialect:
        return OfficialDialect(
            client=f"{_SDK_CLIENT}/{pinned_version(_SDK)}",
            journal=journal,
            interpreter=str(self.interpreter),
            launcher=str(launcher),
        )

    @contextlib.contextmanager
    def open(
        self,
        column: Column,
        *,
        capability: str,
        lens: Lens,
        script: ProviderScript,
        tmp_path: Path,
        without_credential: bool = False,
    ) -> Iterator[_Bound]:
        if column.kind != self.kind:
            raise MatrixSetupError(f"column {column.id} is not of kind {self.kind}")
        log_dir, cwd = tmp_path / "backend-logs", tmp_path / "backend-cwd"
        cwd.mkdir(parents=True, exist_ok=True)
        journal = launch.Journal(tmp_path / "worker-harness")
        journal.directory.mkdir()
        launcher = tmp_path / "codex-peer" / "codex"
        with contextlib.ExitStack() as stack:
            # The worker of the route needs the sign-in file of the vendor
            # tool, as the head of its routing file says, and the adapter
            # looks for that file at the head of each attempt, whatever
            # source of the credential the routes of the file name. Without
            # its credential the home of the cell has the directory of that
            # file and no such file.
            stack.enter_context(private_home(tmp_path, signed_in=not without_credential))
            dialect = self._dialect(journal, launcher)
            provider = stack.enter_context(
                LoopbackProvider(script, credential=CANARY_CREDENTIAL, dialect=dialect)
            )
            peer = write_peer(
                launcher.parent,
                provider=provider.url + dialect.path,
                cli_version=pinned_version(_RUNTIME),
                faults=self.faults,
            )
            # The adapter probes the worker once per process and interpreter.
            # A cell shows its own probe: under the harness, in its journal.
            worker_client.qualify_worker_runtime.cache_clear()
            stack.callback(worker_client.qualify_worker_runtime.cache_clear)
            stack.enter_context(self._interposed(peer.launcher, journal))
            # The boundary of the matrix: exactly what a workflow calls.
            backend = resolve_for_role(
                matrix_columns.ROLE_BY_CAPABILITY[capability],
                routing=load_routing(config_path=column.routing_path(COLUMN_PRODUCT)),
                capability=capability,
                native_shell=False,
                allow_trusted_project_instructions=False,
                cwd=str(cwd),
                log_dir=log_dir,
                limit_policy=GUARD_POLICY if lens == "guarded" else None,
            )
            try:
                yield _Bound(
                    backend,
                    provider,
                    # The journal holds the command line of every worker launch:
                    # clause C9 reads it with the adapter's own records.
                    log_dirs=(log_dir, journal.directory),
                    peer=peer,
                    journal=journal,
                )
            finally:
                with contextlib.suppress(Exception):
                    backend.close()  # the runner has observed close() already
                violations = journal.violations()
                failures = peer.failures()
                if violations:
                    raise MatrixSetupError(
                        f"column {column.id}: a worker of the cell reached beyond the "
                        f"scripted provider: {'; '.join(violations)}"
                    )
                if failures:
                    raise MatrixSetupError(
                        f"column {column.id}: the scripted peer did not play the "
                        f"session: {'; '.join(failures)}"
                    )


register(CodexOfficialDriver())
