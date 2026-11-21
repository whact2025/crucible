<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Crucible test guide

The ordinary suite verifies the reusable solution-building workflow and the
SWE-bench/DeepSWE adapters without paid model calls or a Docker daemon.

```bash
uv run python -m pytest -q
uv run ruff check .
git diff --check
```

The configured marker filter excludes `llm_live` and `live_docker`. Run a
focused set first, then the complete non-live suite for cross-cutting changes.
Linux-only process containment, filesystem, and settlement tests require their
actual host facilities; report platform skips explicitly.

## Hosted CI

CI assigns every discovered test module to one of eight groups and runs each
file intact. The groups share one inventory in `scripts/test_shard.py`; new
test files are included automatically. The final quality gate requires every
group to succeed and supply its coverage, test report, and inventory. It
combines coverage before enforcing the 55% floor.

The ephemeral Ubuntu test runners enable unprivileged user namespaces and
verify root mapping before the quota helper's real descriptor-execution
tests. A missing runner prerequisite fails setup instead of bypassing those
tests.

## Contract checks

Each regression preserves an observable behavior or failure boundary. Keep
scalar fields, wire types, authority identities, and rollback or publication
phases independently covered. A shared successful cascade can own multiple
assertions. Gate doubles settle their exact writer epoch so a dispatch
regression fails its behavior assertion with cleanup still intact.

| Area | Representative tests |
|---|---|
| Current product and docs | `test_axioms.py`, `test_documentation.py`, `test_workflows_dispatcher.py` |
| Plans and accepted authority | `test_workflows_plan_validation.py`, `test_plan_provenance.py`, `test_plan_source_integrity.py` |
| Build and adversarial gates | `test_workflows_build_verify.py`, `test_workflows_breach.py`, `test_workflows_tier_breach_policy.py` |
| Suite behavior | `test_workflows_swebench.py`, `test_workflows_deepswe.py`, suite dataset/container/verifier tests |
| Parallel work | `test_workflows_readonly_scheduler.py`, `test_workflows_writer_scheduler.py`, `test_workflow_outcomes.py` |
| Mutation and recovery | Mutation-journal, writer-epoch, settlement, rollback, reconcile, and journal-recovery tests |
| Routes and reviews | Backend conformance, config/features/limits, pricing, rubric, and calibration tests |
| Distribution | `test_root_wheel.py`, `test_editable_install_identity.py` |

The root pytest bootstrap verifies that `import crucible` resolves to the
checkout being tested. Use worktrees shaped `<unique-parent>/crucible` with
Assay as its sibling, and install that checkout's development environment.
Testing another editable checkout is invalid evidence.

## Backend conformance matrix

`tests/backend_matrix/` plays one catalogue of scenarios against every backend
column and judges each on what a workflow can see: the result or typed error,
usage and completeness flags, the trace, tool dispatch and what is still
running afterwards. It is the acceptance test for backend gap fixes: a
code-fixable gap is a failing cell before its fix and a passing cell after.

A cell is one scenario on one column in one lens. Its id reads
`<kind>/<scenario>/<role>/<lens>`: the driver kind of the column (a catalog
backend and transport, or `toy`), the scenario (`S01`) or static check (`H02`),
the column's role within its kind (`primary`, `secondary`, `blocked`) and the
lens (`raw`, `guarded` with the diagnostic limit guard bound, or `static`).
Columns come from the shipped profile catalogs; profile names and model ids
live in `backend_matrix/columns.py` only.

A cell computes deviation codes from a closed vocabulary
(`backend_matrix/contract.py`), and
[the ledger](backend_matrix/ledger.yaml) says which are known gaps:

- a cell with no entry passes only with no deviation;
- a cell with an entry is an expected failure only while it fails with exactly
  the recorded codes, under the declared feature values recorded with it;
- it fails with "gap fixed, delete the entry" when it stops deviating, with
  "different deviation" for any other set, and with `decl.changed` when the
  route's declaration moved.

A fix deletes its entry in the same change. Only a behaviour cell of a real
backend and the static cells `H01` and `H02` can have an entry: the toy cells
and the other static cells check the matrix itself and are never expected
failures. The gap registry in the same file is permanent, and
[the declaration snapshot](backend_matrix/declared_profiles.yaml) pins every
column's declared profile, so a declaration cannot be weakened to turn a cell
green. Nothing in the package skips: skip and xfail markers are refused at
collection, a column that cannot run is an error, and
[the inventory](backend_matrix/inventory.yaml) pins the cell count of each
kind.

A scenario lands with its lying counterpart. To add one, define it in
`backend_matrix/scenarios.py` (policy, provider script, ToolSet, lenses,
expected result and a difference table that is total over its governing
feature field), teach `HonestBackend` in `backend_matrix/drivers/toy.py` to
play it, and add at least one liar there with the exact codes its cell must
yield. Static cell H03 rejects a scenario without a liar.

A real column is driven by replacing the network, never the adapter: the
backend is built with `resolve_for_role` on its shipped routing file and
answered by a scripted provider on loopback. The hosted gateway route keeps
its production URL and [an origin redirect](backend_matrix/tap.py) answers
that origin on loopback; the local route points at the scripted provider.
The seam canary `S00` of each column fails unless the request is the SDK's own
(path, user agent, credential, the host the client addressed), and every
matrix module ends with a check that the canaries of its columns passed.

A reference column is driven by replacing the vendor executable instead. The
Claude session route keeps its shipped routing file and gets one overlay,
`extra.cli_path`, which points at
[a scripted peer](backend_matrix/peers/claude_cli.py). The adapter's Agent SDK
starts the peer as it starts the vendor tool. The peer takes each model turn
from the scripted provider and sends each tool call through the SDK's control
channel to the real dispatch bridge. Its canary fails unless the SDK started
it: the two probes of the session route ran first, the environment was
sanitised, and the control handshake preceded the prompt. The peer's journal
names the variables of its environment and keeps none of their values beyond
the three the canary reads. Nothing was observed of the real tool, so every
frame of the peer is `constructed` and its session accounting is an emulation.
What the peer does not emulate ends the cell as a setup error, never as an
observation. A known gap on a reference column needs the owner's ruling in its
ledger entry.

The Codex app-server route is driven the same way: `extra.cli_path` points at
[a scripted peer](backend_matrix/peers/codex_app_server.py), which the pinned
`codex-app-server-sdk` starts where the route looks for the vendor tool. That
route accepts only a native executable, so for as long as a cell holds a
backend the driver replaces the adapter's native-binary predicate by one that
also accepts the peer's launcher. It is the one product name the driver
patches, it is listed in the driver's `private_seams`, and the predicate's own
tests stay. The version probe is answered with a version the adapter has
reviewed, and the sign-in file is a fake one in a home directory of the cell,
which the driver makes `HOME` while the backend is open. The peer speaks the
app-server's JSON-RPC, takes each model response from the scripted provider
and reaches the ToolSet through the loopback tool server its command line
names. Its notifications are built to the generated types of the vendor's
official SDK, with one exception that says so: the exact usage of a response,
which no installed vendor type states and on which the complete accounting of
the column rests. The usage counters of the thread are emulated, and the type
of their notification names two sets of them and the meaning of neither: the
envelope fills `last` with one model response, a choice no source states. The
green of this column does not rest on it (its adapter reads the thread's
total there); what the official-SDK column reports for a turn of several
responses does, and the test modules of both columns play the other reading.
The wire of both Codex kinds names no served model; declared difference
`D-CDX-SERVED` asserts on every result cell of those kinds that the trace
records none, and static cell `H03` fails when the ledger's list leaves one
out.

A column whose adapter runs a locked worker environment is driven inside
that environment. The DeepSeek route keeps its shipped routing file and gets
one overlay, `extra.expected_worker_runtime_fingerprint`, captured in the cell
from the installed environment. The adapter's own supervisor starts the
worker, under [the worker harness](backend_matrix/peers/worker_harness.py):
the harness installs the network guard before anything of the environment is
imported and, for the request launch, the origin redirect, so that the pinned
SDK's request for the production origin is answered by the scripted provider.
The canary fails unless the request came from that worker: the harness
journalled a redirected launch of the locked interpreter before the worker
ran, the request carries the pinned SDK's user agent, the credential and the
production host, and the route was built with the fingerprint captured in the
cell.

The Codex official-SDK route runs its locked worker too, and it has no
provider origin to redirect: the SDK in the worker starts the vendor's
command-line runtime, which its environment bundles. The driver starts the
request launch of the worker on [a stage](backend_matrix/peers/codex_worker.py)
under the harness. The stage puts the scripted peer of the app-server column
where the SDK looks for that runtime and then runs the real worker module, so
the worker, the pinned SDK and its generated types run for real and validate
every notification of a cell. The canary fails unless the peer's parent is a
journalled worker of the locked interpreter on the stage and the stage
recorded what it replaced. The harness also refuses to start any program that
lies inside the worker's own environment, because such a program opens its
sockets where the network guard does not see them: without the stage the
bundled runtime never starts, and the attempt is a violation.

The scripted provider picks a reply by position and tool-call ids, never by
what a request contains. Where a wire keeps no conversation, the dialect of
the column states what each request must replay of the earlier rounds, and
every scenario with a ToolSet reports a request that does not as
`request.content:replay`. For the DeepSeek route that is the reasoning item,
the function call and its output, in order. For the gateway routes it is,
after the opening messages, the assistant turn that asked for a tool directly
followed by the tool message that answers it, round by round. A rule holds
the shape of the conversation (kinds, order, call ids), not its text. The
Claude session keeps its conversation itself, so its dialect states no rule.

Every interpreter of a worker environment (`backends/<name>_worker/.venv`)
that any test starts runs under the same harness. A suite-wide fixture in
`tests/conftest.py` interposes where the worker adapters start one (the
DeepSeek supervisor, `subprocess` as the Kimi adapter sees it, `subprocess`
and `asyncio` as the worker client of the official Codex route sees them),
decides by the executable and passes every other launch through untouched. A
worker that reaches beyond loopback fails the test that started it, whatever
the worker did with the exception. A test that starts such an interpreter
itself uses the fixture's `run` or `popen` in place of `subprocess`. Test
modules named after a worker adapter (`kimi`, `deepseek`, `codex`) also run
with the network guard in the test process.

A scripted reply is an envelope under `backend_matrix/envelopes/<dialect>/`
with only its slots filled, and each envelope states where its shape comes
from: `recorded` from the provider, `documented` by the vendor or a live
observation, or `constructed` to satisfy the SDK's types. The weakest envelope
a cell used is its grade. A difference that no feature field expresses is a
declared difference of the ledger, with a product citation and the alternative
contract that is asserted instead; it replaces no contract clause.

```bash
uv sync --frozen --reinstall --link-mode copy --project backends/deepseek_worker
uv sync --frozen --project backends/codex_worker
uv run python -m pytest -q tests/backend_matrix
```

The first two commands install the locked DeepSeek and Codex worker
environments, which the DeepSeek column, the official-SDK Codex column and
the tests marked `matrix_worker` need. The Codex environment holds the
vendor's command-line runtime, which the official SDK depends on: most of its
size (about 340 MB installed), and never started by a test. A missing or
drifted environment is an error with its command in the message, never a
skip. Whether a column can be driven is asked once per run, under the same
cleared environment as its cells: a credential, a proxy or a
`CRUCIBLE_*_WORKER_PYTHON` override of the machine decides nothing. Clone
inside the Linux filesystem: the DeepSeek pin hashes its lock byte
for byte and every installed file with its mode, so a checkout on a Windows
filesystem reached through WSL can fail it. The pin also covers the worker's
source directory with any bytecode in it, and it is checked again before
every launch: a test that loads worker source in its own process loads it
without writing bytecode (as `tests/test_backends_deepseek.py` does), or it
would change the pin under the tests that run beside it in the same checkout.
CI installs each of them in every test shard. The Kimi worker environment
(`uv sync --frozen --project backends/kimi_worker`) is not installed in CI yet:
the tests that need it still skip without it, which is the known gap of cell
`kimi/H01/-/static`.

On a machine without the worker environment, leave these tests out explicitly
and keep the configured expression:

```bash
uv run python -m pytest -q -m "not matrix_worker and not live_docker and not llm_live"
```

A `-m` on the command line replaces the expression configured in
`pyproject.toml`, so a shorter `-m "not matrix_worker"` would select the live
tests, which call providers on a machine that holds credentials. pytest prints
how many tests were deselected, and the verifier below fails a CI run in which
a cell did not run.

The run proves the oracle on the toy driver (honest cells pass, every liar
yields exactly its codes), runs the driven columns and the static cells. It
needs Linux, no network and no credentials; a network guard fails any test of
the package that reaches beyond loopback. The scripted peer of a reference
column runs in processes of its own, and its launcher installs the same guard
there: a peer that reaches beyond loopback crashes, which ends its cell as a
setup error.

After the CI shards, the coverage job runs the cell-inventory verifier on
their JUnit files:

```bash
uv run python -m tests.backend_matrix.verify
```

It fails when an expected cell is missing or ran twice, when a proof that a
module must hold by name (`REQUIRED_PROOFS` in `backend_matrix/cells.py`:
product behaviour no clause states, so no cell shows when it stops) or the
closing check of a module did not run or failed, when an item of the
package was skipped, when an expected failure does not name exactly the gaps
the ledger attributes to its cell, when a ledgered cell passed, and when the
cell count of a kind differs from the pinned one. A cell is ledgered by an
entry of its own or by a column-wide entry of its kind. A column-wide gap is
owed on every behaviour cell of the kind whose definition has the clause of
its code, except the seam canary, a cell whose own entry shows that the check
did not run (a raise, a guard rewrite, the cost flagged incomplete) and a
cell the inventory pins under `column_wide_not_owed` with its reason. A run
that writes a JUnit file also writes
`matrix-report-<n>.json` beside it: the codes, gaps, declared values, clause
outcomes and grade of every cell the process judged.

## Live tests

Docker probes and paid model calibration are separate, explicitly selected
lanes. Run only the authorized tests and exact route. Containers and model
sessions can consume substantial resources; their results need complete
cleanup and identity evidence. Hidden benchmark evaluation is a product run,
not an incidental unit-test action.

Report tests executed, failures, skipped platform checks, and any unavailable
live gates. Never claim that offline tests establish solved benchmark tasks or
promote a diagnostic backend route.
