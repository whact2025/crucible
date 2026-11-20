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

A scripted reply is an envelope under `backend_matrix/envelopes/<dialect>/`
with only its slots filled, and each envelope states where its shape comes
from: `recorded` from the provider, `documented` by the vendor or a live
observation, or `constructed` to satisfy the SDK's types. The weakest envelope
a cell used is its grade. A difference that no feature field expresses is a
declared difference of the ledger, with a product citation and the alternative
contract that is asserted instead; it replaces no contract clause.

```bash
uv run python -m pytest -q tests/backend_matrix
```

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

It fails when an expected cell is missing or ran twice, when an item of the
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
