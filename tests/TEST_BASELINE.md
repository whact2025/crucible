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

A shared guard fails any test that ends with a workspace lock chain or a
registered workspace authority that appeared during it, and names what the
test left. Settle the lock owner a test caught with `retry_cleanup`, exit
every lock it entered, and join every thread that takes a lock before the
test returns.

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
Columns come from the shipped profile catalogs; profile names live in
`backend_matrix/columns.py` only, and model ids are read from the catalogs at
run time and typed nowhere in the package.

The scenarios come in three sets (`backend_matrix/scenarios.py`): the starter
set (`S00`, `S01`, `S03`, `S04`), pack A (limits on turns, tool dispatch, cost
and tool-result bytes, refusals, a write as progress) and pack B, which is
every scenario in neither and is still landing. The toy columns and every
real kind run all three (`ENABLEMENT` in `backend_matrix/cells.py`); the
static cell `H02` of a kind that did not run pack A would report it. On a
column pack A is ten cells: the turn limit (`S10a`) in both lenses and, in
the raw lens, the suite's default turn count (`S11`), the dispatch-budget
denial (`S12`), the USD overrun of one response (`S13`), the tool-result
byte budget (`S18`), three forms of refusal (`S23a` a content-filter stop,
`S23b` a refusal part of the message, `S23c` an HTTP policy rejection) and a
write as progress (`S40`). The primary and the secondary column of a kind
run the starter set and pack A. Pack B runs on the primary column of a kind
only, each scenario from the change that defines it. A blocked column runs
its three scenarios. A form of refusal that the wire of a kind cannot carry
is pinned not expressible in [the inventory](backend_matrix/inventory.yaml)
and has no cell: the refusal part on the Claude columns, the content-filter
stop and the refusal part on both Codex kinds. No such pin exists on an
open-weight kind.

The two turn cells hold a count. The turn limit asks every turn of the count
of the provider, so a limit that is typed after fewer tool-bearing requests
leaves the script unconsumed. The suite's turn count holds a turn control on
the wire to the count the call asked for. On a route that declares an
enforced turn limit (`hard` or `provider_stop`) it also holds every request
to carry the count, where the wire has a field for one; a dialect says
whether its wire does. Of the real kinds only the Claude wire has one: on
every other real column that cell cannot tell an adapter that takes the
count from one that cuts it to a ceiling of its own, and how such a ceiling
is reported is stated by no clause. A registry record whose closing state no
clause states yet names the question that is owed a ruling
(`awaits_ruling`), and static cell `H03` fails when a cell of such a record
carries no code of the gap, whatever the cell shows: the change that enters
the ruling makes the scenario assert it and takes the field off the record.
The record of the DeepSeek turn ceiling waits for one.

Two static cells run on the toy columns only: `P01` (the protocol surface)
and `P02` (the profile binding). So on a real column no cell asserts that the
backend instance declares the profile routing computes for its route, or that
its limits resolve as that profile implies. The variant of a scenario is
selected from the routed profile, never from the instance, so an instance
cannot choose its expectation; and clause C7 compares the profile digest of
the trace with the routed profile on every result cell, which stands in for
the first half. The DeepSeek column has no such stand-in: its trace binds no
profile digest, which is a column-wide known gap, so nothing there ties the
instance to its route.

A cell computes deviation codes from a closed vocabulary
(`backend_matrix/contract.py`), and
[the ledger](backend_matrix/ledger.yaml) says which are known gaps:

- a cell with no entry passes only with no deviation;
- a cell with an entry is an expected failure only while it fails with exactly
  the recorded codes, under the declared feature values recorded with it;
- it fails with "gap fixed, delete the entry" when it stops deviating, with
  "different deviation" for any other set, and with `decl.changed` when the
  route's declaration moved.

A fix deletes its entry in the same change. The ledger counts its entries
under two ceilings, one for the starter set with pack A and one for pack B,
and states a ceiling frozen (`ceilings.frozen`) once every scenario it
counts is defined and every cell it counts passes or has its entry. That of
pack A is frozen; that of pack B rises while the scenarios of the pack
land. A frozen ceiling is the count of its entries, and static cell `H03`
fails on any other number: the change that deletes an entry lowers it by
one, so no later entry fits into room that a fix left. Only a behaviour
cell of a real backend and the static cells `H01` and `H02` can have an
entry: the toy cells and the other static cells check the matrix itself and
are never expected failures. The gap registry in the same file is
permanent, and
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
The peer emulates one terminal of the tool: a session that has used up its
turn allowance ends, after the tool call of its last allowed turn, with the
error result the pinned SDK names for it (`error_max_turns`) and a status
other than zero, and the model is not asked again. A refusal reaches the
adapter in the two forms the frames of the tool can carry: a model turn whose
stop reason is the refusal, and the error terminal of a request the provider
rejected with an HTTP status, with the accounting the tests of the product
state for that terminal. Every cell that rests on one of these frames
rests on an emulation of the reference. What the peer does not emulate (the
budget terminal between two turns) ends the cell as a setup error, never as an
observation. A known gap on a reference column needs the owner's ruling in its
ledger entry. A ruling that keeps a gap is always written in the entries of
the gap: on a registry record a ruling is the licence to withdraw a cell of
that record, and static cell `H03` refuses one that withdraws none. A ruling
may be a decision panel's under the owner's standing instruction; its text
says so.

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
The notifications carry one form of refusal: a turn that failed with an error
whose kind names a policy of the provider. They have no finish reason of a
model response and no refusal part of a message, so the two other forms are
pinned not expressible on both Codex kinds. The route declares no turn
control, so its turn-limit cell asserts that none is sent and that a finite
script finishes cleanly.
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

The Kimi routes are driven the same way, and nothing is overlaid on their
routing files: the endpoint variable expands to the production endpoint, so
routing admission, the writer gate and the route's feature profile apply. The
adapter starts its worker through `subprocess`, where the same harness is put
in front of the worker: the vendor's client inside the locked interpreter
asks the production origin and is answered by the scripted provider, and the
tool bridge of the call stays on loopback. The canary fails unless the
harness journalled a redirected launch of that interpreter and the request
names the pinned runtime and client, the credential and the production host.
The two diagnostic routes opt into a cost upper bound in their routing files.
Such a route returns a result only for an invocation with a budget, and its
limit guard binds only with the cost policy the file names. Their cells are
called as the route asks: with the matrix's route budget, which no scripted
exchange reaches, unless the scenario sets a budget of its own, and with
that policy in the guarded lens. A result of such a route is a cost interval
whose accounting flags are cleared whenever the provider's reply carries no
count of cache reads, so the flags alone would leave its numbers unasserted.
Where a scenario owes complete accounting, clause C6 therefore holds the
interval itself: it is there, it covers the reported usage, its lower bound
is the reported cost, it has one segment for every response the provider
billed, and the reported tokens are the billed ones. It also holds the two
bounds of every segment to what the price row the product ships for the
route's model gives for the usage the segment records
(`usage.cost_bounds:amounts`): that row is the rate table of the route's
cost policy. The bounds follow from the cache counts a segment records, so
C6 holds those as well, to what the reply of the segment's response carried
(`usage.cost_bounds:counts`): the scripted provider reads the counts back
from the bytes of every billed reply, and a count a reply did not carry is
unknown, never zero. The reported cost is held to the matrix rate card, as
on every column. In the guarded lens clause C5 reads from the limit guard's
terminal whether a result the guard let stand is rankable, and reports
`guard:not_rankable` where it is not. Under a cost policy the guard never
calls a result rankable, so on a route whose routing file opts into the cost
upper bound C5 asks something else, by declared difference
`D-COST-BOUND-UNRANKED`: the marks of a diagnostic result (the terminal of
the cost policy the file names, one physical attempt, the cost authority
kept, not rankable), each with a code of its own when it is missing. The
ledger lists the cells of such routes and `H03` holds the list to exactly
them. The accounting flags stay owed there, so the Kimi result cells are
expected failures for as long as the provider's reply carries no count of
cache reads.

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

Four known gaps of the reference columns lean on a proof that the verifier
requires by name, because the codes of their cells do not say what the gap
is, or what its record says of the result. On a write the trace of the
Claude adapter binds a tool-authority digest
that is not the shared one: the proof plays the write of both Claude columns
and shows that the value is the shared digest of the same tools with no path
scope stated, and that the cell shows no code once the adapter's hash is
bound to the ToolSet that was attached. The result of a dispatch-budget
denial on the official Codex route flags its accounting incomplete, which
switches the comparison of its usage off: the proof shows that it reports no
usage for responses the provider billed. On the app-server route the result
of the same denial flags its accounting incomplete too, and reports every
billed response; a result that had lost its usage would show the same two
codes, so the proof judges the result as one that claims its accounting
complete and finds no code about the size of its usage. Where the limit
guard replaces a result of the official route, its two codes do not say what
the guard missed: a twin of the test that plays both readings of `last`
shows that the guard misses the token flag alone and that nothing but the
usage code moves with the reading. Each proof has lying counterparts beside
it that show it failing on what it is there to exclude. An entry must not
outlive a proof it leans on: `RECORD_PROOFS` in `backend_matrix/cells.py`
names, by registry record, the proofs its entries lean on, and static cell
`H03` fails when an entry names such a record while the verifier does not
require the proof that plays the entry's cell.

A column whose profile the catalogs call blocked runs three scenarios as soon
as its kind is driven: the canary, the plain call and the admission scenario
`S51`. The primary column of every real kind runs that scenario too: it is
a scenario of pack B. An admission cell builds a backend for every
capability a workflow can ask routing for, on the routing files of both
products, and calls none. What routing does with each capability today, and
the word the catalogs have for the column, are pinned for the cell in
[the inventory](backend_matrix/inventory.yaml): the cell passes while routing
does exactly that and no provider request is made, so a gate that opens or
closes changes the pin in the same change. A capability is pinned with one
of three outcomes. `admitted`: routing built the adapter and logged no
requirement of the role that the profile of the route lacks. `retained`:
routing built the adapter and logged such a requirement, which it does on a
route that asks for diagnostic admission; the cell reads the warning of the
routing module for it. Or the type of the exception routing refused with.
Today both gateway kinds are retained for the text and the read-only
capability, whose roles ask for structured output, and the writer of the
blocked Kimi column is refused. The DeepSeek driver binds the fingerprint of
the installed worker to the shipped routes, as a run does. There the cell
asks for every capability a second time on the file as shipped, and the pin
states that routing refuses all three: no route of that kind is built
before a run has bound the worker. An admission cell asks as the driver of
its column asks for every cell (no native shell, no trusted project
instructions, an explicit working directory), on the shipped routes. What
routing refuses to a caller that asks otherwise, or to a writer route
without its opt-in where the shipped route carries it, no cell plays:
`tests/test_backends_config.py` holds those refusals for the Kimi routes
and `tests/test_backends_deepseek.py` for the DeepSeek routes.

The scripted provider picks a reply by position and tool-call ids, never by
what a request contains. Where a wire keeps no conversation, the dialect of
the column states what each request must replay of the earlier rounds, and
every scenario with a ToolSet reports a request that does not as
`request.content:replay`. For the DeepSeek route that is the reasoning item,
the function call and its output, in order. For the gateway routes it is,
after the opening messages, the assistant turn that asked for a tool directly
followed by the tool message that answers it, round by round. For the Kimi
routes it is the same, and an assistant turn that asked for a tool comes back
with the reasoning content the provider sent with it. A rule holds the shape
of the conversation (kinds, order, call ids, whether reasoning is there), not
its text. The Claude session keeps its conversation itself, so its dialect
states no rule.

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
a cell used is its grade. The envelopes of a refusal are `constructed` on
every dialect: no refusal of a provider was observed or is documented in
reach, so a gap that a refusal cell shows cannot be closed on them. The
registry record of a gap that is about what a provider sends names the grade
its cells close on (`closes_on_grade`), and so does, on a reference column,
the record of a gap that is about what the vendor's tool relays of such a
reply or sends when a session is stopped. Static cell `H03` fails when a
cell of such a record carries no code of the gap while the dialect of its
kind still answers it from a weaker envelope: the comparator asks for an
entry to go as soon as its cell is green, and this holds the deletion back
until a reply of that grade answers the cell. A cell of a reference may be
answered from several frames, and every one of them has to be of that grade
before its entry can go. An
envelope names the SDK release it was built for
in one field, `sdk_version`, as `<distribution> <version>`. A static test
holds each to what reads the dialect, the test environment or a worker's
lock, so a lock that moves fails there until the envelope was read against
the new release. The same test refuses a release of three dotted numbers in
the comment lines of an envelope, in its provenance outside `sdk_version`
and in the docstring of a module; it reads no other text. A difference that
no feature field expresses is a declared difference of the ledger, with a
product citation and the alternative contract that is asserted instead. One
of them restates a line of a clause (what C5 asks of a route under a cost
policy, above), and there the clause asserts the alternative itself; no
other one touches a clause, and none may touch the accounting clause C6.

```bash
uv sync --frozen --reinstall --link-mode copy --project backends/deepseek_worker
uv sync --frozen --reinstall --link-mode copy --project backends/kimi_worker
uv sync --frozen --project backends/codex_worker
uv run python -m pytest -q tests/backend_matrix
```

The first three commands install the locked DeepSeek, Kimi and Codex worker
environments, which their columns and the tests marked `matrix_worker` need.
The Codex environment holds the vendor's command-line runtime, which the
official SDK depends on: most of its size (about 340 MB installed), and never
started by a test. A missing or drifted environment is an error with its
command in the message, never a skip. The DeepSeek and the Kimi command
install in copy mode. The DeepSeek pin requires it: it refuses an installed
file with more than one link, which an install that links from uv's cache
leaves. The Kimi pin takes any regular file and does not require it; its
command has the same shape so that the two environments are installed one
way. The Codex command is the plain locked sync: the official route qualifies
its worker by a version probe and has no rule about links. Whether a column
can be driven is asked once per run, under the same cleared environment as
its cells: a credential, a proxy, a TLS trust path or a
`CRUCIBLE_*_WORKER_PYTHON` override of the machine decides nothing. Clone
inside the Linux filesystem: the DeepSeek pin hashes its lock byte for byte
and every installed file with its mode, so a checkout on a Windows filesystem
reached through WSL can fail it. The pin also covers the worker's source
directory with any bytecode in it, and it is checked again before every
launch: a test that loads worker source in its own process loads it without
writing bytecode (as `tests/test_backends_deepseek.py` and
`tests/test_backends_kimi.py` do), or it would change the pin under the tests
that run beside it in the same checkout. The Kimi pin hashes its source
directory and its installed tree the same way, without the rule about links.
CI installs each of them in every test shard. A test outside the matrix that
needs the Kimi environment takes the `require_kimi_worker` fixture, which
fails without it, and the `matrix_worker` marker; static cell `H01` fails
when a `skipif` marker on a worker interpreter returns to a test file.

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
cell the inventory pins under `column_wide_not_owed` with its reason: a cell
whose scenario waives the flag that asks for the check (a refusal that is an
HTTP rejection bills nothing, so it owes no accounting flag), where the cell
itself does not ask it either. A run
that writes a JUnit file also writes
`matrix-report-<n>.json` beside it: the codes, gaps, declared values, clause
outcomes and grade of every cell the process judged, and for each kind what
its driver bypasses and the private names of the product that the driver and
the required proofs of the kind's module replace. The names a required proof
replaces are listed beside the proofs (`PROOF_SEAMS`), and a static test
holds that list to what the proofs replace.

## Live tests

Docker probes and paid model calibration are separate, explicitly selected
lanes. Run only the authorized tests and exact route. Containers and model
sessions can consume substantial resources; their results need complete
cleanup and identity evidence. Hidden benchmark evaluation is a product run,
not an incidental unit-test action.

Report tests executed, failures, skipped platform checks, and any unavailable
live gates. Never claim that offline tests establish solved benchmark tasks or
promote a diagnostic backend route.
