<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# `breach` — Adversarially challenge a candidate

Breach runs bounded Breaker, test-author, Defender, and Judge stages over an
explicit candidate diff. It separates generated attack specifications from
source-grounded evidence and executable observations.

```bash
uv run crucible breach --spec /path/to/corpus --diff /path/to/candidate.patch \
  --cwd /path/to/repository --budget-usd 10
```

The Breaker proposes declarative attacks. An isolated author creates test code;
deterministic code executes the pre-fix attack. The Defender changes only
production scope, then deterministic code reruns named observations. The Judge
adjudicates evidence and target closure. Success additionally requires final
shipping build/verification. Missing observations, schema failures, persistent
blocking attacks, or non-pass adjudication cannot count as success.

A mutation journal encloses repair. Failure restores the workspace only after
writer settlement is proven; unresolved cleanup remains an exact blocking
terminal. The workflow owns attack authoring and execution through language-aware
runners and observation contracts. Models receive source and scoped write
tools; deterministic workflow code owns the verification commands.

A captured Corpus diff can be selected with `--diff-hash` instead of a file.
Budget is cumulative across attack/repair/review work. Response-repair bounds
are distinct from semantic adversarial rounds. See [rubric contracts](../../rubrics/README.md).

## Options

`uv run crucible breach --help` reports accepted values, defaults, mutual
exclusions, and the precise invocation scope.

| Option | Purpose |
|---|---|
| `--authority-digest` | Exact sha256:<hex> digest returned by encode_breach_authority(). The workflow rejects a missing pair, byte mismatch, issue mismatch, or uncovered diff path before acquiring a backend. |
| `--authority-file` | Path to a crucible.breach-authority.v2/v3 JSON packet. Parent tiers should produce it with encode_breach_authority() and must also pass --authority-digest. |
| `--backend-config` | Explicit closed routing YAML override; roles and discovery are defined by the workflow. |
| `--backend-limit-deadline-seconds` | Finite positive monotonic deadline for each provider invocation; pair with limit mode. |
| `--backend-limit-mode` | Explicit strict or diagnostic provider-limit admission; pair with the invocation deadline. |
| `--breach-tests` | Path (absolute, or relative to --cwd) for the breach test file the Defender writes attack tests to. Default: tests/test_breach.py relative to cwd. |
| `--breaker-rubric` | Rubric name (under crucible/rubrics/) for the Breaker phase. Default: ``impl_breaker``. SWE-bench dispatch passes ``swebench_impl_breaker`` to get the alternative-interpretation attack-generator specialized for GitHub-issue ambiguity. |
| `--budget-usd` | USD ceiling; consult this workflow’s per-call or cumulative budget semantics. |
| `--build-timeout-s` | Per-round build-phase subprocess timeout in seconds. Default matches _build_verify.DEFAULT_BUILD_TIMEOUT_S (60 min). Forwarded from tier when invoked as a sub-workflow. |
| `--cwd` | Working directory for build/verify subprocesses (default: process cwd). Must be the target repo root for the implementation under test. |
| `--deepswe-metadata-file` | Path to a JSON file holding the per-instance DeepSWE metadata (container_id, logs_dir, task_id, ext_id). When set, breach runs in CONTAINER MODE: the Defender receives only path-confined source/write tools and adapter-native shells are disabled, so it cannot reach the host-side tasks/<id>/tests clone. Each round's _build_verify.build_and_verify runs with gate='dev_verify' + extra_parameters={deepswe_metadata: ...} so the project-native test runner gates the round inside the authoring container — picking up the Defender's freshly-written adversarial tests via the project's own test discovery. |
| `--diagnostic-cost-policy` | Explicit diagnostic-only conservative cost interval policy; results remain non-promotable. |
| `--diff` | Path to a unified diff file to breach-test. |
| `--diff-base-treeish` | Exact full Git object ID against which the diff was captured. Required for an unbound --diff-hash; a standalone --diff path defaults to the current locked HEAD. |
| `--diff-hash` | `sha256:<hex>` corpus key for an existing diff artifact (typically Implementation.diff_ref). |
| `--implementer-read-call-limit` | Maximum consecutive source-tool reads per test-author or Defender invocation without a successful write (default: 96). |
| `--issue-text` | Original issue / intent text threaded into the Breaker rubric via the ``{spec_intent}`` placeholder. SWE-bench dispatch fills this from the per-instance ``problem_statement`` so the Breaker can attack alternative interpretations of the issue rather than only the diff. |
| `--judge-rubric` | Rubric name for the Judge phase. Default: ``breach_judge``. |
| `--max-rounds` | Maximum Breaker/Defender/Judge cycles (default: 5). |
| `--max-turns` | Maximum conversational turns in the scope reported by this command’s --help. |
| `--persistence-veto-threshold` | An attack that survives N rounds halts with a reject verdict (default: 3). |
| `--response-retry-budget` | Response retry budget. See parser help for accepted values and defaults. |
| `--spec` | Required Corpus directory; canonical version manifests live under versions/. |
| `--swebench-metadata-file` | Path to public SWE-bench authoring metadata (instance_id, repo, version, container_id). Hidden test_patch/FAIL_TO_PASS/PASS_TO_PASS fields are rejected. The Defender receives path-confined edit tools and each round executes project-visible tests in the container. |
| `--verify-timeout-s` | Per-round verification subprocess timeout in seconds (default: 3600). Forwarded from Tier when composed. |

## Evidence

Read the workflow’s typed terminal output and retained records before claiming
success. Normal Corpus runs retain manifest, saga, summary, result, and outcome
files; events, traces, and domain artifacts are conditional. Source:
[workflows/breach.py](../../workflows/breach.py).

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Breaker found no attacks, or Judge emitted `decision=pass` and every required attack had a named, passing post-fix observation; in either case the final shipping build+verify also passed. |
| `1` | `revise` remained at max rounds, the persistence-veto threshold was reached, or cumulative budget was exhausted; workspace restored. |
| `2` | Judge emitted `decision=reject`, or setup/authority/runner preflight failed before a complete adversarial result; workspace restored when a transaction had begun. |
| `3` | Breaker/Judge emitted `inconclusive`, a required post-fix observation was missing/errored, a schema/transport failure prevented adjudication, a positively quiesced mutation terminal prevented adjudication, or setup never produced a complete review; workspace restored. Unresolved cleanup raises the exact mutation terminal instead of returning a code. |
