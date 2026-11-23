<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Isolated DeepSeek Responses worker

This independently locked project supplies the dedicated diagnostic DeepSeek
Responses adapter. It accepts one closed request, creates a bounded client,
and disables hidden retries, redirects, and ambient endpoint discovery.

```bash
uv sync --project backends/deepseek_worker --frozen --reinstall --link-mode copy
uv run crucible backend_config_bind_deepseek \
  --template products/swebench/backends_deepseek_diagnostic.yaml \
  --output /absolute/path/to/new-run-inputs/backends_deepseek_bound.yaml
```

Copy installation is part of the runtime identity: cache hardlinks introduce
another mutation authority. The provider-free materializer authenticates the
installed worker and binds its exact fingerprint into a new routing file.
Construction does not silently bless whichever environment is currently present.

The route uses its exact configured Flash model and Responses endpoint with
streaming and client-side tools. The request alias is `deepseek-flash`; the
vendor moves it between snapshots without an id change, so the adapter attests
the returned alias, not a snapshot. Cost is recorded at the vendor's
published peak rate: it equals the charge at peak and is twice the charge
off-peak, an upper bound of the amount billed. Credentials enter only the
authorized invocation's private input channel. They never enter process
arguments, worker probe/SBOM output, or retained diagnostics. Stateless
continuation is bounded and transient; raw reasoning is not copied into public
artifacts or tool output.

The Linux supervisor proves descendant-empty settlement. Installed dependencies
must match the reviewed lock inventory. Diagnostic workspace-write opt-ins stay
confined to their declared product roles, and the route remains non-promotable.
See the [workflow reference](../../docs/workflows/backend_config_bind_deepseek.md).
