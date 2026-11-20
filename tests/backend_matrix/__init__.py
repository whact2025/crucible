# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Backend conformance matrix: one set of scenarios run against every backend.

A *cell* is one scenario played against one column in one lens. Its id reads
``<kind>/<scenario>/<role>/<lens>``:

- ``kind`` is the driver kind of the column (a catalog ``(backend, transport)``
  pair, or ``toy`` for the adapter the matrix proves its oracle on);
- ``scenario`` is the stable id of the behaviour (``S01``) or of a static
  check (``P04``, ``H02``);
- ``role`` is the column's role within its kind (``primary``, ``secondary``,
  ``blocked``; a toy column is named after its governing declared values);
- ``lens`` is ``raw`` (no limit policy), ``guarded`` (the diagnostic limit
  guard bound, as a workflow pool binds it) or ``static``.

A cell computes a set of deviation codes from a closed vocabulary
(:mod:`.contract`). The ledger (:mod:`.ledger`) says which sets are known gaps:
a cell passes with an empty set, is an expected failure only when its set is
exactly the recorded one, and fails otherwise. Nothing in this package skips.
"""
