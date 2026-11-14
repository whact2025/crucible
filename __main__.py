# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Crucible workflow dispatcher entrypoint.

``python -m crucible <workflow> [args...]`` is equivalent to
``python -m crucible.workflows <workflow> [args...]``. Use ``--list`` to
enumerate discoverable workflow modules.
"""

import sys

from crucible.workflows.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
