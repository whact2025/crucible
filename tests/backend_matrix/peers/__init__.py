# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Peers: the executables a driver puts where an adapter looks for a vendor tool.

A peer is run by the adapter's own SDK, as the vendor tool would be, and takes
what the model says from the scripted provider of the cell. Each is standard
library only and free of package-relative imports: it runs in a process of its
own, outside the test environment.
"""
