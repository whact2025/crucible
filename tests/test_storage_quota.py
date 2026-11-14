# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
from pathlib import Path

import pytest

from crucible.common.storage_quota import (
    DirectoryGrowthQuota,
    KernelQuotaBackend,
    KernelQuotaLease,
    StorageQuotaSettlementReceipt,
    StorageQuotaBackendUnavailable,
    StorageQuotaCleanupError,
    StorageQuotaExceeded,
)


_TOKEN = "a" * 43


@pytest.fixture(autouse=True)
def _without_operator_quota_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests independent of a host-installed quota helper."""

    monkeypatch.delenv("CRUCIBLE_QUOTA_HELPER", raising=False)


def test_growth_quota_allows_preexisting_baseline_and_small_growth(
    tmp_path: Path,
) -> None:
    (tmp_path / "baseline.bin").write_bytes(b"x" * 8192)

    with DirectoryGrowthQuota(
        (tmp_path,), limit_bytes=1024 * 1024, poll_seconds=0.01,
    ) as quota:
        (tmp_path / "small.bin").write_bytes(b"ok")

    assert not quota.exceeded


def test_growth_quota_trips_callback_and_raises_on_exit(tmp_path: Path) -> None:
    callbacks: list[str] = []

    with pytest.raises(StorageQuotaExceeded, match="exceeded cap"):
        with DirectoryGrowthQuota(
            (tmp_path,),
            limit_bytes=512,
            poll_seconds=0.01,
            on_exceeded=lambda: callbacks.append("kill"),
        ):
            # Filesystems allocate at least one block, so this exceeds even
            # when the apparent payload is small.
            (tmp_path / "large.bin").write_bytes(b"x" * 8192)

    assert callbacks == ["kill"]


def test_growth_quota_deduplicates_nested_roots(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()

    quota = DirectoryGrowthQuota(
        (tmp_path, nested), limit_bytes=1024, poll_seconds=0.01,
    )

    assert quota.paths == (tmp_path.resolve(),)


def test_growth_quota_reserves_and_releases_kernel_backend(tmp_path: Path) -> None:
    calls: list[tuple] = []
    guarded = tmp_path / "guarded"
    guarded.mkdir()

    class _Backend:
        command = ("quota-helper",)

        def reconcile(self):
            calls.append(("reconcile",))

        def reserve(self, paths, *, limit_bytes):
            calls.append(("reserve", paths, limit_bytes))
            return KernelQuotaLease(_TOKEN)

        def release(self, lease):
            calls.append(("release", lease.token))

        def retire(self, lease):
            assert not guarded.exists()
            calls.append(("retire", lease.token))

    with DirectoryGrowthQuota(
        (guarded,),
        limit_bytes=1024,
        poll_seconds=0.01,
        kernel_backend=_Backend(),
        require_kernel=True,
    ) as quota:
        with pytest.raises(
            StorageQuotaCleanupError,
            match="unproven initial release",
        ):
            quota.retire_kernel_lease()
        assert not any(call[0] == "retire" for call in calls)
    guarded.rmdir()
    settlement = quota.retire_kernel_lease()
    assert settlement is not None
    assert settlement == StorageQuotaSettlementReceipt.from_payload(
        settlement.to_payload()
    )
    assert settlement.close.kernel_release_proven is True
    assert settlement.close.require_kernel is True
    assert settlement.close.enforcement_outcome == "within_limit"
    assert settlement.close.paths == (str(guarded.resolve()),)
    assert _TOKEN not in json.dumps(settlement.to_payload())
    repeated = quota.retire_kernel_lease()
    assert repeated is not None
    assert repeated.fingerprint == settlement.fingerprint

    assert calls[0] == ("reconcile",)
    assert calls[1][0] == "reserve"
    assert calls[-3:] == [
        ("release", _TOKEN),
        ("retire", _TOKEN),
        ("retire", _TOKEN),
    ]


def test_growth_quota_distinguishes_kernel_release_failure_from_exceeded(
    tmp_path: Path,
) -> None:
    class _Backend:
        command = ("quota-helper",)

        def reconcile(self):
            pass

        def reserve(self, paths, *, limit_bytes):
            return KernelQuotaLease(_TOKEN)

        def release(self, lease):
            raise RuntimeError("helper unavailable during release")

        def retire(self, lease):
            pytest.fail("retirement cannot follow a failed release")

    quota = DirectoryGrowthQuota(
        (tmp_path,),
        limit_bytes=1024,
        poll_seconds=0.01,
        kernel_backend=_Backend(),
        require_kernel=True,
    )
    with pytest.raises(
        StorageQuotaCleanupError,
        match="helper unavailable during release",
    ) as caught:
        with quota:
            pass

    assert type(caught.value) is StorageQuotaCleanupError
    assert quota._close_phase is None  # noqa: SLF001 - no hidden proof retained


def test_post_workspace_retirement_failure_remains_fail_closed(
    tmp_path: Path,
) -> None:
    guarded = tmp_path / "guarded"
    guarded.mkdir()

    class _Backend:
        command = ("quota-helper",)

        def reconcile(self):
            pass

        def reserve(self, paths, *, limit_bytes):
            return KernelQuotaLease(_TOKEN)

        def release(self, lease):
            pass

        def retire(self, lease):
            raise RuntimeError("helper unavailable after workspace cleanup")

    with DirectoryGrowthQuota(
        (guarded,),
        limit_bytes=1024,
        poll_seconds=0.01,
        kernel_backend=_Backend(),
        require_kernel=True,
    ) as quota:
        pass
    guarded.rmdir()

    with pytest.raises(
        StorageQuotaCleanupError,
        match="helper unavailable after workspace cleanup",
    ):
        quota.retire_kernel_lease()


def test_exceeded_quota_retains_cleanup_evidence_after_retirement(
    tmp_path: Path,
) -> None:
    guarded = tmp_path / "guarded"
    guarded.mkdir()

    class _Backend:
        command = ("quota-helper",)

        def reconcile(self):
            pass

        def reserve(self, paths, *, limit_bytes):
            return KernelQuotaLease(_TOKEN)

        def release(self, lease):
            pass

        def retire(self, lease):
            assert not guarded.exists()

    quota = DirectoryGrowthQuota(
        (guarded,),
        limit_bytes=512,
        poll_seconds=60,
        kernel_backend=_Backend(),
        require_kernel=True,
    )
    with pytest.raises(StorageQuotaExceeded, match="exceeded cap"):
        with quota:
            (guarded / "large.bin").write_bytes(b"x" * 8192)
    (guarded / "large.bin").unlink()
    guarded.rmdir()

    settlement = quota.retire_kernel_lease()
    assert settlement is not None
    assert settlement.close.enforcement_outcome == "exceeded"


def test_scan_failure_retains_cleanup_evidence_after_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.common.storage_quota as quota_module

    guarded = tmp_path / "guarded"
    guarded.mkdir()

    class _Backend:
        command = ("quota-helper",)

        def reconcile(self):
            pass

        def reserve(self, paths, *, limit_bytes):
            return KernelQuotaLease(_TOKEN)

        def release(self, lease):
            pass

        def retire(self, lease):
            assert not guarded.exists()

    real_allocated = quota_module._allocated_bytes
    samples = 0

    def fail_final_sample(paths):
        nonlocal samples
        samples += 1
        if samples == 1:
            return real_allocated(paths)
        raise OSError("injected final scan failure")

    monkeypatch.setattr(quota_module, "_allocated_bytes", fail_final_sample)
    quota = DirectoryGrowthQuota(
        (guarded,),
        limit_bytes=1024,
        poll_seconds=60,
        kernel_backend=_Backend(),
        require_kernel=True,
    )
    quota.__enter__()
    with pytest.raises(StorageQuotaExceeded, match="scan failed closed"):
        quota.__exit__(None, None, None)
    guarded.rmdir()

    settlement = quota.retire_kernel_lease()
    assert settlement is not None
    assert settlement.close.enforcement_outcome == "scan_failed"


def test_storage_quota_settlement_parser_rejects_nonterminal_projection(
    tmp_path: Path,
) -> None:
    guarded = tmp_path / "guarded"
    guarded.mkdir()

    class _Backend:
        command = ("quota-helper",)

        def reconcile(self):
            pass

        def reserve(self, paths, *, limit_bytes):
            return KernelQuotaLease(_TOKEN)

        def release(self, lease):
            pass

        def retire(self, lease):
            pass

    with DirectoryGrowthQuota(
        (guarded,),
        limit_bytes=1024,
        poll_seconds=0.01,
        kernel_backend=_Backend(),
        require_kernel=True,
    ) as quota:
        pass
    guarded.rmdir()
    settlement = quota.retire_kernel_lease()
    assert settlement is not None
    payload = settlement.to_payload()
    close = payload["close"]
    assert type(close) is dict
    close["monitor_stopped"] = False
    with pytest.raises(ValueError, match="lacks sampling shutdown proof"):
        StorageQuotaSettlementReceipt.from_payload(payload)


def test_storage_quota_refuses_receipt_when_monitor_did_not_join(
    tmp_path: Path,
) -> None:
    quota = DirectoryGrowthQuota(
        (tmp_path,), limit_bytes=1024, poll_seconds=0.01,
    )
    quota.__enter__()
    real_thread = quota._thread  # noqa: SLF001 - deterministic failure injection
    assert real_thread is not None
    quota._stop.set()  # noqa: SLF001 - retire the real test monitor first
    real_thread.join(timeout=2)
    assert not real_thread.is_alive()

    class _StuckMonitor:
        def join(self, timeout):
            assert timeout == 2.0

        def is_alive(self):
            return True

    quota._thread = _StuckMonitor()  # type: ignore[assignment]  # noqa: SLF001

    with pytest.raises(StorageQuotaCleanupError, match="monitor did not stop"):
        quota.__exit__(None, None, None)
    assert quota._close_phase is None  # noqa: SLF001 - no hidden proof retained


def test_storage_quota_refuses_receipt_after_unexpected_monitor_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.common.storage_quota as quota_module

    guarded = tmp_path / "guarded"
    guarded.mkdir()

    class _Backend:
        command = ("quota-helper",)

        def reconcile(self):
            pass

        def reserve(self, paths, *, limit_bytes):
            return KernelQuotaLease(_TOKEN)

        def release(self, lease):
            pass

        def retire(self, lease):
            pytest.fail("retirement cannot follow monitor failure")

    baseline_sampled = threading.Event()
    fail_monitor = threading.Event()
    real_allocated = quota_module._allocated_bytes

    def fail_background_sample(paths):
        if threading.current_thread().name == "crucible-storage-quota":
            baseline_sampled.set()
            assert fail_monitor.wait(timeout=2)
            raise RuntimeError("injected unexpected monitor failure")
        return real_allocated(paths)

    monkeypatch.setattr(quota_module, "_allocated_bytes", fail_background_sample)
    quota = DirectoryGrowthQuota(
        (guarded,),
        limit_bytes=1024,
        poll_seconds=0.01,
        kernel_backend=_Backend(),
        require_kernel=True,
    )
    quota.__enter__()
    assert baseline_sampled.wait(timeout=2)
    fail_monitor.set()

    with pytest.raises(
        StorageQuotaCleanupError,
        match="unexpected monitor failure",
    ):
        quota.__exit__(None, None, None)

    assert quota._close_phase is None  # noqa: SLF001 - no hidden proof retained


def test_monitor_join_failure_still_attempts_release_and_preserves_both(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class _Backend:
        command = ("quota-helper",)

        def reconcile(self):
            pass

        def reserve(self, paths, *, limit_bytes):
            return KernelQuotaLease(_TOKEN)

        def release(self, lease):
            calls.append("release")
            raise RuntimeError("injected release failure")

        def retire(self, lease):
            pytest.fail("retirement cannot follow an unsettled close")

    quota = DirectoryGrowthQuota(
        (tmp_path,),
        limit_bytes=1024,
        poll_seconds=0.01,
        kernel_backend=_Backend(),
        require_kernel=True,
    )
    quota.__enter__()
    real_thread = quota._thread  # noqa: SLF001 - deterministic failure injection
    assert real_thread is not None
    quota._stop.set()  # noqa: SLF001 - retire the real test monitor first
    real_thread.join(timeout=2)

    class _FailedJoin:
        def join(self, timeout):
            raise RuntimeError("injected monitor join failure")

        def is_alive(self):
            pytest.fail("join failure has no liveness proof")

    quota._thread = _FailedJoin()  # type: ignore[assignment]  # noqa: SLF001

    with pytest.raises(
        StorageQuotaCleanupError,
        match="injected monitor join failure",
    ) as caught:
        quota.__exit__(None, None, None)

    assert calls == ["release"]
    assert any("release also failed" in note for note in caught.value.__notes__)
    assert quota._close_phase is None  # noqa: SLF001 - no hidden proof retained


def test_retirement_rechecks_backend_binding_after_helper_proof(
    tmp_path: Path,
) -> None:
    guarded = tmp_path / "guarded"
    guarded.mkdir()

    class _Backend:
        command = ("quota-helper",)

        def reconcile(self):
            pass

        def reserve(self, paths, *, limit_bytes):
            return KernelQuotaLease(_TOKEN)

        def release(self, lease):
            pass

        def retire(self, lease):
            self.command = ("substituted-helper",)

    with DirectoryGrowthQuota(
        (guarded,),
        limit_bytes=1024,
        poll_seconds=0.01,
        kernel_backend=_Backend(),
        require_kernel=True,
    ) as quota:
        pass
    guarded.rmdir()

    with pytest.raises(StorageQuotaCleanupError, match="changed during proof"):
        quota.retire_kernel_lease()


def test_kernel_quota_helper_protocol_is_exact_owner_and_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.common.storage_quota as quota_module

    calls: list[list[str]] = []

    def run(args, **kwargs):
        calls.append(list(args))
        # Leading '-' is valid in an opaque URL-safe token and must not be
        # mistaken for another CLI option during release.
        stdout = (b"-" + b"a" * 42 + b"\n") if "reserve" in args else b""
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(quota_module, "run_process_tree", run)
    backend = KernelQuotaBackend(("quota-helper",))

    backend.reconcile()
    lease = backend.reserve((tmp_path,), limit_bytes=4096)
    backend.release(lease)
    backend.retire(lease)

    assert calls[0] == [
        "quota-helper", "reconcile", "--owner-host", socket.gethostname(),
    ]
    reserve = calls[1]
    assert reserve[:2] == ["quota-helper", "reserve"]
    assert reserve[reserve.index("--owner-host") + 1] == socket.gethostname()
    assert reserve[reserve.index("--owner-pid") + 1] == str(os.getpid())
    assert int(reserve[reserve.index("--owner-start-token") + 1]) > 0
    assert reserve[-2:] == ["--path", str(tmp_path)]
    token = "-" + "a" * 42
    assert calls[2] == [
        "quota-helper", "release", f"--token={token}",
    ]
    assert calls[3] == [
        "quota-helper", "retire", f"--token={token}",
    ]


def test_kernel_quota_helper_release_failure_has_cleanup_error_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.common.storage_quota as quota_module

    def run(args, **kwargs):
        return subprocess.CompletedProcess(
            args, 2, stdout=b"", stderr=b"release rejected",
        )

    monkeypatch.setattr(quota_module, "run_process_tree", run)

    with pytest.raises(
        StorageQuotaCleanupError,
        match="release rejected",
    ):
        KernelQuotaBackend(("quota-helper",)).release(
            KernelQuotaLease(_TOKEN)
        )


@pytest.mark.parametrize(
    ("operation", "stdout", "stderr", "error_type"),
    (
        ("reconcile", b"unexpected\n", b"", StorageQuotaBackendUnavailable),
        ("reconcile", b"", b"warning\n", StorageQuotaBackendUnavailable),
        ("release", b"unexpected\n", b"", StorageQuotaCleanupError),
        ("release", b"", b"warning\n", StorageQuotaCleanupError),
    ),
)
def test_kernel_quota_silent_operations_reject_noisy_success(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    stdout: bytes,
    stderr: bytes,
    error_type: type[Exception],
) -> None:
    import crucible.common.storage_quota as quota_module

    def run(args, **kwargs):
        return subprocess.CompletedProcess(
            args, 0, stdout=stdout, stderr=stderr,
        )

    monkeypatch.setattr(quota_module, "run_process_tree", run)
    backend = KernelQuotaBackend(("quota-helper",))

    with pytest.raises(error_type, match="unexpected output"):
        if operation == "reconcile":
            backend.reconcile()
        else:
            backend.release(KernelQuotaLease(_TOKEN))


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    (
        (b"", b""),
        (b"a" * 43, b""),
        (b"a" * 42 + b"\n", b""),
        (b"a" * 43 + b"\nextra\n", b""),
        (b"\xff" * 43 + b"\n", b""),
        (b"a" * 43 + b"\n", b"warning\n"),
    ),
)
def test_kernel_quota_reserve_rejects_noncanonical_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: bytes,
    stderr: bytes,
) -> None:
    import crucible.common.storage_quota as quota_module

    def run(args, **kwargs):
        return subprocess.CompletedProcess(
            args, 0, stdout=stdout, stderr=stderr,
        )

    monkeypatch.setattr(quota_module, "run_process_tree", run)

    with pytest.raises(StorageQuotaBackendUnavailable):
        KernelQuotaBackend(("quota-helper",)).reserve(
            (tmp_path,), limit_bytes=4096,
        )


def test_kernel_quota_helper_retirement_failure_has_cleanup_error_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.common.storage_quota as quota_module

    def run(args, **kwargs):
        return subprocess.CompletedProcess(
            args, 2, stdout=b"", stderr=b"retirement not yet proven",
        )

    monkeypatch.setattr(quota_module, "run_process_tree", run)

    with pytest.raises(
        StorageQuotaCleanupError,
        match="retirement not yet proven",
    ):
        KernelQuotaBackend(("quota-helper",)).retire(
            KernelQuotaLease(_TOKEN)
        )


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    ((b"unexpected\n", b""), (b"", b"warning\n")),
)
def test_kernel_quota_retirement_rejects_noisy_success(
    monkeypatch: pytest.MonkeyPatch,
    stdout: bytes,
    stderr: bytes,
) -> None:
    import crucible.common.storage_quota as quota_module

    def run(args, **kwargs):
        return subprocess.CompletedProcess(
            args, 0, stdout=stdout, stderr=stderr,
        )

    monkeypatch.setattr(quota_module, "run_process_tree", run)

    with pytest.raises(StorageQuotaCleanupError, match="unexpected output"):
        KernelQuotaBackend(("quota-helper",)).retire(
            KernelQuotaLease(_TOKEN)
        )


def test_growth_quota_fails_when_kernel_is_required_without_helper(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.delenv("CRUCIBLE_QUOTA_HELPER", raising=False)
    with pytest.raises(StorageQuotaBackendUnavailable, match="required"):
        DirectoryGrowthQuota(
            (tmp_path,), limit_bytes=1024, require_kernel=True,
        )
