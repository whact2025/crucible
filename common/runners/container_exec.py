# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Contain process trees spawned by benchmark runner commands.

``docker exec`` does not tie an exec process tree to the client lifetime.  A
test can daemonize (or merely background a delayed write), let its foreground
shell return success, and mutate the shared workspace after the gate has
accepted it. This module stages a small static, root-owned supervisor in
the authoring container and makes that supervisor the subreaper for each
verification command.

The supervisor remains root and starts only its command child as the
container's configured non-root uid/gid. Product-owned profiles bind the
allowed mount topology without changing the supervisor mechanics. It
inventories the complete
descendant tree through ``/proc``, then TERM/KILLs and reaps every descendant
before returning.  Any process which outlives the foreground command makes the
gate fail even when containment succeeds.  If the supervisor itself times out
or dies, the source container is force-removed: continuing to use a workspace
with an unaccounted writer would not be a safe recovery.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


_SOURCE_CONTAINER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_CANONICAL_CONTAINER_ID_RE = re.compile(r"[0-9a-f]{64}")
_NUMERIC_USER_RE = re.compile(r"([1-9][0-9]*):([1-9][0-9]*)")
_TMPFS_SIZE_RE = re.compile(r"size=([1-9][0-9]*)([kmgt]?)")
_TRUSTED_DIRECTORY = "/tmp/.crucible-exec-supervisor"
_TRUSTED_BINARY = f"{_TRUSTED_DIRECTORY}/supervisor"
_COMPLETION_PREFIX = f"{_TRUSTED_DIRECTORY}/completion-"
_POISON_DIRECTORY_NAME = "poisoned-containers"
_SUPERVISOR_FAILURE_RC = 126
_SUPERVISOR_SIGNAL_RC = 120
_SUPERVISOR_TIMEOUT_RC = 124
_SUPERVISOR_DESCENDANT_RC = 125
_MAX_SUPERVISED_PROCESSES = 4096
_MIN_TMPFS_BYTES = 256 * 1024 * 1024
_MAX_TMPFS_BYTES = 4096 * 1024 * 1024
_SUPERVISOR_CLEANUP_FRACTION = 0.5
CONTAINMENT_UNUSABLE_RC = _SUPERVISOR_FAILURE_RC
# No attested command outcome can be 122: the supervisor reports a command
# status above 119 as 123, and 120, 124, 125 and 126 are its own verdicts.
CONTAINMENT_NOT_STARTED_RC = 122


@dataclass(frozen=True, slots=True)
class ContainerExecProfile:
    """Trusted container topology accepted by the exec supervisor.

    Process containment is product-neutral, but mount destinations and the
    helper-cache namespace are properties of the product sandbox.  Callers
    must select a Crucible-owned profile; repository metadata never supplies
    these values.
    """

    name: str
    workspace_destination: str
    git_destination: str
    logs_destination: str | None
    allowed_phases: frozenset[str]
    helper_cache_env: str


DEEPSWE_EXEC_PROFILE = ContainerExecProfile(
    name="deepswe",
    workspace_destination="/app",
    git_destination="/app/.git",
    logs_destination="/logs",
    allowed_phases=frozenset({"authoring", "baseline"}),
    helper_cache_env="CRUCIBLE_DEEPSWE_HELPER_DIR",
)


_SUPERVISOR_SOURCE = r'''
#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <limits.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define MAX_PROCESSES 4096
#define EXIT_CHILD_SIGNAL 120
#define EXIT_CHILD_LARGE_STATUS 123
#define EXIT_TIMEOUT 124
#define EXIT_DESCENDANTS 125
#define EXIT_INTERNAL 126
#define NONCE_HEX_LENGTH 64

struct process_entry {
    pid_t pid;
    pid_t ppid;
    bool descendant;
};

static volatile sig_atomic_t stop_signal = 0;

static void remember_signal(int signal_number) {
    stop_signal = signal_number;
}

static int64_t monotonic_ms(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return -1;
    }
    return (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

static void short_pause(void) {
    struct timespec delay = {.tv_sec = 0, .tv_nsec = 10000000};
    while (nanosleep(&delay, &delay) != 0 && errno == EINTR) {
    }
}

static bool decimal_name(const char *name) {
    if (*name == '\0') {
        return false;
    }
    for (const unsigned char *p = (const unsigned char *)name; *p; ++p) {
        if (!isdigit(*p)) {
            return false;
        }
    }
    return true;
}

static bool only_expected_processes(pid_t supervisor) {
    DIR *directory = opendir("/proc");
    if (directory == NULL) {
        return false;
    }
    bool clean = true;
    struct dirent *entry;
    while ((entry = readdir(directory)) != NULL) {
        if (!decimal_name(entry->d_name)) {
            continue;
        }
        char *end = NULL;
        long raw_pid = strtol(entry->d_name, &end, 10);
        if (end == NULL || *end != '\0' || raw_pid <= 0 || raw_pid > INT_MAX) {
            clean = false;
            break;
        }
        pid_t pid = (pid_t)raw_pid;
        if (pid != 1 && pid != supervisor) {
            clean = false;
            break;
        }
    }
    closedir(directory);
    return clean;
}

static bool valid_nonce(const char *nonce) {
    if (nonce == NULL || strlen(nonce) != NONCE_HEX_LENGTH) {
        return false;
    }
    for (size_t index = 0; index < NONCE_HEX_LENGTH; ++index) {
        char value = nonce[index];
        if (!((value >= '0' && value <= '9') ||
              (value >= 'a' && value <= 'f'))) {
            return false;
        }
    }
    return true;
}

static bool write_completion_attestation(const char *nonce, int outcome) {
    char path[256];
    int path_length = snprintf(
        path, sizeof(path),
        "/tmp/.crucible-exec-supervisor/completion-%s", nonce
    );
    if (path_length < 0 || (size_t)path_length >= sizeof(path)) {
        return false;
    }
    int descriptor = open(
        path,
        O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
        0400
    );
    if (descriptor < 0) {
        return false;
    }
    char record[128];
    int record_length = snprintf(
        record, sizeof(record), "%s %d\n", nonce, outcome
    );
    bool ok = record_length > 0 && (size_t)record_length < sizeof(record);
    size_t written = 0;
    while (ok && written < (size_t)record_length) {
        ssize_t count = write(
            descriptor, record + written, (size_t)record_length - written
        );
        if (count > 0) {
            written += (size_t)count;
        } else if (count < 0 && errno == EINTR) {
            continue;
        } else {
            ok = false;
        }
    }
    if (ok && fsync(descriptor) != 0) {
        ok = false;
    }
    if (close(descriptor) != 0) {
        ok = false;
    }
    return ok;
}

static int attest_outcome(const char *nonce, int outcome) {
    if (!write_completion_attestation(nonce, outcome)) {
        perror("crucible supervisor: completion attestation");
        return EXIT_INTERNAL;
    }
    return outcome;
}

static bool read_parent_pid(pid_t pid, pid_t *parent) {
    char path[64];
    char buffer[4096];
    snprintf(path, sizeof(path), "/proc/%ld/stat", (long)pid);
    FILE *stream = fopen(path, "re");
    if (stream == NULL) {
        return false;
    }
    bool ok = fgets(buffer, sizeof(buffer), stream) != NULL;
    fclose(stream);
    if (!ok) {
        return false;
    }
    char *close_paren = strrchr(buffer, ')');
    char state = '\0';
    long parsed_parent = -1;
    if (close_paren == NULL ||
        sscanf(close_paren + 1, " %c %ld", &state, &parsed_parent) != 2 ||
        parsed_parent < 0 || parsed_parent > INT_MAX) {
        return false;
    }
    *parent = (pid_t)parsed_parent;
    return true;
}

static size_t descendant_census(
    pid_t supervisor, struct process_entry *entries, size_t capacity
) {
    DIR *directory = opendir("/proc");
    if (directory == NULL) {
        return SIZE_MAX;
    }
    size_t count = 0;
    struct dirent *entry;
    while ((entry = readdir(directory)) != NULL) {
        if (!decimal_name(entry->d_name)) {
            continue;
        }
        char *end = NULL;
        long raw_pid = strtol(entry->d_name, &end, 10);
        if (end == NULL || *end != '\0' || raw_pid <= 0 ||
            raw_pid > INT_MAX || (pid_t)raw_pid == supervisor) {
            continue;
        }
        pid_t parent;
        if (!read_parent_pid((pid_t)raw_pid, &parent)) {
            continue;
        }
        if (count == capacity) {
            closedir(directory);
            return SIZE_MAX;
        }
        entries[count++] = (struct process_entry){
            .pid = (pid_t)raw_pid,
            .ppid = parent,
            .descendant = parent == supervisor,
        };
    }
    closedir(directory);

    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i < count; ++i) {
            if (entries[i].descendant) {
                continue;
            }
            for (size_t j = 0; j < count; ++j) {
                if (entries[j].descendant &&
                    entries[i].ppid == entries[j].pid) {
                    entries[i].descendant = true;
                    changed = true;
                    break;
                }
            }
        }
    }
    size_t descendants = 0;
    for (size_t i = 0; i < count; ++i) {
        if (entries[i].descendant) {
            entries[descendants++] = entries[i];
        }
    }
    return descendants;
}

static bool reap_adopted_children(
    pid_t foreground,
    int *foreground_status,
    bool *foreground_done,
    bool *found_descendant
) {
    for (;;) {
        int status;
        pid_t waited = waitpid(-1, &status, WNOHANG);
        if (waited > 0) {
            if (waited == foreground && !*foreground_done) {
                *foreground_status = status;
                *foreground_done = true;
            } else {
                /*
                 * A short-lived adopted child can exit between the
                 * foreground wait and the /proc census.  Its zombie is still
                 * proof that work outlived the foreground command; do not
                 * silently reap away that evidence.
                 */
                *found_descendant = true;
            }
            continue;
        }
        if (waited == 0 || errno == ECHILD) {
            return true;
        }
        if (errno != EINTR) {
            return false;
        }
    }
}

static bool terminate_descendants(
    pid_t supervisor,
    pid_t foreground,
    int *foreground_status,
    bool *foreground_done,
    int64_t term_grace_ms,
    int64_t absolute_cleanup_deadline,
    bool *found_descendant
) {
    struct process_entry entries[MAX_PROCESSES];
    int64_t cleanup_started = monotonic_ms();
    if (cleanup_started < 0) {
        return false;
    }
    int64_t available_ms = absolute_cleanup_deadline - cleanup_started;
    int64_t bounded_grace_ms = term_grace_ms;
    if (available_ms < bounded_grace_ms) {
        bounded_grace_ms = available_ms > 0 ? available_ms : 0;
    }
    int64_t term_deadline = cleanup_started + bounded_grace_ms / 2;
    int signal_number = SIGTERM;
    int64_t hard_deadline = cleanup_started + bounded_grace_ms;
    for (;;) {
        if (!reap_adopted_children(
                foreground,
                foreground_status,
                foreground_done,
                found_descendant)) {
            return false;
        }
        size_t count = descendant_census(
            supervisor, entries, MAX_PROCESSES
        );
        if (count == SIZE_MAX) {
            return false;
        }
        if (count == 0) {
            return reap_adopted_children(
                foreground,
                foreground_status,
                foreground_done,
                found_descendant
            );
        }
        *found_descendant = true;
        int64_t now = monotonic_ms();
        if (now < 0) {
            return false;
        }
        if (now >= hard_deadline) {
            /* Best-effort final kill before host-side quarantine takes over. */
            for (size_t index = count; index > 0; --index) {
                if (kill(entries[index - 1].pid, SIGKILL) != 0 &&
                    errno != ESRCH) {
                    return false;
                }
            }
            return false;
        }
        if (now >= term_deadline) {
            signal_number = SIGKILL;
        }
        /* Repeated censuses close fork-after-census races. */
        for (size_t index = count; index > 0; --index) {
            if (kill(entries[index - 1].pid, signal_number) != 0 &&
                errno != ESRCH) {
                return false;
            }
        }
        short_pause();
    }
}

static bool parse_unsigned(const char *raw, unsigned long *value) {
    if (raw == NULL || *raw == '\0' || *raw == '-') {
        return false;
    }
    errno = 0;
    char *end = NULL;
    unsigned long parsed = strtoul(raw, &end, 10);
    if (errno != 0 || end == NULL || *end != '\0') {
        return false;
    }
    *value = parsed;
    return true;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        return 0;
    }
    if (argc < 8 || strcmp(argv[6], "--") != 0 || !valid_nonce(argv[5])) {
        fprintf(stderr, "crucible supervisor: invalid invocation\n");
        return EXIT_INTERNAL;
    }
    unsigned long raw_uid, raw_gid, raw_command_deadline, raw_cleanup_deadline;
    if (!parse_unsigned(argv[1], &raw_uid) ||
        !parse_unsigned(argv[2], &raw_gid) ||
        !parse_unsigned(argv[3], &raw_command_deadline) ||
        !parse_unsigned(argv[4], &raw_cleanup_deadline) ||
        raw_uid == 0 || raw_gid == 0 ||
        raw_uid > UINT_MAX || raw_gid > UINT_MAX ||
        raw_command_deadline == 0 || raw_command_deadline > INT64_MAX ||
        raw_cleanup_deadline <= raw_command_deadline ||
        raw_cleanup_deadline > INT64_MAX) {
        fprintf(stderr, "crucible supervisor: invalid limits or identity\n");
        return EXIT_INTERNAL;
    }
    if (prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0 ||
        prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0) {
        perror("crucible supervisor: prctl");
        return EXIT_INTERNAL;
    }
    if (!only_expected_processes(getpid())) {
        fprintf(stderr,
            "crucible supervisor: pre-existing container processes detected\n");
        return EXIT_INTERNAL;
    }
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = remember_signal;
    sigemptyset(&action.sa_mask);
    const int handled_signals[] = {SIGTERM, SIGINT, SIGHUP, SIGQUIT};
    for (size_t index = 0;
         index < sizeof(handled_signals) / sizeof(handled_signals[0]);
         ++index) {
        sigaction(handled_signals[index], &action, NULL);
    }

    int64_t command_deadline = (int64_t)raw_command_deadline;
    int64_t cleanup_deadline = (int64_t)raw_cleanup_deadline;
    int64_t initial_now = monotonic_ms();
    if (initial_now < 0) {
        return EXIT_INTERNAL;
    }
    if (initial_now >= command_deadline) {
        fprintf(stderr, "crucible supervisor: command deadline expired\n");
        return attest_outcome(argv[5], EXIT_TIMEOUT);
    }

    pid_t child = fork();
    if (child < 0) {
        perror("crucible supervisor: fork");
        return EXIT_INTERNAL;
    }
    if (child == 0) {
        signal(SIGCHLD, SIG_DFL);
        signal(SIGTERM, SIG_DFL);
        signal(SIGINT, SIG_DFL);
        signal(SIGHUP, SIG_DFL);
        if (setgroups(0, NULL) != 0 ||
            setresgid((gid_t)raw_gid, (gid_t)raw_gid, (gid_t)raw_gid) != 0 ||
            setresuid((uid_t)raw_uid, (uid_t)raw_uid, (uid_t)raw_uid) != 0) {
            perror("crucible supervisor: identity drop");
            _exit(EXIT_INTERNAL);
        }
        if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
            perror("crucible supervisor: no_new_privs");
            _exit(EXIT_INTERNAL);
        }
        execvp(argv[7], &argv[7]);
        perror("crucible supervisor: exec");
        _exit(127);
    }

    int foreground_status = 0;
    bool foreground_done = false;
    bool timed_out = false;
    while (!foreground_done) {
        pid_t waited = waitpid(child, &foreground_status, WNOHANG);
        if (waited == child) {
            foreground_done = true;
            break;
        }
        if (waited < 0 && errno != EINTR) {
            perror("crucible supervisor: waitpid");
            break;
        }
        int64_t now = monotonic_ms();
        if (now < 0 || now >= command_deadline || stop_signal != 0) {
            timed_out = stop_signal == 0;
            break;
        }
        short_pause();
    }

    bool found_descendant = !foreground_done;
    if (!terminate_descendants(
            getpid(),
            child,
            &foreground_status,
            &foreground_done,
            cleanup_deadline - command_deadline,
            cleanup_deadline,
            &found_descendant)) {
        fprintf(stderr,
            "crucible supervisor: descendant containment failed\n");
        return EXIT_INTERNAL;
    }

    int outcome;
    if (stop_signal != 0) {
        fprintf(stderr, "crucible supervisor: interrupted during command\n");
        outcome = EXIT_INTERNAL;
    } else if (timed_out) {
        fprintf(stderr, "crucible supervisor: command deadline expired\n");
        outcome = EXIT_TIMEOUT;
    } else if (found_descendant) {
        fprintf(stderr,
            "crucible supervisor: command left background descendants; "
            "all descendants were terminated\n");
        outcome = EXIT_DESCENDANTS;
    } else if (!foreground_done) {
        outcome = EXIT_INTERNAL;
    } else if (WIFSIGNALED(foreground_status)) {
        fprintf(stderr, "crucible supervisor: command ended by signal %d\n",
            WTERMSIG(foreground_status));
        outcome = EXIT_CHILD_SIGNAL;
    } else if (!WIFEXITED(foreground_status)) {
        outcome = EXIT_INTERNAL;
    } else {
        int child_status = WEXITSTATUS(foreground_status);
        outcome = child_status <= 119 ? child_status : EXIT_CHILD_LARGE_STATUS;
    }
    return attest_outcome(argv[5], outcome);
}
'''


class ExecContainmentError(RuntimeError):
    """The trusted exec supervisor could not be prepared or invoked.

    Without an explicit ``returncode`` the error means that Docker or the
    supervisor failed before the command could start.
    """

    def __init__(
        self,
        message: str,
        *,
        returncode: int = CONTAINMENT_NOT_STARTED_RC,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise ExecContainmentError(
            "development-verifier containment deadline expired",
            returncode=_SUPERVISOR_TIMEOUT_RC,
        )
    return remaining


def _run_docker(
    args: list[str],
    *,
    deadline: float,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_remaining(deadline),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExecContainmentError(
            "development-verifier Docker operation timed out",
            returncode=_SUPERVISOR_TIMEOUT_RC,
        ) from exc
    except FileNotFoundError as exc:
        raise ExecContainmentError(
            f"docker not on PATH: {exc}", returncode=127,
        ) from exc
    if check and result.returncode != 0:
        raise ExecContainmentError(
            f"docker {' '.join(args[:2])} failed rc={result.returncode}: "
            f"{(result.stderr or '')[-1000:]}",
        )
    return result


def _validate_source_container(
    container_id: str,
    *,
    deadline: float,
    profile: ContainerExecProfile = DEEPSWE_EXEC_PROFILE,
) -> tuple[int, int, str, Path, Path | None]:
    result = _run_docker(
        ["inspect", "--", container_id], deadline=deadline,
    )
    try:
        payload = json.loads(result.stdout)
        if not isinstance(payload, list) or len(payload) != 1:
            raise ValueError("expected one container")
        inspected = payload[0]
        config = inspected["Config"]
        host_config = inspected["HostConfig"]
        state = inspected["State"]
        canonical_id = inspected["Id"]
        labels = config.get("Labels") or {}
        mounts = inspected.get("Mounts") or []
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExecContainmentError(
            "source container inspect returned an invalid shape"
        ) from exc
    if (
        not isinstance(canonical_id, str)
        or _CANONICAL_CONTAINER_ID_RE.fullmatch(canonical_id) is None
    ):
        raise ExecContainmentError(
            "source container inspect returned an invalid canonical id"
        )

    if state.get("Running") is not True:
        raise ExecContainmentError("source container is not running")
    if labels.get("io.crucible.managed") != "true" or labels.get(
        "io.crucible.phase"
    ) not in profile.allowed_phases:
        raise ExecContainmentError(
            f"{profile.name} contained execution requires a managed "
            f"container phase in {sorted(profile.allowed_phases)!r}"
        )
    if config.get("Entrypoint") != ["/bin/sleep"] or config.get("Cmd") != [
        "infinity"
    ]:
        raise ExecContainmentError(
            f"{profile.name} contained execution requires the managed "
            "sleep/infinity entrypoint profile"
        )
    match = _NUMERIC_USER_RE.fullmatch(str(config.get("User") or ""))
    if match is None:
        raise ExecContainmentError(
            "source container must configure an unambiguous non-root uid:gid"
        )
    uid, gid = int(match.group(1)), int(match.group(2))

    hidden_roots = {PurePosixPath("/tests"), PurePosixPath("/solution")}
    workspace_destination = PurePosixPath(profile.workspace_destination)
    git_destination = PurePosixPath(profile.git_destination)
    logs_destination = (
        PurePosixPath(profile.logs_destination)
        if profile.logs_destination is not None else None
    )
    if (
        not workspace_destination.is_absolute()
        or not git_destination.is_absolute()
        or git_destination != workspace_destination / ".git"
        or logs_destination is not None
        and not logs_destination.is_absolute()
    ):
        raise ExecContainmentError("contained-exec profile paths are invalid")
    required_mounts = {
        workspace_destination: True,
        git_destination: False,
    }
    if logs_destination is not None:
        required_mounts[logs_destination] = True
    destinations: dict[PurePosixPath, dict[str, Any]] = {}
    for mount in mounts:
        if not isinstance(mount, dict):
            raise ExecContainmentError("source container has invalid mounts")
        destination = PurePosixPath(str(mount.get("Destination") or ""))
        if destination in destinations:
            raise ExecContainmentError(
                "source container has duplicate mount destinations"
            )
        destinations[destination] = mount
        if any(
            destination == root or root in destination.parents
            for root in hidden_roots
        ):
            raise ExecContainmentError(
                f"{profile.name} contained-exec source exposes hidden-test "
                "mounts"
            )
    if set(destinations) != set(required_mounts):
        raise ExecContainmentError(
            f"{profile.name} contained-exec source has an unexpected host "
            f"mount; required destinations are "
            f"{sorted(str(path) for path in required_mounts)!r}"
        )
    for destination, writable in required_mounts.items():
        mount = destinations[destination]
        source = str(mount.get("Source") or "")
        if (
            mount.get("Type") != "bind"
            or mount.get("RW") is not writable
            or not PurePosixPath(source).is_absolute()
        ):
            mode = "writable" if writable else "read-only"
            raise ExecContainmentError(
                f"{profile.name} contained execution requires a host bind "
                "mounted "
                f"{mode} at {destination}"
            )
    app_source = PurePosixPath(
        str(destinations[workspace_destination]["Source"])
    )
    git_source = PurePosixPath(
        str(destinations[git_destination]["Source"])
    )
    logs_source = (
        PurePosixPath(str(destinations[logs_destination]["Source"]))
        if logs_destination is not None else None
    )
    if (
        git_source != app_source / ".git"
        or app_source == PurePosixPath("/")
        or logs_source == PurePosixPath("/")
        or logs_source is not None
        and (
            app_source == logs_source
            or app_source in logs_source.parents
            or logs_source in app_source.parents
        )
    ):
        raise ExecContainmentError(
            f"{profile.name} contained-exec host mount sources are not isolated"
        )

    cap_drop = {str(value).upper() for value in host_config.get("CapDrop") or []}
    cap_add = {str(value).upper().removeprefix("CAP_") for value in (
        host_config.get("CapAdd") or []
    )}
    security_options = host_config.get("SecurityOpt") or []
    no_new_privileges = False
    unexpected_security_option = False
    for raw_option in security_options:
        key, separator, value = str(raw_option).casefold().partition("=")
        if not separator:
            key, separator, value = key.partition(":")
        if key != "no-new-privileges" or (
            separator and value not in {"1", "true"}
        ):
            unexpected_security_option = True
        else:
            no_new_privileges = True

    tmpfs = host_config.get("Tmpfs")
    tmp_options = (
        [option.casefold() for option in str(tmpfs.get("/tmp", "")).split(",")]
        if isinstance(tmpfs, dict) and set(tmpfs) == {"/tmp"}
        else []
    )
    tmp_size_tokens = [
        match
        for option in tmp_options
        if (match := _TMPFS_SIZE_RE.fullmatch(option)) is not None
    ]
    tmp_size_bytes = 0
    if len(tmp_size_tokens) == 1:
        amount = int(tmp_size_tokens[0].group(1))
        unit = tmp_size_tokens[0].group(2)
        multiplier = {
            "": 1,
            "k": 1024,
            "m": 1024**2,
            "g": 1024**3,
            "t": 1024**4,
        }[unit]
        tmp_size_bytes = amount * multiplier
    expected_tmp_options = {"rw", "exec", "nosuid", "nodev"}
    pids_limit = host_config.get("PidsLimit")
    if (
        cap_drop != {"ALL"}
        or cap_add != {"KILL", "SETGID", "SETUID"}
        or not no_new_privileges
        or unexpected_security_option
        or host_config.get("Privileged") is not False
        or host_config.get("NetworkMode") != "none"
        or host_config.get("ReadonlyRootfs") is not True
        or host_config.get("PidMode") != ""
        or host_config.get("IpcMode") not in {"", "private"}
        or host_config.get("UTSMode") != ""
        or host_config.get("UsernsMode") != ""
        or host_config.get("CgroupnsMode") not in {"", "private"}
        or host_config.get("Devices") != []
        or host_config.get("DeviceRequests") not in (None, [])
        or isinstance(pids_limit, bool)
        or not isinstance(pids_limit, int)
        or not 1 <= pids_limit <= _MAX_SUPERVISED_PROCESSES
        or len(tmp_options) != 5
        or set(tmp_options) - {match.group(0) for match in tmp_size_tokens}
        != expected_tmp_options
        or not _MIN_TMPFS_BYTES <= tmp_size_bytes <= _MAX_TMPFS_BYTES
    ):
        raise ExecContainmentError(
            "source container does not satisfy the contained-exec security "
            "profile"
        )
    return (
        uid,
        gid,
        canonical_id,
        Path(str(app_source)),
        Path(str(logs_source)) if logs_source is not None else None,
    )


def _helper_cache_directory(
    profile: ContainerExecProfile = DEEPSWE_EXEC_PROFILE,
) -> Path:
    configured = os.environ.get(profile.helper_cache_env, "").strip()
    if not configured:
        configured = os.environ.get(
            "CRUCIBLE_CONTAINER_EXEC_HELPER_DIR", "",
        ).strip()
    if configured:
        return Path(configured).expanduser().resolve()
    data_root = Path("/data")
    if data_root.is_dir() and os.access(data_root, os.W_OK | os.X_OK):
        return data_root / "crucible" / "helpers"
    return Path.home() / ".cache" / "crucible" / "helpers"


def _validated_helper_cache_directory(
    *, app_source: Path, logs_source: Path | None,
    profile: ContainerExecProfile = DEEPSWE_EXEC_PROFILE,
) -> Path:
    """Return a host-only cache path inaccessible through writable mounts."""

    directory = _helper_cache_directory(profile).resolve()
    sources = [app_source.resolve()]
    if logs_source is not None:
        sources.append(logs_source.resolve())
    for source in sources:
        if directory == source or source in directory.parents:
            raise ExecContainmentError(
                "exec-helper cache overlaps a candidate-writable host mount"
            )
    return directory


def _valid_cached_helper(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and metadata.st_mode & 0o022 == 0
        and os.access(path, os.X_OK)
    )


def _read_helper_nofollow(path: Path) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ExecContainmentError(
            f"cannot open trusted exec helper without symlink traversal: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o022 != 0
        ):
            raise ExecContainmentError(
                "trusted exec helper changed after cache validation"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _probe_helper_nofollow(
    path: Path, *, deadline: float,
) -> subprocess.CompletedProcess[bytes]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o022 != 0
            or metadata.st_mode & 0o111 == 0
        ):
            raise ExecContainmentError(
                "trusted exec helper changed before self-test"
            )
        return subprocess.run(
            [f"/proc/self/fd/{descriptor}", "--self-test"],
            pass_fds=(descriptor,),
            capture_output=True,
            timeout=min(5.0, _remaining(deadline)),
            check=False,
        )
    finally:
        os.close(descriptor)


def _secure_private_directory(directory: Path, *, label: str) -> None:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        directory.chmod(0o700)
        metadata = directory.lstat()
    except OSError as exc:
        raise ExecContainmentError(
            f"cannot secure {label} {directory}: {exc}"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077 != 0
    ):
        raise ExecContainmentError(f"{label} is not a private owned directory")


def _poison_path(directory: Path, container_id: str) -> Path:
    return directory / _POISON_DIRECTORY_NAME / container_id


def _poisoned_source_reason(directory: Path, container_id: str) -> str | None:
    path = _poison_path(directory, container_id)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"unreadable durable poison marker: {exc}"
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077 != 0
    ):
        return "invalid durable poison marker"
    try:
        return _read_helper_nofollow(path).decode("utf-8", errors="replace")[:2000]
    except ExecContainmentError as exc:
        return f"unreadable durable poison marker: {exc}"


def _poison_source(
    directory: Path,
    container_id: str,
    reason: str,
    *,
    existing_ok: bool = True,
) -> tuple[bool, str]:
    poison_directory = directory / _POISON_DIRECTORY_NAME
    try:
        _secure_private_directory(
            poison_directory, label="exec-containment poison directory",
        )
        path = _poison_path(directory, container_id)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        payload = (reason.strip() or "unattested contained execution")[:2000].encode(
            "utf-8", errors="replace",
        )
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            return (
                existing_ok,
                "source container already has a durable in-flight/unusable "
                "marker",
            )
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("durable poison marker write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory_fd = os.open(poison_directory, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True, "source container durably marked unusable"
    except (ExecContainmentError, OSError) as exc:
        return False, f"FAILED to persist source-container poison marker: {exc}"


def _clear_source_poison(directory: Path, container_id: str) -> str:
    path = _poison_path(directory, container_id)
    poison_directory = path.parent
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077 != 0
        ):
            return "durable in-flight marker changed before completion"
        path.unlink()
        directory_fd = os.open(poison_directory, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        return f"could not clear durable in-flight marker: {exc}"
    return ""


def _acquire_helper_lock(lock_file: Any, *, deadline: float) -> None:
    """Acquire the cross-process compiler lock within the caller's deadline."""

    while True:
        try:
            fcntl.flock(
                lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
            return
        except BlockingIOError:
            time.sleep(min(0.05, _remaining(deadline)))
        except OSError as exc:
            raise ExecContainmentError(
                f"cannot lock exec-helper cache: {exc}"
            ) from exc


def _ensure_supervisor(
    *,
    deadline: float,
    directory: Path | None = None,
    profile: ContainerExecProfile = DEEPSWE_EXEC_PROFILE,
) -> Path:
    source_digest = hashlib.sha256(
        _SUPERVISOR_SOURCE.encode("utf-8")
    ).hexdigest()[:20]
    directory = (
        _helper_cache_directory(profile) if directory is None else directory
    )
    _secure_private_directory(directory, label="exec-helper cache")
    helper = directory / f"container-exec-supervisor-{source_digest}"
    lock_path = directory / ".container-exec-supervisor.lock"
    with lock_path.open("a+b") as lock_file:
        _acquire_helper_lock(lock_file, deadline=deadline)
        if _valid_cached_helper(helper):
            try:
                probe = _probe_helper_nofollow(helper, deadline=deadline)
            except (ExecContainmentError, OSError, subprocess.SubprocessError):
                probe = None
            if probe is not None and probe.returncode == 0:
                return helper
            helper.unlink(missing_ok=True)

        source_fd, source_name = tempfile.mkstemp(
            prefix=".exec-supervisor-", suffix=".c", dir=directory,
        )
        output_fd, output_name = tempfile.mkstemp(
            prefix=".exec-supervisor-", dir=directory,
        )
        os.close(output_fd)
        source_path = Path(source_name)
        output_path = Path(output_name)
        try:
            with os.fdopen(source_fd, "w", encoding="utf-8") as source_file:
                source_file.write(_SUPERVISOR_SOURCE)
            try:
                compiled = subprocess.run(
                    [
                        "cc", "-std=c11", "-O2", "-static", "-s",
                        "-Wall", "-Wextra", "-Werror",
                        str(source_path), "-o", str(output_path),
                    ],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=_remaining(deadline),
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ExecContainmentError(
                    "static exec-supervisor build timed out",
                    returncode=_SUPERVISOR_TIMEOUT_RC,
                ) from exc
            except FileNotFoundError as exc:
                raise ExecContainmentError(
                    "cc is required to build the static exec supervisor",
                    returncode=127,
                ) from exc
            if compiled.returncode != 0:
                raise ExecContainmentError(
                    "static exec-supervisor build failed: "
                    f"{(compiled.stderr or '')[-2000:]}"
                )
            output_path.chmod(0o500)
            probe = subprocess.run(
                [str(output_path), "--self-test"],
                capture_output=True,
                timeout=min(5.0, _remaining(deadline)),
                check=False,
            )
            if probe.returncode != 0:
                raise ExecContainmentError(
                    "static exec-supervisor self-test failed"
                )
            os.replace(output_path, helper)
            helper.chmod(0o500)
            return helper
        finally:
            source_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


def _cleanup_staged_helper(container_id: str, *, deadline: float) -> str:
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        return "trusted helper cleanup deadline expired"
    try:
        result = subprocess.run(
            [
                "docker", "exec", "--user", "0:0", container_id,
                "/bin/rm", "-rf", _TRUSTED_DIRECTORY,
            ],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=remaining,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"trusted helper cleanup failed: {exc}"
    if result.returncode != 0:
        return (
            f"trusted helper cleanup failed rc={result.returncode}: "
            f"{(result.stderr or '')[-1000:]}"
        )
    return ""


def _stage_supervisor(
    container_id: str,
    helper: Path,
    *,
    deadline: float,
) -> None:
    """Stream the static helper into the private tmpfs as root.

    Docker's archive-based ``cp`` path cannot address a container tmpfs on
    every storage driver.  Stream through ``dd`` directly, after atomically
    replacing the candidate-inaccessible root-owned staging directory.  No
    shell startup files or candidate-controlled text are evaluated as root.
    """

    _run_docker(
        [
            "exec", "--user", "0:0", container_id,
            "/bin/rm", "-rf", _TRUSTED_DIRECTORY,
        ],
        deadline=deadline,
    )
    _run_docker(
        [
            "exec", "--user", "0:0", container_id,
            "/bin/mkdir", "-m", "0700", _TRUSTED_DIRECTORY,
        ],
        deadline=deadline,
    )
    try:
        payload = _read_helper_nofollow(helper)
        result = subprocess.run(
            [
                "docker", "exec", "-i", "--user", "0:0", container_id,
                "/bin/dd", f"of={_TRUSTED_BINARY}", "bs=1048576",
            ],
            input=payload,
            capture_output=True,
            timeout=_remaining(deadline),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExecContainmentError(
            "static exec-supervisor staging timed out",
            returncode=_SUPERVISOR_TIMEOUT_RC,
        ) from exc
    except OSError as exc:
        raise ExecContainmentError(
            f"cannot stage static exec supervisor: {exc}", returncode=127,
        ) from exc
    if result.returncode != 0:
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")
        raise ExecContainmentError(
            "static exec-supervisor staging failed "
            f"rc={result.returncode}: {stderr[-1000:]}"
        )
    _run_docker(
        [
            "exec", "--user", "0:0", container_id,
            "/bin/chmod", "0500", _TRUSTED_BINARY,
        ],
        deadline=deadline,
    )


def _completion_attestation_error(
    container_id: str,
    *,
    nonce: str,
    expected_returncode: int,
    deadline: float,
) -> str:
    path = f"{_COMPLETION_PREFIX}{nonce}"
    try:
        completed = _run_docker(
            [
                "exec", "--user", "0:0", container_id,
                "/bin/cat", path,
            ],
            deadline=deadline,
            check=False,
        )
    except ExecContainmentError as exc:
        return f"completion attestation could not be read: {exc}"
    if completed.returncode != 0:
        return (
            "completion attestation is missing or unreadable: "
            f"rc={completed.returncode} stderr={(completed.stderr or '')[-500:]}"
        )
    expected = f"{nonce} {expected_returncode}\n"
    if completed.stdout != expected:
        return "completion attestation did not match nonce and Docker returncode"
    return ""


def _quarantine_source(container_id: str, *, deadline: float) -> tuple[bool, str]:
    failure = "source-container quarantine did not run"
    for attempt in range(3):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            failure = "source-container quarantine cleanup deadline expired"
            break
        try:
            result = subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=remaining,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            failure = f"source-container quarantine failed: {exc}"
        else:
            if result.returncode == 0:
                return (
                    True,
                    "source container force-removed after containment failure",
                )
            detail = (result.stderr or "")[-1000:]
            normalized = detail.casefold()
            if "no such container" in normalized or "no such object" in normalized:
                return (
                    True,
                    "source container already absent after containment failure",
                )
            failure = (
                f"source-container quarantine failed rc={result.returncode}: "
                f"{detail}"
            )
        if attempt < 2:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(0.05, remaining / 2))
    return False, failure


def _make_source_unusable(
    container_id: str,
    *,
    cache_directory: Path,
    reason: str,
    cleanup_deadline: float,
) -> str:
    poisoned, poison = _poison_source(cache_directory, container_id, reason)
    quarantined, quarantine = _quarantine_source(
        container_id, deadline=cleanup_deadline,
    )
    state = "quarantine_confirmed" if quarantined else "quarantine_FAILED"
    poison_state = "poison_confirmed" if poisoned else "poison_FAILED"
    return (
        f"{reason}; {poison_state}: {poison}; {state}: {quarantine}"
    )


def run_contained_docker_exec(
    container_id: str,
    command: str,
    *,
    timeout: float,
    cleanup_timeout: float = 5.0,
    profile: ContainerExecProfile = DEEPSWE_EXEC_PROFILE,
) -> tuple[int, str, str]:
    """Run one command and prove that none of its descendants survive.

    Returns ``(returncode, stdout, stderr)``.  The return code is

    - ``CONTAINMENT_NOT_STARTED_RC`` when the invocation was invalid or Docker
      or the supervisor failed before the command started: nothing ran, and
      the source container stays usable unless its in-flight marker could
      not be withdrawn (the next call then reports the container unusable);
    - 124 when the execution window closed before the command started, and
      127 when Docker or the helper compiler is missing on the host: nothing
      ran either;
    - ``CONTAINMENT_UNUSABLE_RC`` when the source container is unusable: its
      in-flight marker could not be armed, or the command started and its
      completion could not be attested;
    - otherwise the supervisor's attested outcome for the command, 124 for a
      command that exceeded the execution window.
    """

    if not isinstance(container_id, str) or not _SOURCE_CONTAINER_RE.fullmatch(
        container_id
    ):
        return CONTAINMENT_NOT_STARTED_RC, "", "invalid source container id"
    if not isinstance(command, str) or not command:
        return (
            CONTAINMENT_NOT_STARTED_RC,
            "",
            "contained command must be non-empty",
        )
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or isinstance(cleanup_timeout, bool)
        or not isinstance(cleanup_timeout, (int, float))
        or not math.isfinite(float(cleanup_timeout))
        or timeout <= 0.0
        or cleanup_timeout <= 0.0
    ):
        return 124, "", "contained command has no execution window"

    execution_deadline = time.monotonic() + timeout
    cleanup_deadline = execution_deadline + cleanup_timeout
    # The supervisor must finish before the shared hard deadline so the host
    # retains a bounded opportunity to attest completion, remove the staged
    # helper, or force-remove an unattested source container. Every operation
    # still shares ``cleanup_deadline`` as its outer cap.
    supervisor_cleanup_deadline = (
        execution_deadline
        + cleanup_timeout * _SUPERVISOR_CLEANUP_FRACTION
    )
    staged = False
    command_started = False
    marker_armed = False
    result: subprocess.CompletedProcess[str] | None = None
    cache_directory: Path | None = None
    try:
        (
            uid,
            gid,
            canonical_id,
            app_source,
            logs_source,
        ) = _validate_source_container(
            container_id,
            deadline=execution_deadline,
            profile=profile,
        )
        # Names and short IDs can be rebound after inspect. Pin every root exec,
        # cleanup, and quarantine operation to Docker's immutable full ID.
        container_id = canonical_id
        cache_directory = _validated_helper_cache_directory(
            app_source=app_source,
            logs_source=logs_source,
            profile=profile,
        )
        _secure_private_directory(cache_directory, label="exec-helper cache")
        poisoned_reason = _poisoned_source_reason(
            cache_directory, container_id,
        )
        if poisoned_reason is not None:
            diagnostic = _make_source_unusable(
                container_id,
                cache_directory=cache_directory,
                reason=(
                    "source container is durably unusable from a prior "
                    f"unattested execution: {poisoned_reason}"
                ),
                cleanup_deadline=cleanup_deadline,
            )
            return CONTAINMENT_UNUSABLE_RC, "", diagnostic
        nonce = secrets.token_hex(32)
        armed, arm_diagnostic = _poison_source(
            cache_directory,
            container_id,
            f"contained execution in flight: nonce={nonce}",
            existing_ok=False,
        )
        if not armed:
            return (
                CONTAINMENT_UNUSABLE_RC,
                "",
                "refusing to start without a durable in-flight marker: "
                f"{arm_diagnostic}",
            )
        marker_armed = True
        helper = _ensure_supervisor(
            deadline=execution_deadline,
            directory=cache_directory,
            profile=profile,
        )
        staged = True
        _stage_supervisor(container_id, helper, deadline=execution_deadline)
        _remaining(execution_deadline)
        execution_deadline_ms = max(1, int(execution_deadline * 1000))
        supervisor_cleanup_deadline_ms = max(
            execution_deadline_ms + 1,
            int(supervisor_cleanup_deadline * 1000),
        )
        command_started = True
        result = _run_docker(
            [
                "exec", "--user", "0:0", container_id,
                _TRUSTED_BINARY,
                str(uid), str(gid),
                str(execution_deadline_ms),
                str(supervisor_cleanup_deadline_ms),
                nonce, "--",
                "/bin/bash", "-c", command,
            ],
            deadline=cleanup_deadline,
            check=False,
        )
        attestation_error = _completion_attestation_error(
            container_id,
            nonce=nonce,
            expected_returncode=result.returncode,
            deadline=cleanup_deadline,
        )
        if attestation_error:
            assert cache_directory is not None
            diagnostic = _make_source_unusable(
                container_id,
                cache_directory=cache_directory,
                reason=attestation_error,
                cleanup_deadline=cleanup_deadline,
            )
            staged = False
            stderr = f"{result.stderr or ''}\n{diagnostic}".strip()
            return CONTAINMENT_UNUSABLE_RC, result.stdout or "", stderr
        if (
            result.returncode == _SUPERVISOR_FAILURE_RC
            or result.returncode >= 128
            # A command that ran is never reported as one that did not.
            or result.returncode == CONTAINMENT_NOT_STARTED_RC
        ):
            assert cache_directory is not None
            diagnostic = _make_source_unusable(
                container_id,
                cache_directory=cache_directory,
                reason=(
                    "trusted supervisor reported an internal or signal failure "
                    f"after attested cleanup: rc={result.returncode}"
                ),
                cleanup_deadline=cleanup_deadline,
            )
            staged = False
            stderr = f"{result.stderr or ''}\n{diagnostic}".strip()
            return CONTAINMENT_UNUSABLE_RC, result.stdout or "", stderr
        clear_error = _clear_source_poison(cache_directory, container_id)
        if clear_error:
            diagnostic = _make_source_unusable(
                container_id,
                cache_directory=cache_directory,
                reason=(
                    "completion was attested but durable in-flight state "
                    f"could not be cleared: {clear_error}"
                ),
                cleanup_deadline=cleanup_deadline,
            )
            staged = False
            stderr = f"{result.stderr or ''}\n{diagnostic}".strip()
            return CONTAINMENT_UNUSABLE_RC, result.stdout or "", stderr
        marker_armed = False
    except ExecContainmentError as exc:
        diagnostics = str(exc)
        if command_started:
            assert cache_directory is not None
            diagnostics = _make_source_unusable(
                container_id,
                cache_directory=cache_directory,
                reason=(
                    "contained Docker execution ended without a completion "
                    f"attestation: {diagnostics}"
                ),
                cleanup_deadline=cleanup_deadline,
            )
            staged = False
            return CONTAINMENT_UNUSABLE_RC, "", diagnostics
        if marker_armed and cache_directory is not None:
            clear_error = _clear_source_poison(cache_directory, container_id)
            if clear_error:
                diagnostics = f"{diagnostics}; {clear_error}"
            else:
                marker_armed = False
        return exc.returncode, "", diagnostics
    finally:
        if staged:
            cleanup_error = _cleanup_staged_helper(
                container_id, deadline=cleanup_deadline,
            )
        else:
            cleanup_error = ""

    assert result is not None
    stderr = result.stderr or ""
    returncode = result.returncode
    if cleanup_error:
        stderr = f"{stderr}\n{cleanup_error}".strip()
        if returncode == 0:
            returncode = _SUPERVISOR_DESCENDANT_RC
    return returncode, result.stdout or "", stderr


__all__ = [
    "CONTAINMENT_NOT_STARTED_RC",
    "CONTAINMENT_UNUSABLE_RC",
    "ContainerExecProfile",
    "DEEPSWE_EXEC_PROFILE",
    "ExecContainmentError",
    "run_contained_docker_exec",
]
