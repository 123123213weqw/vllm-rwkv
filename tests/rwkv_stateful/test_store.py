# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from vllm.rwkv_stateful.store import FilesystemSnapshotStore


def test_filesystem_store_is_content_addressed_and_session_addressable(
    tmp_path: Path,
) -> None:
    store = FilesystemSnapshotStore(tmp_path)
    first = store.put("session-a", b"state")
    second = store.put("session-b", b"state")
    assert first == second
    snapshot, blob = store.get_for_session("session-a")
    assert snapshot == first
    assert blob == b"state"
    assert Path(first.uri.removeprefix("file://")).stat().st_mode & 0o777 == 0o600
    store.delete_session("session-a")
    with pytest.raises(FileNotFoundError):
        store.get_for_session("session-a")


def test_filesystem_store_rejects_external_uri_and_bad_checksum(tmp_path: Path) -> None:
    store = FilesystemSnapshotStore(tmp_path)
    snapshot = store.put("session-a", b"state")
    with pytest.raises(ValueError, match="outside"):
        store.get("file:///etc/passwd")
    with pytest.raises(ValueError, match="checksum"):
        store.get(snapshot.uri, expected_sha256="0" * 64)
