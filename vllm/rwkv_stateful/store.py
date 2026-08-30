# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class SnapshotObject:
    uri: str
    sha256: str
    size_bytes: int


class FilesystemSnapshotStore:
    """Content-addressed snapshot store with atomic per-session manifests."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.objects = self.root / "objects"
        self.sessions = self.root / "sessions"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.sessions.mkdir(parents=True, exist_ok=True)

    def _session_manifest(self, session_id: str) -> Path:
        key = hashlib.sha256(session_id.encode()).hexdigest()
        return self.sessions / f"{key}.json"

    @staticmethod
    def _atomic_write(path: Path, data: bytes, *, mode: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary_path, mode)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def put(self, session_id: str, blob: bytes) -> SnapshotObject:
        if not session_id:
            raise ValueError("session_id must not be empty")
        checksum = hashlib.sha256(blob).hexdigest()
        path = self.objects / checksum[:2] / f"{checksum}.rwkvstate"
        if not path.exists():
            self._atomic_write(path, blob, mode=0o600)
        snapshot = SnapshotObject(
            uri=path.as_uri(), sha256=checksum, size_bytes=len(blob)
        )
        manifest = {"session_id": session_id, **asdict(snapshot)}
        self._atomic_write(
            self._session_manifest(session_id),
            json.dumps(manifest, sort_keys=True).encode(),
            mode=0o600,
        )
        return snapshot

    def _path_from_uri(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise ValueError("filesystem snapshot URIs must use file://")
        path = Path(unquote(parsed.path)).resolve()
        try:
            path.relative_to(self.objects)
        except ValueError as exc:
            raise ValueError("snapshot URI is outside the configured store") from exc
        return path

    def get(self, uri: str, *, expected_sha256: str = "") -> bytes:
        path = self._path_from_uri(uri)
        blob = path.read_bytes()
        checksum = hashlib.sha256(blob).hexdigest()
        if expected_sha256 and checksum != expected_sha256:
            raise ValueError(
                f"snapshot checksum mismatch: expected {expected_sha256}, "
                f"got {checksum}"
            )
        return blob

    def get_for_session(self, session_id: str) -> tuple[SnapshotObject, bytes]:
        raw = json.loads(self._session_manifest(session_id).read_text())
        if raw.get("session_id") != session_id:
            raise ValueError("session manifest does not match requested session")
        snapshot = SnapshotObject(
            uri=raw["uri"],
            sha256=raw["sha256"],
            size_bytes=int(raw["size_bytes"]),
        )
        blob = self.get(snapshot.uri, expected_sha256=snapshot.sha256)
        if len(blob) != snapshot.size_bytes:
            raise ValueError("snapshot size does not match session manifest")
        return snapshot, blob

    def delete_session(self, session_id: str) -> None:
        self._session_manifest(session_id).unlink(missing_ok=True)


__all__ = ["FilesystemSnapshotStore", "SnapshotObject"]
