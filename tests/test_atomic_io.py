from __future__ import annotations

import errno
import os
from pathlib import Path
import stat

import pytest

from gnn_siamese.utils import atomic_io
from gnn_siamese.utils.atomic_io import atomic_write_text


def _own_temporaries(destination: Path) -> list[Path]:
    return list(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_atomic_write_text_creates_destination_and_parent(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "manifest.json"

    atomic_write_text(destination, "complete content")

    assert destination.read_text(encoding="utf-8") == "complete content"
    assert _own_temporaries(destination) == []


def test_atomic_write_text_replaces_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_text("old content", encoding="utf-8")
    destination.chmod(0o644)

    atomic_write_text(destination, "new complete content")

    assert destination.read_text(encoding="utf-8") == "new complete content"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o644
    assert _own_temporaries(destination) == []


@pytest.mark.parametrize("mode", [0o640, 0o600])
def test_atomic_write_text_preserves_existing_destination_mode(tmp_path: Path, mode: int) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_text("old content", encoding="utf-8")
    destination.chmod(mode)

    atomic_write_text(destination, "new content")

    assert destination.read_text(encoding="utf-8") == "new content"
    assert stat.S_IMODE(destination.stat().st_mode) == mode


def test_atomic_write_text_new_destination_uses_regular_creation_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest.json"
    control = tmp_path / "regular.txt"
    control.write_text("control", encoding="utf-8")
    observed: dict[str, int] = {}
    real_open = os.open

    def _record_open(path, flags, mode=0o777):
        if Path(path).name.startswith(f".{destination.name}."):
            observed["requested_mode"] = mode
        return real_open(path, flags, mode)

    monkeypatch.setattr(os, "open", _record_open)
    atomic_write_text(destination, "new content")

    assert observed["requested_mode"] == 0o666
    assert stat.S_IMODE(destination.stat().st_mode) == stat.S_IMODE(control.stat().st_mode)


def test_atomic_write_text_uses_temporary_in_destination_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "run" / "manifest.json"
    observed: dict[str, Path] = {}
    real_replace = os.replace

    def _record_replace(source: str | Path, target: str | Path) -> None:
        observed["source"] = Path(source)
        observed["target"] = Path(target)
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", _record_replace)
    atomic_write_text(destination, "payload")

    assert observed["source"].parent == destination.parent
    assert observed["target"] == destination


def test_failure_before_replace_preserves_destination_and_cleans_only_own_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_text("last valid manifest", encoding="utf-8")
    unrelated = tmp_path / "unrelated.tmp"
    unrelated.write_text("keep me", encoding="utf-8")

    def _fail_fsync(file_descriptor: int) -> None:
        raise OSError("simulated file sync failure")

    monkeypatch.setattr(os, "fsync", _fail_fsync)

    with pytest.raises(OSError, match="simulated file sync failure"):
        atomic_write_text(destination, "incomplete replacement")

    assert destination.read_text(encoding="utf-8") == "last valid manifest"
    assert _own_temporaries(destination) == []
    assert unrelated.read_text(encoding="utf-8") == "keep me"


def test_fdopen_failure_closes_descriptor_and_cleans_only_own_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_text("last valid manifest", encoding="utf-8")
    unrelated = tmp_path / ".manifest.json.foreign.tmp"
    unrelated.write_text("keep me", encoding="utf-8")
    observed: dict[str, int | Path] = {}
    real_create_temporary = atomic_io._create_temporary

    def _record_create(parent: Path, destination_name: str) -> tuple[Path, int]:
        temporary_path, descriptor = real_create_temporary(parent, destination_name)
        observed["temporary_path"] = temporary_path
        observed["descriptor"] = descriptor
        return temporary_path, descriptor

    def _fail_fdopen(file_descriptor: int, *args, **kwargs):
        assert file_descriptor == observed["descriptor"]
        raise RuntimeError("simulated fdopen failure")

    monkeypatch.setattr(atomic_io, "_create_temporary", _record_create)
    monkeypatch.setattr(os, "fdopen", _fail_fdopen)

    with pytest.raises(RuntimeError, match="simulated fdopen failure"):
        atomic_write_text(destination, "replacement")

    descriptor = observed["descriptor"]
    assert isinstance(descriptor, int)
    with pytest.raises(OSError) as error:
        os.fstat(descriptor)
    assert error.value.errno == errno.EBADF
    assert not Path(observed["temporary_path"]).exists()
    assert destination.read_text(encoding="utf-8") == "last valid manifest"
    assert unrelated.read_text(encoding="utf-8") == "keep me"


def test_repeated_fdopen_failures_do_not_accumulate_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_text("last valid manifest", encoding="utf-8")
    descriptors: list[int] = []
    temporary_paths: list[Path] = []
    real_create_temporary = atomic_io._create_temporary

    def _record_create(parent: Path, destination_name: str) -> tuple[Path, int]:
        temporary_path, descriptor = real_create_temporary(parent, destination_name)
        temporary_paths.append(temporary_path)
        descriptors.append(descriptor)
        return temporary_path, descriptor

    def _fail_fdopen(file_descriptor: int, *args, **kwargs):
        raise RuntimeError("simulated repeated fdopen failure")

    monkeypatch.setattr(atomic_io, "_create_temporary", _record_create)
    monkeypatch.setattr(os, "fdopen", _fail_fdopen)
    open_descriptors_before = len(list(Path("/proc/self/fd").iterdir()))

    for _ in range(12):
        with pytest.raises(RuntimeError, match="simulated repeated fdopen failure"):
            atomic_write_text(destination, "replacement")

    open_descriptors_after = len(list(Path("/proc/self/fd").iterdir()))
    assert open_descriptors_after == open_descriptors_before
    for descriptor in descriptors:
        with pytest.raises(OSError) as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF
    assert all(not path.exists() for path in temporary_paths)
    assert destination.read_text(encoding="utf-8") == "last valid manifest"


def test_successful_fdopen_transfer_does_not_double_close_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest.json"
    observed: dict[str, int] = {}
    manual_closes: list[int] = []
    real_fdopen = os.fdopen
    real_close = os.close

    def _record_fdopen(file_descriptor: int, *args, **kwargs):
        observed["descriptor"] = file_descriptor
        return real_fdopen(file_descriptor, *args, **kwargs)

    def _record_close(file_descriptor: int) -> None:
        manual_closes.append(file_descriptor)
        real_close(file_descriptor)

    monkeypatch.setattr(os, "fdopen", _record_fdopen)
    monkeypatch.setattr(os, "close", _record_close)
    monkeypatch.setattr(atomic_io, "_fsync_directory", lambda directory: None)

    atomic_write_text(destination, "published content")

    descriptor = observed["descriptor"]
    assert descriptor not in manual_closes
    with pytest.raises(OSError) as error:
        os.fstat(descriptor)
    assert error.value.errno == errno.EBADF
    assert destination.read_text(encoding="utf-8") == "published content"


def test_fdopen_error_remains_primary_when_descriptor_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_text("last valid manifest", encoding="utf-8")
    observed: dict[str, int] = {}
    real_close = os.close

    def _fail_fdopen(file_descriptor: int, *args, **kwargs):
        observed["descriptor"] = file_descriptor
        raise RuntimeError("primary fdopen failure")

    def _close_then_fail(file_descriptor: int) -> None:
        if file_descriptor == observed.get("descriptor"):
            real_close(file_descriptor)
            raise OSError("secondary descriptor close failure")
        real_close(file_descriptor)

    monkeypatch.setattr(os, "fdopen", _fail_fdopen)
    monkeypatch.setattr(os, "close", _close_then_fail)

    with pytest.raises(RuntimeError, match="primary fdopen failure"):
        atomic_write_text(destination, "replacement")

    assert destination.read_text(encoding="utf-8") == "last valid manifest"
    assert _own_temporaries(destination) == []


def test_cleanup_failure_does_not_hide_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_text("last valid manifest", encoding="utf-8")

    def _fail_replace(source: str | Path, target: str | Path) -> None:
        raise OSError("primary publication failure")

    def _fail_unlink(self: Path, *args, **kwargs) -> None:
        raise PermissionError("secondary cleanup failure")

    monkeypatch.setattr(os, "replace", _fail_replace)
    monkeypatch.setattr(Path, "unlink", _fail_unlink)

    with pytest.raises(OSError, match="primary publication failure"):
        atomic_write_text(destination, "replacement")

    assert destination.read_text(encoding="utf-8") == "last valid manifest"


def test_permission_failure_preserves_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_text("last valid manifest", encoding="utf-8")

    def _fail_fchmod(file_descriptor: int, mode: int) -> None:
        raise PermissionError("simulated permission failure")

    monkeypatch.setattr(os, "fchmod", _fail_fchmod)

    with pytest.raises(PermissionError, match="simulated permission failure"):
        atomic_write_text(destination, "replacement")

    assert destination.read_text(encoding="utf-8") == "last valid manifest"
    assert _own_temporaries(destination) == []


def test_write_flush_fsync_close_precede_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest.json"
    events: list[str] = []
    state = {"closed": False}
    real_fdopen = os.fdopen
    real_fsync = os.fsync
    real_replace = os.replace

    class _RecordingHandle:
        def __init__(self, handle) -> None:
            self._handle = handle

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            result = self._handle.__exit__(exc_type, exc_value, traceback)
            state["closed"] = self._handle.closed
            events.append("close")
            return result

        def fileno(self) -> int:
            return self._handle.fileno()

        def write(self, content: str) -> int:
            events.append("write")
            return self._handle.write(content)

        def flush(self) -> None:
            events.append("flush")
            self._handle.flush()

    def _record_fdopen(file_descriptor: int, *args, **kwargs):
        return _RecordingHandle(real_fdopen(file_descriptor, *args, **kwargs))

    def _record_fsync(file_descriptor: int) -> None:
        events.append("fsync")
        real_fsync(file_descriptor)

    def _record_replace(source: str | Path, target: str | Path) -> None:
        source_path = Path(source)
        assert source_path.read_text(encoding="utf-8") == "complete payload"
        assert state["closed"] is True
        events.append("replace")
        real_replace(source, target)

    monkeypatch.setattr(os, "fdopen", _record_fdopen)
    monkeypatch.setattr(os, "fsync", _record_fsync)
    monkeypatch.setattr(os, "replace", _record_replace)

    atomic_write_text(destination, "complete payload")

    assert events[0:5] == ["write", "flush", "fsync", "close", "replace"]


def test_directory_close_failure_is_best_effort_after_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest.json"
    real_close = os.close
    directory_descriptor: dict[str, int] = {}
    real_open = os.open

    def _record_open(path, flags, mode=0o777):
        descriptor = real_open(path, flags, mode)
        if Path(path) == tmp_path:
            directory_descriptor["value"] = descriptor
        return descriptor

    def _fail_directory_close(file_descriptor: int) -> None:
        if file_descriptor == directory_descriptor.get("value"):
            real_close(file_descriptor)
            raise OSError("simulated directory close failure")
        real_close(file_descriptor)

    monkeypatch.setattr(os, "open", _record_open)
    monkeypatch.setattr(os, "close", _fail_directory_close)

    atomic_write_text(destination, "published content")

    assert destination.read_text(encoding="utf-8") == "published content"
