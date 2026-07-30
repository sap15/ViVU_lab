"""Small atomic filesystem publication helpers."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import BinaryIO, Callable
from uuid import uuid4


def atomic_publish(
    destination: str | Path,
    writer: Callable[[BinaryIO], None],
    *,
    validator: Callable[[Path], None] | None = None,
) -> None:
    """Publish callback-produced binary content atomically when supported.

    The callback receives an open binary file backed by an exclusive temporary
    in the destination directory. The temporary is flushed, synchronized,
    closed, and optionally validated before ``os.replace`` publishes it.
    Existing destinations retain their permission bits; new files use
    ``0o666`` filtered by the process umask. Directory synchronization is best
    effort because it is not supported consistently by every filesystem.
    """

    destination_path = Path(destination)
    parent = destination_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    existing_mode = None
    try:
        existing_mode = stat.S_IMODE(destination_path.stat().st_mode)
    except FileNotFoundError:
        pass

    temporary_path: Path | None = None
    descriptor: int | None = None
    descriptor_owned = False
    try:
        temporary_path, descriptor = _create_temporary(parent, destination_path.name)
        descriptor_owned = True
        try:
            handle = os.fdopen(descriptor, mode="w+b")
        except BaseException:
            try:
                os.close(descriptor)
            except BaseException:
                pass
            descriptor_owned = False
            raise

        descriptor_owned = False
        primary_error: BaseException | None = None
        primary_traceback = None
        try:
            if existing_mode is not None:
                os.fchmod(handle.fileno(), existing_mode)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException as exc:
            primary_error = exc
            primary_traceback = exc.__traceback__

        try:
            if primary_error is None:
                handle.__exit__(None, None, None)
            else:
                handle.__exit__(
                    type(primary_error),
                    primary_error,
                    primary_traceback,
                )
        except BaseException:
            try:
                os.fstat(descriptor)
            except OSError:
                pass
            else:
                try:
                    os.close(descriptor)
                except BaseException:
                    pass
            if primary_error is None:
                raise

        if primary_error is not None:
            raise primary_error.with_traceback(primary_traceback)

        if validator is not None:
            validator(temporary_path)

        os.replace(temporary_path, destination_path)
        _fsync_directory(parent)
    except BaseException:
        if descriptor_owned and descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException:
                pass
            descriptor_owned = False
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass
        raise


def atomic_write_text(
    destination: str | Path,
    content: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Publish complete text atomically on filesystems supporting ``os.replace``.

    The temporary file is created in the destination directory so publication
    does not cross filesystem boundaries. Existing destinations retain their
    permission bits; new files use ``0o666`` filtered by the process umask, as
    regular text-file creation does. Directory synchronization is best effort
    because opening or syncing directories is not supported everywhere.
    """

    def write_text(handle: BinaryIO) -> None:
        handle.write(content.encode(encoding))

    atomic_publish(destination, write_text)


def _create_temporary(parent: Path, destination_name: str) -> tuple[Path, int]:
    """Create an exclusive same-directory temporary with regular file permissions."""

    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC

    while True:
        path = parent / f".{destination_name}.{uuid4().hex}.tmp"
        try:
            descriptor = os.open(path, flags, 0o666)
        except FileExistsError:
            continue
        return path, descriptor


def _fsync_directory(directory: Path) -> None:
    """Best-effort synchronization of a directory entry."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY

    try:
        directory_fd = os.open(directory, flags)
    except OSError:
        return

    try:
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
    finally:
        try:
            os.close(directory_fd)
        except OSError:
            pass
