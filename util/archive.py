import tarfile
from pathlib import Path

from simstack.models import FileStack


def pack_directory(folder: Path) -> FileStack:
    folder = Path(folder)
    archive = folder.parent / f"{folder.name}.tar.gz"
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(folder, arcname=folder.name)
    return FileStack.from_local_file(archive, in_memory=True)


def unpack_directory(archive: FileStack, dest: Path) -> Path:
    dest = Path(dest)
    staging = dest / "archive"
    staging.mkdir(parents=True, exist_ok=True)
    archive_path = archive.get(local_dir=staging)
    out = dest / "tree"
    out.mkdir()
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(out, filter="data")
    children = [path for path in out.iterdir()]
    if len(children) != 1 or not children[0].is_dir():
        raise ValueError(f"archive {archive.name} does not contain one directory")
    return children[0]
