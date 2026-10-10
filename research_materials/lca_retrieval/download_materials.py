"""Download immutable numerical packages and verify their full SHA-256 hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(entry, target):
    if target.exists():
        if target.stat().st_size != entry["bytes"] or sha(target) != entry["sha256"]:
            raise ValueError(f"Existing file does not match the release: {target}")
        print(f"Already verified: {target}", flush=True)
        return
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(entry["url"], headers={"User-Agent": "code2hyp-materials-downloader/1"})
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"Downloaded file does not match the release: {temporary}")
    temporary.rename(target)
    print(f"Verified: {target} ({size:,} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", choices=("earlier", "prospective", "weighted_top100", "all"), default="weighted_top100")
    parser.add_argument("--output", type=Path, default=Path("downloads"))
    args = parser.parse_args()
    registry = json.loads(Path(__file__).with_name("releases.json").read_text())
    selected = registry if args.package == "all" else {args.package: registry[args.package]}
    for name, package in selected.items():
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        for entry in package["assets"]:
            filename = entry["name"]
            if Path(filename).name != filename or not entry["url"].startswith("https://github.com/Malomalsky/code2hyp-artifacts/releases/download/"):
                raise ValueError("Invalid release asset entry")
            download(entry, folder / filename)


if __name__ == "__main__":
    main()
