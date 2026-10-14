"""Private, checkpointed Qwen reranking on a free Kaggle T4."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    assert Path("/kaggle/working").is_dir()
    manifests = list(Path("/kaggle/input").rglob("reranker_manifest.json"))
    assert len(manifests) == 1, "Expected one bound private dataset"
    root = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    assert manifest["machine_shape"] == "NvidiaTeslaT4"
    assert sha(Path(__file__)) == manifest["launcher_sha256"]
    for name, digest in manifest["files"].items():
        assert Path(name).name == name and sha(root / name) == digest, name
    out = Path("/kaggle/working/reranker_top100")
    out.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), out / "submitted_launcher.py")
    shutil.copy2(manifests[0], out / manifests[0].name)
    env = {**os.environ, "USE_TF": "0", "TOKENIZERS_PARALLELISM": "false",
           "HF_HOME": "/kaggle/temp/qwen-reranker-hf"}
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                    "transformers==4.57.1", "numpy==2.1.3"], check=True, env=env, timeout=1200)
    subprocess.run([sys.executable, str(root / "qwen_reranker_20261013.py"),
                    "--input", str(root), "--out", str(out), "--max-seconds", "21600"],
                   check=True, env=env, timeout=22800)
    if (out / "reranker_receipt.json").exists():
        print("COMPLETED ALL 320000 PAIRS", flush=True)
    else:
        assert (out / "reranker_checkpoint.npz").exists()
        print("PARTIAL OUTPUT: resume missing queries, do not restart", flush=True)


if __name__ == "__main__":
    main()
