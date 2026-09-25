"""Verify the Hugging Face release: re-download every uploaded file and compare sha256 + size.

This is an INDEPENDENT check — it does not trust the upload step. It reads the manifest, downloads
each artifact from the Hub over direct HTTPS (proxies disabled, which matters in the authoring
sandbox), hashes the bytes it received, and compares against the locally-computed hash.

Note: `hf_hub_download` is deliberately NOT used here — in this environment it returned an empty
file (sha256 e3b0c442...), which would have produced a false FAIL. Direct HTTPS is the honest check.
"""
import hashlib
import json
import os
import sys
import urllib.request

REPO_ID = "thundercode/SatQuery"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "repo"))
SRC = r"C:/Users/anish/satquery-ai"


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_of_bytes(b):
    return hashlib.sha256(b).hexdigest()


def opener_no_proxy():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch(url, token, opener):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with opener.open(req, timeout=120) as r:
        return r.read()


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set.")
        return 2

    opener = opener_no_proxy()

    # repo metadata (sizes + listing) over the API
    api = json.loads(
        fetch(f"https://huggingface.co/api/models/{REPO_ID}?blobs=true", token, opener).decode()
    )
    print(f"repo            : {api.get('id')}")
    print(f"private         : {api.get('private')}")
    print(f"sha (HEAD)      : {api.get('sha')}")
    print(f"lastModified    : {api.get('lastModified')}")
    remote = {s["rfilename"]: s for s in (api.get("siblings") or [])}
    print(f"files on Hub    : {len(remote)}")
    print()

    manifest = json.load(open(os.path.join(REPO, "models", "manifest.json"), encoding="utf-8"))

    ok = bad = 0
    print(f"{'STATUS':7} {'hf_path':40} {'remote bytes':>14} {'local bytes':>13}")
    print("-" * 82)
    for a in manifest["artifacts"]:
        hf_path = a["hf_path"]
        local = os.path.join(SRC, a["path"])
        local_bytes = os.path.getsize(local)
        local_hash = sha256_of(local)

        s = remote.get(hf_path)
        if s is None:
            print(f"{'MISSING':7} {hf_path:40} {'-':>14} {local_bytes:>13,}")
            bad += 1
            continue
        remote_bytes = s.get("size")
        try:
            content = fetch(
                f"https://huggingface.co/{REPO_ID}/resolve/main/{hf_path}", token, opener
            )
        except Exception as e:
            print(f"{'ERROR':7} {hf_path:40} {'-':>14} {local_bytes:>13,}   {e}")
            bad += 1
            continue
        remote_hash = sha256_of_bytes(content)

        size_ok = remote_bytes == local_bytes == len(content)
        hash_ok = remote_hash == local_hash
        status = "MATCH" if (size_ok and hash_ok) else "DIFFER"
        ok += status == "MATCH"
        bad += status != "MATCH"
        print(f"{status:7} {hf_path:40} {remote_bytes:>14,} {local_bytes:>13,}")
        if not hash_ok:
            print(f"        local  sha256 {local_hash}")
            print(f"        remote sha256 {remote_hash}")

    print()
    print(f"artifacts verified : {ok}")
    print(f"artifacts failed   : {bad}")
    print()
    for name in ["README.md", "MODEL_CARD.md", "models/manifest.json", "models/checksums.sha256"]:
        print(f"  {'OK ' if name in remote else 'MISS'} {name}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
