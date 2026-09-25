"""Generate RELEASE_MANIFEST.md for the curated public release (Phase 5).

Lists every file in the curated release tree with its size and sha256, computed from disk.
Also emits a machine-readable models/checksums.sha256-style listing for the whole release.
"""
import hashlib
import os
import sys
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "repo"))
OUT = os.path.join(REPO, "RELEASE_MANIFEST.md")

SKIP_NAMES = {"RELEASE_MANIFEST.md"}


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    rows = []
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = sorted(d for d in dirnames if d not in {"__pycache__", ".git"})
        for fn in sorted(filenames):
            if fn in SKIP_NAMES:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, REPO).replace(os.sep, "/")
            rows.append((rel, os.path.getsize(full), sha256_of(full)))

    rows.sort()

    total = sum(r[1] for r in rows)
    lines = []
    lines.append("# Release Manifest")
    lines.append("")
    lines.append("**Repository:** `Anish-lab-blip/SatQuery-AI` (public)")
    lines.append(f"**Generated:** {datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()}")
    lines.append("**Generator:** `release/tools/generate_release_manifest.py` (computed from disk, never typed)")
    lines.append("")
    lines.append(f"**Files:** {len(rows)} · **Total size:** {total:,} bytes ({total/1024:.1f} KiB)")
    lines.append("")
    lines.append("Every file in this release, with its size and sha256. Verify a checkout with:")
    lines.append("")
    lines.append("```bash")
    lines.append("# for each row:  echo '<sha256>  <path>' | sha256sum -c -")
    lines.append("```")
    lines.append("")
    lines.append("| # | Path | Bytes | sha256 |")
    lines.append("|---|---|---|---|")
    for i, (rel, size, h) in enumerate(rows, 1):
        lines.append(f"| {i} | `{rel}` | {size:,} | `{h}` |")
    lines.append("")
    lines.append("## Machine-readable listing")
    lines.append("")
    lines.append("```")
    for rel, size, h in rows:
        lines.append(f"{h}  {rel}")
    lines.append("```")
    lines.append("")
    lines.append("## Contents summary")
    lines.append("")
    lines.append("| Area | What it is |")
    lines.append("|---|---|")
    lines.append("| `README.md` | the repository front page |")
    lines.append("| `MODEL_CARD.md` | model card for the six trained artifacts |")
    lines.append("| `docs/` | the research + engineering documentation set |")
    lines.append("| `docs/architecture/` | the deep architecture reference (multi-part) |")
    lines.append("| `models/manifest.json` | generated manifest of the six trained artifacts |")
    lines.append("| `models/checksums.sha256` | generated checksums for the six trained artifacts |")
    lines.append("| `screenshots/` | real live-run captures |")
    lines.append("")
    lines.append("## What this release deliberately does NOT contain")
    lines.append("")
    lines.append("- **Backbone weights.** They are fetched from the Hugging Face Hub, pinned by revision.")
    lines.append("- **Secrets.** No tokens, keys, or environment files.")
    lines.append("- **The private deployment repositories.** Their sources are not published here.")
    lines.append("- **Datasets.** Acquisition procedures are documented; the data is not redistributed.")
    lines.append("- **A licence file.** None has been selected yet — this is an OPEN item.")

    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"wrote {OUT}")
    print(f"files: {len(rows)}  total: {total:,} bytes")
    for rel, size, h in rows:
        print(f"  {size:>10,}  {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
