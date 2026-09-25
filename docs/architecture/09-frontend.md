# 09 — The Frontend

**Parent:** [Architecture hub](README.md) · **Status tags:** `IMPLEMENTED` · `VERIFIED` ·
`MEASURED` · `NOT RUN` · `SUPPORTED` · `OPEN`

**Sources of truth for this chapter, all read before writing:**

| Source | Lines | What it establishes |
|---|---|---|
| `frontend/assets/js/core.js` | 1029 | the 8 execution events (`SQ.EVENT_NAMES`), the 9-state trace spine, the deterministic policy, the raster synthesiser |
| `frontend/assets/js/live.js` | 392 | the real client: `/api/assets` → `/api/infer`, base-URL resolution, error translation |
| `frontend/assets/js/mission.js` | ~1000 | the Analyze console: `runLive` / `runMock`, `markState`, the trace fill formula, the intent panel |
| `frontend/_headers` | 77 | the Cloudflare cache/security rules and the measured concatenation finding |
| `frontend/*.html` | 11 files | the page set, the shared header `<nav>` with the GitHub and Hugging Face links |
| `scripts/stage_pages.mjs` | ~380 | the reference-driven staging pipeline, its exit codes and its 25 MiB limit |
| `docs/DEPLOYMENT_DECISION.md` | 205 | the hermeticity audit, the film, what was deliberately not created |
| `docs/DEPLOYMENT_TOPOLOGY.md` | 248 | §3.1 the Pages tier; the "EXCEPT `mission.html`" correction |
| `docs/FINAL_DELIVERY_TODO.md` | 366 | §1.4 the status board, §1.7 item 9 the Cloudflare finding, §5 B-08, §6 E-11/E-13/E-14 |
| `DELIVERY_REPORT_2026-09-25.md` | 299 | §3 the live validation and the **harness trap**; §5 the cache-busting measurement |
| session `HANDOFF_NEXT_AGENT.md` | 149 | §4 the hard constraints, including the 308 redirect and the harness rules |

> **The honesty rule this chapter inherits.** `frontend/HANDOFF.md` and the style guide both forbid
> presenting a synthetic value as a measured one. This chapter therefore labels every figure with
> where it came from, and it names the one place where the shipped code does something the
> documentation around it does not describe.

---

## 1. Where the frontend sits

The frontend is the **first** of the four tiers. It is a static site served by Cloudflare Pages.

```
USER
  │  HTTPS
  ▼
Cloudflare Pages (static frontend)     ← frontend/ , staged via scripts/stage_pages.mjs
  │  HTTPS, JSON
  ▼
Render (orchestrator / API gateway)    ← deploy/render/ , render.yaml blueprint
  │  server-to-server
  ▼
GitHub Codespace (FastAPI inference)   ← deploy/codespace/
```
(`docs/DEPLOYMENT_TOPOLOGY.md` §1)

`docs/DEPLOYMENT_TOPOLOGY.md` §3.1 gives the tier's responsibility in one line:

> *"**Responsibility:** serve the static site. No backend, no secrets, no API calls of any kind
> (verified hermetic — see `DEPLOYMENT_DECISION.md` §3)."*

and then the document's own header corrects that claim for one page:

> *"§3.1's "no API calls of any kind (verified hermetic)" holds for every static page EXCEPT
> `mission.html`, which calls the orchestrator."* (`docs/DEPLOYMENT_TOPOLOGY.md` header note)

### 1.1 The hermeticity audit, and its one exception

`docs/DEPLOYMENT_DECISION.md` §3 records the audit that made the static-only deployment viable:

> *"Audited across all of `frontend/` (excluding `.tools/`): **zero** occurrences of `fetch(`,
> `XMLHttpRequest`, `axios`, `EventSource`, `WebSocket`, `/v1/`, `import.meta.env` or `process.env`.
> The only URL-shaped string anywhere is the SVG XML namespace at `frontend/assets/js/core.js:31`,
> which is not a fetch."* (`docs/DEPLOYMENT_DECISION.md` §3)

The namespace string is real and is an XML namespace, not a network call:

```js
svg: function (tag, attrs) {
  var n = document.createElementNS('http://www.w3.org/2000/svg', tag);
```
(`frontend/assets/js/core.js:30-31`)

> *"The staging audit reports **`external network deps: 0 (HERMETIC)`**."*
> (`docs/DEPLOYMENT_DECISION.md` §3)

That audit predates the Analyze console. `_headers` now carries a dated correction of the same claim:

> *"NOTE (2026-09-25): the site is no longer fully static — mission.html loads live.js and POSTs to
> the Render orchestrator (the Analyze console), so a real CSP would also need a connect-src for that
> API origin."* (`frontend/_headers:14-17`)

The page set is therefore **mostly static, with one live page**, and the two statements are not in
conflict: the audit measured what it measured on the tree it measured, and the correction names the
change.

### 1.2 Fonts are self-hosted, which is what makes the site hermetic

> *"`assets/css/system.css` previously opened with a render-blocking `@import` of the Google Fonts CSS
> API. That `@import` is gone, replaced by 10 `@font-face` blocks pointing at 11 woff2 files in
> `assets/fonts/` (518,198 B total, plus `OFL.txt`). This matters more than it looks: a CSS `@import`
> is render-blocking **and** transitively script-blocking — a classic synchronous `<script>` waits on
> pending stylesheets, so a font-host stall could kill the site's JS."*
> (`HANDOFF_NEXT_AGENT.md` §4.1, session workspace)

---

## 2. The eleven pages

The site is **eleven** deployable HTML files: ten content pages plus a real 404 page.

| # | File | `<title>` | Role |
|---|---|---|---|
| 1 | `index.html` | `SATQUERY — Ask the Earth a question.` | the landing page; hosts the self-hosted launch film |
| 2 | `mission.html` | `SATQUERY — Analyze` | **the Analyze console** — the only page that calls the API |
| 3 | `architecture.html` | `SATQUERY — Architecture` | the architecture walk-through; carries its own sample query |
| 4 | `atlas.html` | `SATQUERY — Earth Query Atlas` | the query atlas |
| 5 | `benchmark.html` | `SATQUERY — Benchmark Lab` | per-specialist metrics with honest status labels |
| 6 | `research.html` | `SATQUERY — Research Ledger` | research entries traced to real artifacts/limitations |
| 7 | `journey.html` | `SATQUERY — The Lab` | build history by phase |
| 8 | `run.html` | `SATQUERY — Anatomy of a Run` | a real captured `ResultEnvelope`, rendered |
| 9 | `video.html` | `SATQUERY — Film archive` | the launch film |
| 10 | `references.html` | `SATQUERY — References` | provenance and credits |
| 11 | `404.html` | `SATQUERY — Not found` | a real 404 in the site's design language |

The 404 page was added deliberately and is counted:

> *"`frontend/404.html` | Real 404 page in the existing design language (light/warm/ochre, one accent,
> no rounded cards). Auto-discovered by the staging seed list, so its references are walked — **11
> deployable pages now**."* (`docs/DEPLOYMENT_DECISION.md` §6)

### 2.1 The header, shared by all eleven

Every page carries the same `<nav>` fragment, with the GitHub and Hugging Face links last:

```html
<a class="navlink navlink--ext" href="https://github.com/Anish-lab-blip/SatQuery-AI" target="_blank" rel="noopener" title="SatQuery source repository on GitHub">GitHub</a>
<a class="navlink navlink--ext" href="https://huggingface.co/thundercode/SatQuery" target="_blank" rel="noopener" title="SatQuery model on Hugging Face">Hugging Face</a>
```
(`frontend/mission.html:38-39`)

The **GitHub** target is the only **public** repository, and that is why it is the one linked:

> *"Target: `https://github.com/Anish-lab-blip/SatQuery-AI` — the ONLY **public** repo (Frontend/
> Backend/Inference are private → their links would 404 for the audience)."*
> (`docs/FINAL_DELIVERY_TODO.md` §4, P9-T01)

The **Hugging Face** link is B-01, and its history is worth recording because it shows the blocker
lifecycle:

| Stage | Status |
|---|---|
| earlier | *"`B-01` | HF page + token not created | No HF header link; no HF doc push | Owner → P9-T02 | link to a pinned model page in the interim | **BLOCKED**" |
| 2026-09-25 | *"**CLOSED** 2026-09-25 — owner supplied `https://huggingface.co/thundercode/SatQuery`; link added to all 11 pages"* (`docs/FINAL_DELIVERY_TODO.md` §5) |

The verification is a live DOM query, not a file grep:

> *"`E-13` | B-01 | live DOM query for the HF anchor |
> `a[href*="huggingface.co/thundercode/SatQuery"]` present on the deployed site | VERIFIED"*
> (`docs/FINAL_DELIVERY_TODO.md` §6)

**Measured, in the working tree:** all eleven files carry exactly one occurrence of each link.

| Link | Files carrying it | Occurrences per file |
|---|---|---|
| `https://huggingface.co/thundercode/SatQuery` | 11 / 11 | 1 |
| `https://github.com/Anish-lab-blip/SatQuery-AI` | 11 / 11 | 1 |

(measured by grepping `frontend/*.html`)

The link is styled by a class added with it, so the external-link affordance is part of the same
change:

> *"Files: all 11 `frontend/*.html` header `<nav>`, reusing the existing `.navlink--ext` pattern."*
> (`docs/FINAL_DELIVERY_TODO.md` §4, P9-T02)

### 2.2 The deployed revision

| Repo | Role | Deployed HEAD |
|---|---|---|
| `Anish-lab-blip/SatQuery-Frontend` | Cloudflare Pages (static) | `2d7ae53b482d` |

(`docs/FINAL_DELIVERY_TODO.md` §6 E-10; `DELIVERY_REPORT_2026-09-25.md` §1)

and the three commits that produced it:

| Commit | Subject | Files |
|---|---|---|
| `ff46eba42b18` | correct the lexical-router misroute; same-shape change demo pair; measured calibration curve; HF header link | 17 |
| `d413d3672311` | give the change-demo pair new URLs; drop the ineffective cache carve-out | 8 (+5, −2) |
| `2d7ae53b482d` | sibling `SQ.policy` misroute; state which calibration diagram is plotted | 2 |

(`DELIVERY_REPORT_2026-09-25.md` §1)

> **The deployed repo's root *is* the local `frontend/` directory.** *"`Anish-lab-blip/SatQuery-Frontend`
> — Cloudflare Pages; **repo root == local `frontend/`**"* (session `HANDOFF_NEXT_AGENT.md` §2). There
> is no build step in the deployed repo; the staging script is a local packaging convenience, not a
> CI pipeline.

---

## 3. The staging pipeline — `scripts/stage_pages.mjs`

Cloudflare Pages has **no `.assetsignore`**, so the deployable tree must be curated. That is what the
staging script is for.

> *"Cloudflare Pages constraints that already bit us: per-file limit is **25 MiB** (26,214,400 B);
> Pages has **no `.assetsignore`**, so you must stage a curated directory (hence `stage_pages.mjs`)."*
> (`HANDOFF_NEXT_AGENT.md` §4.3, session workspace)

### 3.1 It is reference-driven, not a hardcoded list

```js
/**
 * stage_pages.mjs — build a Cloudflare-Pages-deployable staging tree for the
 * SatQuery frontend by walking the ACTUAL asset references of the deployable
 * HTML pages (reference-driven closure), rather than a hardcoded file list.
 *
 * Why reference-driven: a font set under assets/fonts/ and a regenerated film
 * encode are both landing. A hardcoded list would silently omit them; this
 * walks each page's src/href/poster, then each CSS @import/url(), then each JS
 * import/export-from/dynamic-import, and copies the transitive closure.
 */
```
(`scripts/stage_pages.mjs:1-10`)

The limit is a constant in the script:

```js
const PAGES_FILE_LIMIT = 26214400; // 25 MiB (Cloudflare Pages hard limit)
const BIG_WARN_BYTES = 10485760;   // 10 MiB (informational)
```
(`scripts/stage_pages.mjs:43-44`)

### 3.2 Options and exit codes

| Flag | Meaning |
|---|---|
| `--frontend=<dir>` | Source frontend dir (default `<repo>/frontend`) |
| `--out=<dir>` | Staging dir (default `<repo>/.deploy/pages`) |
| `--launch-src=<path>` | Rewrite the homepage launch-film `<video src>` **in the staged copy only** |
| `--include=<file>` | Force-add a file no page references (repeatable) |
| `--no-clean` | Do not wipe the staging dir before staging |
| `--help` / `-h` | usage |

| Exit code | Meaning |
|---|---|
| `0` | success (staged + verified) |
| `2` | a staged file exceeds the 25 MiB Cloudflare Pages per-file limit |
| `3` | a reference in the staged tree does not resolve (broken deploy) |
| `1` | other error |

(`scripts/stage_pages.mjs:25-32`)

> *"**SAFETY: never writes to the frontend/ source tree. Copies out only.**"*
> (`scripts/stage_pages.mjs:34`)

The `--include=` files are needed because the walk is strictly reference-driven:

> *"The `--include=` files are force-added because no page references them; the default is strictly
> reference-driven."* (`HANDOFF_NEXT_AGENT.md` §4.1, session workspace)

`_headers` and `robots.txt` are the canonical examples:

> *"`_headers` and `robots.txt` must be force-included because no page references them.
> `provenance.json` and `CREDITS.md` likewise — they are provenance records, not assets."*
> (`docs/DEPLOYMENT_DECISION.md` §7)

### 3.3 A real measured run

```bash
cd C:/Users/anish/satquery-ai

node scripts/stage_pages.mjs \
  --out=.deploy/dist-final \
  --include=_headers \
  --include=robots.txt \
  --include=assets/img/eo/provenance.json \
  --include=assets/img/eo/CREDITS.md

npx wrangler pages deploy "C:/Users/anish/satquery-ai/.deploy/dist-final" --project-name <name>
```
(`docs/DEPLOYMENT_DECISION.md` §7)

**Measured result of that staging run:**

```
files staged          : 60
total bytes           : 39,173,936 (37.36 MiB)
largest file          : assets/video/satquery-launch-50s.mp4  22,710,313 B (21.66 MiB)
25 MiB headroom left  : 3,504,087 B on the largest file
missing refs in staged : 0
external network deps : 0  (HERMETIC)
exit                  : 0
```
(`docs/DEPLOYMENT_DECISION.md` §7)

An earlier run of the same script reports a slightly different total, and the difference is recorded
rather than reconciled:

```
files staged          : 57
total bytes           : 39,163,483 B (37.35 MiB)
largest file          : 22,710,313 B (21.66 MiB)
missing refs          : 0
external network deps : 0
exit                  : 0
```
(`HANDOFF_NEXT_AGENT.md` §4.1, session workspace)

> **Do not treat either number as a constant.** `HANDOFF_NEXT_AGENT.md` §8 item 10 records exactly
> this hazard: *"Hardcoded counts in docs drift. The RUNBOOK's archive size moved 353 → 792 → 364 →
> 366 as the tree changed. Re-measure rather than trusting a recorded number."* 57 and 60 files are
> two measurements of two trees.

### 3.4 A real bug the script had, and its fix

> *"`RE_CSS_IMPORT`'s bare-token alternative captured the prose word `of` out of a stylesheet comment
> (*"This replaced an @import of the Google Fonts CSS API"*) and failed the run with a phantom missing
> reference. Fixed at the root in `scripts/stage_pages.mjs` by adding `stripComments(ext, text)`,
> called at the top of `extractRefs`: CSS `/* */`, HTML `<!-- -->`, and — deliberately — **block
> comments only for JS**, because stripping `//` naively would truncate anything after a `//` inside a
> string such as `'http://www.w3.org/2000/svg'`. A reference inside a comment is never fetched, so
> this is correct, not a suppression."* (`HANDOFF_NEXT_AGENT.md` §4.1, session workspace)

---

## 4. The Analyze console

`mission.html` is the only page that talks to the backend. Its `<title>` is `SATQUERY — Analyze`, and
its structure is a scientific instrument, not a chat window:

| Element | `id` | Role |
|---|---|---|
| query box | `qtext` | *"What changed here?"* by default (`mission.html:51`) |
| run button | `btnRun` | *"Run query"* (`mission.html:53`) |
| observation block | `obsTail` | `none` → `ready` when a file is selected (`mission.html:67`) |
| primary file input | `fileInput` | hidden; flipped by the drop zone (`mission.html:72`) |
| second file input | `fileInputT0` | the T0 frame for pair tasks (`mission.html:120`) |
| intent panel | `intentHost` | the router's reading, as chips (`mission.html:102`) |
| viewer state | `viewerState` | *"Illustrative frame"* → *"Your upload"* → *"Your upload · analysed"* (`mission.html:141`) |
| plate | `plateImg` | the user's own image (`mission.html:146`) |
| evidence SVG layer | `ev` | region overlay (`mission.html:155`) |
| comparison view | `cmpWrap`, `cmpT0`, `cmpT1`, `cmpRange`, `cmpCredit`, `cmpEmpty` | the T0/T1 wipe (`mission.html:171-185`) |
| answer | `answerHost` | the server's string, verbatim (`mission.html:198`) |
| evidence list | `evHost` | the server's `Evidence` records (`mission.html:211`) |
| confidence | `confC` | `—` until a real result arrives (`mission.html:226`) |
| provenance | `pRun`, `pPolicy`, `pProtocol`, `pSchema` | `awaiting backend` until a real result (`mission.html:239`) |
| trace bar | `trace` | the 9-state spine (`mission.html:268`) |
| event drawer | `drawer`, `evlog` | the raw event log (`mission.html:280-286`) |

### 4.1 The design law the console must obey

> *"**Frontend design law:** light/warm/ochre, ONE accent = ochre `#A5662E`. **NO rounded-rectangle
> card aesthetic** — the target is a *scientific instrument*, not an "AI dashboard". No
> glassmorphism, no drop-shadow-as-elevation, no map tiles or map providers. The retired
> graphite/dark tokens are forbidden."* (`HANDOFF_NEXT_AGENT.md` §7, repo copy)

### 4.2 The viewer modes

The plan (§51) lists viewer tabs `Original`, `Evidence`, `Grounding`, `Change`, `Optical`, `SAR`,
`Fusion`. The shipped console implements a **four-mode** viewer, driven by `setMode()`:

```js
function setMode(m) {
  mode = m;
  var showCmp = (m === 'comparison');
  cmpWrap.hidden = !showCmp;
  plate.style.visibility = showCmp ? 'hidden' : 'visible';
  evNote.hidden = !(m === 'evidence' || m === 'masked');

  if (m === 'evidence') { evNoteText.textContent = 'Awaiting backend — no region, mask or change map has been returned.'; viewerState.textContent = 'Evidence · none'; }
  else if (m === 'masked') { evNoteText.textContent = 'Awaiting backend — no availability or change mask has been returned.'; viewerState.textContent = 'Masked · none'; }
  else if (m === 'comparison') {
    if (cmpWrap.dataset.ready === '1') { viewerState.textContent = 'Comparison'; cmpEmpty.hidden = true; }
    else { viewerState.textContent = 'Comparison · needs pair'; cmpEmpty.hidden = false; }
  } else { viewerState.textContent = plateImg.dataset.uploaded ? (plateImg.dataset.analysed ? 'Your upload · analysed' : 'Your upload') : 'Illustrative frame'; }
  ...
}
```
(`frontend/assets/js/mission.js:840-857`)

> **The plan's seven tabs are not the shipped four modes.** The modes are `original` (the default),
> `evidence`, `masked` and `comparison` — visible in the `if/else` chain above. `Grounding`, `Change`,
> `Optical`, `SAR` and `Fusion` are **not** separate viewer modes in the shipped console; region and
> mask output is rendered through the `ev` overlay on the `evidence`/`masked` modes, and the
> optical/SAR pair is rendered as the `comparison` wipe. This is a divergence between the plan's GUI
> sketch and the built page, and it is recorded rather than papered over.

The empty-state wording is itself a disclosure, not a placeholder: *"Awaiting backend — no region,
mask or change map has been returned."*

### 4.3 The confidence panel refuses to invent a number

```js
confC.textContent = '—';
confNote.textContent = 'Calibrated confidence is reported only with a real result. Until then it reads “—” — not a placeholder number.';
```
(`frontend/assets/js/mission.js:452`, `:457`)

---

## 5. REAL versus PREVIEW — the two drivers, one event seam

`mission.js` opens with the distinction as the file's governing design:

```js
/* =============================================================================
   SATQUERY — ANALYZE (mission)
   Drives the page from the production event seam: SQ.run().ingest(type, payload).

   TWO DRIVERS, ONE EVENT SEAM
   ---------------------------
   * LIVE (default when files are chosen): the browser uploads the user's own
     imagery to `POST /api/assets`, receives asset IDs, posts them to
     `POST /api/infer`, and feeds the REAL result into the same eight events.
     Every value on screen then traces to the server's own response.
   * PREVIEW (no files chosen): the deterministic router still runs so the
     instrument is legible, but the specialist/result stages stay honestly empty
     ("awaiting backend") instead of pretending an analysis happened.

   What is NEVER done in either mode: fabricating an answer, a confidence value,
   an evidence record, a run id or a coordinate. If the backend is unreachable
   the page says which step failed and shows the server's own message.
   ============================================================================= */
```
(`frontend/assets/js/mission.js:1-18`)

### 5.1 The switch is the presence of a selected file

```js
function runQuery() {
  var q = qtext.value.trim() || QUERY;
  /* Live whenever the user has actually selected imagery; preview otherwise.
     This is the whole point: the demo demonstrates the intended workflow, and
     a fixture is never silently substituted for a real upload. */
  if (selectedT1) runLive(q);
  else runMock(q);
}
```
(`frontend/assets/js/mission.js:799-806`)

| Driver | Trigger | `liveRun` | Label shown |
|---|---|---|---|
| `runLive` | a file is selected (`selectedT1` truthy) | `true` | `live · N evidence · transport …` |
| `runMock` | no file selected | `false` | `preview — no files selected` |

The label is set at the top of each driver:

```js
traceNow.textContent = 'preview — no files selected';
```
(`frontend/assets/js/mission.js:563`)

```js
traceNow.textContent = 'live · ' + (ev.length) + ' evidence · transport ' + (out.transport || 'direct');
```
(`frontend/assets/js/mission.js:752`)

### 5.2 What the preview does and does not emit — a correction to the common summary

The preview path is often summarised as "it emits no specialist events". **The shipped code emits all
eight event names in both modes.** What the preview withholds is the *content*, not the event:

```js
/* ----------------------------------------------------------- mock driver --
   PREVIEW ONLY — runs when no file has been chosen. Emits the eight
   production events through the seam with EMPTY payloads. RECEIVE/PARSE/PLAN
   carry the honest interpretation; the specialist + result stages carry
   nothing fabricated. */
```
(`frontend/assets/js/mission.js:550-554`)

The preview's specialist events carry a component **name** and no measurement:

```js
specialists.forEach(function (name, i) {
  at(cursor, function () { engine.ingest('SPECIALIST_STARTED', { component: name, index: i + 1, of: specialists.length, stage: 'EXECUTE' }); });
  cursor += 260;
  at(cursor, function () { engine.ingest('SPECIALIST_COMPLETED', { component: name }); });
  cursor += 120;
});
```
(`frontend/assets/js/mission.js:573-578`)

and the result stages carry explicit emptiness:

```js
at(cursor + 200, function () {
  engine.ingest('EVIDENCE_GENERATED', { regions: [], count: 0, note: 'no specialist output connected' });
});
at(cursor + 460, function () {
  engine.ingest('CONFIDENCE_COMPUTED', { degraded: true, degradation_reason: 'No result produced — no calibrated confidence.' });
});
at(cursor + 700, function () {
  engine.ingest('RESULT_ASSEMBLED', { text: null, task: intent.task, evidence_ids: [], confidence: null, provenance: null });
});
```
(`frontend/assets/js/mission.js:580-588`)

The accurate statement is therefore:

> **The preview emits all eight event names, but no specialist measurement, no evidence record, no
> confidence value, no answer, and no run id.** `EVIDENCE_GENERATED` carries `regions: []` and the
> note *"no specialist output connected"*; `CONFIDENCE_COMPUTED` carries `degraded: true` with the
> reason *"No result produced — no calibrated confidence."*; `RESULT_ASSEMBLED` carries
> `text: null`, `confidence: null`, `provenance: null`.

The style guide's rule applies here: the code is authoritative, and a summary that says "no
specialist events" is not what the code does. Recorded.

### 5.3 The state notes distinguish preview from live *visually*

```js
/* Notes for the PREVIEW driver only. When a real result arrives these are
   overwritten by measured facts (component names, model revisions, timings). */
var STATE_NOTE = {
  RECEIVE: 'received', PARSE: 'interpreted (mock router)', VALIDATE: 'validated (mock router)',
  PLAN: 'routed (mock router)', PREPROCESS: 'awaiting backend', EXECUTE: 'awaiting backend',
  AGGREGATE: 'awaiting backend', VERIFY: 'awaiting backend', RESPOND: 'awaiting backend'
};
```
(`frontend/assets/js/mission.js:367-373`)

and the `is-mock` class is the visual marker:

```js
/**
 * Mark a ControllerState reached, optionally with a measured note.
 *
 * `is-mock` is the "driven by the preview router" styling. When a real result
 * is being rendered the note is a measurement, so the mock class is removed
 * instead — otherwise a genuine run would be visually indistinguishable from
 * the preview, which is exactly the confusion this page is built to avoid.
 */
function markState(id, note, isLive) {
  var n = traceNodes[id];
  if (!n) return;
  n.node.classList.toggle('is-mock', !isLive);
  n.tm.textContent = note !== undefined ? note : (STATE_NOTE[id] || '');
  ...
```
(`frontend/assets/js/mission.js:396-408`)

In the **live** driver the notes become measurements, which is the point of the distinction:

```js
/* State notes become MEASUREMENTS, replacing the preview wording. */
markState('PREPROCESS', specialists.join(', '), true);
markState('EXECUTE', (trace.selected_models || []).map(function (m) { return m.name; }).join(', ') || 'executed', true);
markState('AGGREGATE', ev.length + ' evidence', true);
markState('VERIFY', conf ? (conf.method || 'uncalibrated') : 'no confidence', true);
markState('RESPOND', out.transport ? ('via ' + out.transport) : 'responded', true);
```
(`frontend/assets/js/mission.js:745-750`)

### 5.4 The live driver's event sequence is a record, not an animation

```js
/* ----------------------------------------------------------- live driver -
   The real flow. Uploads the user's files, then runs the analysis and feeds
   the server's own result into the same eight events.

   The event sequence is emitted around the network calls rather than faked on
   a timer: QUERY_RECEIVED/UNDERSTOOD/ROUTE_SELECTED are genuinely known before
   the request (they are the client's own reading), SPECIALIST_STARTED is
   emitted when the request is dispatched, and SPECIALIST_COMPLETED through
   RESULT_ASSEMBLED are emitted from the response. So the trace's shape is a
   real record of when the work happened, not a plausible-looking animation. */
```
(`frontend/assets/js/mission.js:591-600`)

The three pre-dispatch events are emitted before `SQ.live.run` is called; the rest are emitted inside
its `.then()`:

```js
engine.ingest('QUERY_RECEIVED', { query: query });
engine.ingest('QUERY_UNDERSTOOD', { intent: intent, dispatched: choice.substituted ? forced : null });
engine.ingest('ROUTE_SELECTED', { intent: intent, specialists: specialists, policy: 'deterministic-rule', policyVersion: 'pc-3.2.1', force_task: forced });

var started = {};
specialists.forEach(function (name) { started[name] = performance.now(); });

SQ.live
  .run(filesToSend, query, { forceTask: forced })
  .then(function (out) { ... });
```
(`frontend/assets/js/mission.js:654-668`)

> **A nuance worth stating.** `SPECIALIST_STARTED`/`COMPLETED` are emitted **from the response**, not
> at dispatch time — the code comment above says "SPECIALIST_STARTED is emitted when the request is
> dispatched", but the implementation emits both inside the `.then()` (lines 678-687), measuring
> `elapsed_ms` from a timestamp taken *before* the request. The timestamps are honest (they bracket
> the real network call); the event *ordering* is response-time. Recorded because the comment and the
> code differ on this one point.

### 5.5 The failure path never fills the gap

```js
.catch(function (err) {
  /* HONEST FAILURE. Say which step failed, and show the server's message.
     The specialist/result stages stay unfilled rather than being given
     invented content, and the trace states are marked as not executed. */
  var stage = (err && err.stage) || 'request';
  var status = (err && err.status) ? ' (HTTP ' + err.status + ')' : '';
  traceNow.textContent = 'failed at ' + stage + status;
  answerHost.innerHTML = '<span class="answer__empty label">No result — the ' + stage + ' step failed</span>';
  ...
  markState('PREPROCESS', 'not executed', false);
  markState('EXECUTE', 'not executed', false);
  markState('AGGREGATE', 'not executed', false);
  markState('VERIFY', 'not executed', false);
  markState('RESPOND', 'not executed', false);
```
(`frontend/assets/js/mission.js:773-791`)

The plate caption also refuses to keep calling the image illustrative once a real run has touched it:

```js
/* The plate caption must not keep calling the image illustrative once a
   real analysis has run on it. The image is still the user's own file,
   so the credit names the run rather than claiming the imagery is a
   SatQuery output -- the picture is input, the FINDINGS are output.
   BOTH halves of the caption are driven: the leading <b> is a literal in
   the markup, and leaving it as "Illustrative" would keep asserting the
   frame is a stand-in while showing the user's own upload. */
plateCreditLead.textContent = 'Your upload';
plateCredit.textContent = 'analysed in run ' + (env.run_id || '—');
```
(`frontend/assets/js/mission.js:754-762`)

### 5.6 The mode is disclosed in the answer's own tail

```js
answerHost.textContent = answerText;
ansTail.textContent = 'live';
```
(`frontend/assets/js/mission.js:714-715`)

and on an empty answer it says so rather than leaving the previous text:

```js
answerHost.innerHTML = '<span class="answer__empty label">The engine returned no answer text for this task</span>';
ansTail.textContent = 'empty';
```
(`frontend/assets/js/mission.js:718-719`)

---

## 6. The eight execution events, and the trace bar

### 6.1 The event names

```js
SQ.EVENT_NAMES = [
  'QUERY_RECEIVED', 'QUERY_UNDERSTOOD', 'ROUTE_SELECTED',
  'SPECIALIST_STARTED', 'SPECIALIST_COMPLETED',
  'EVIDENCE_GENERATED', 'CONFIDENCE_COMPUTED', 'RESULT_ASSEMBLED'
];
```
(`frontend/assets/js/core.js:616-620`)

`core.js`'s own header states the integration seam this defines:

> *"The run engine is deliberately dumb: it renders whatever events it receives. Nothing about the
> visuals depends on the events being synthetic. Swapping the mock driver for a websocket / SSE feed
> of the same event names is the entire integration surface."* (`frontend/assets/js/core.js:8-11`)

and the seam itself:

```js
/**
 * THE INTEGRATION SEAM.
 * Feed real execution events here — same names, same payload shapes — and
 * every visual state in the prototype updates identically.
 */
ingest: function (type, payload) {
```
(`frontend/assets/js/core.js:737-742`)

### 6.2 The event envelope

Every emitted event carries four fields:

```js
function emit(type, payload) {
  var ev = {
    type: type,
    t: performance.now() - t0,
    seq: state.events.length + 1,
    payload: payload || {}
  };
  state.events.push(ev);
  listeners.forEach(function (fn) { try { fn(ev, state); } catch (e) { console.error(e); } });
}
```
(`frontend/assets/js/core.js:709-718`)

| Field | Meaning |
|---|---|
| `type` | one of the eight names |
| `t` | milliseconds since `QUERY_RECEIVED` |
| `seq` | 1-based sequence number within the run |
| `payload` | the event-specific body |

The event drawer prints exactly this:

```js
function logEvent(ev) {
  var line = document.createElement('div');
  line.innerHTML = '<span class="n">' + U.pad(ev.seq, 2) + '  +' + Math.round(ev.t) + 'ms  </span>' +
    '<span class="k">' + ev.type + '</span>  ' +
    '<span>' + JSON.stringify(ev.payload) + '</span>';
  evlog.appendChild(line);
  evlog.scrollTop = evlog.scrollHeight;
}
```
(`frontend/assets/js/mission.js:428-435`)

### 6.3 The nine-state spine

The trace bar is **not** the eight events. It is a **nine-state** `ControllerState` spine, and the
eight events map onto it:

```js
var STATES = ['RECEIVE', 'PARSE', 'VALIDATE', 'PLAN', 'PREPROCESS', 'EXECUTE', 'AGGREGATE', 'VERIFY', 'RESPOND'];
var EVENT_TO_STATE = {
  QUERY_RECEIVED: 'RECEIVE', QUERY_UNDERSTOOD: 'PARSE', ROUTE_SELECTED: 'PLAN',
  SPECIALIST_STARTED: 'PREPROCESS', SPECIALIST_COMPLETED: 'EXECUTE',
  EVIDENCE_GENERATED: 'AGGREGATE', CONFIDENCE_COMPUTED: 'VERIFY', RESULT_ASSEMBLED: 'RESPOND'
};
```
(`frontend/assets/js/mission.js:361-366`)

| Event | State(s) marked |
|---|---|
| `QUERY_RECEIVED` | `RECEIVE` |
| `QUERY_UNDERSTOOD` | `PARSE` **and** `VALIDATE` |
| `ROUTE_SELECTED` | `PLAN` |
| `SPECIALIST_STARTED` | `PREPROCESS` |
| `SPECIALIST_COMPLETED` | `EXECUTE` |
| `EVIDENCE_GENERATED` | `AGGREGATE` |
| `CONFIDENCE_COMPUTED` | `VERIFY` |
| `RESULT_ASSEMBLED` | `RESPOND` |

The subscription is the mechanism:

```js
case 'QUERY_UNDERSTOOD':
  renderIntent(ev.payload.intent, ev.payload.dispatched || null);
  intentTail.textContent = ev.payload.dispatched ? ('routed as ' + ev.payload.dispatched) : 'resolved';
  markState('PARSE', undefined, liveRun); markState('VALIDATE', undefined, liveRun);
  traceNow.textContent = 'interpreting';
  break;
```
(`frontend/assets/js/mission.js:511-516`)

`VALIDATE` has **no event of its own** and is marked together with `PARSE`. The `ControllerState` enum
in `core/schemas.py:79-89` defines all nine; the frontend's `STATES` array is a literal transcription
of it.

### 6.4 The fill formula, and the measured 94.4444 %

The bar's width is a pure function of the furthest state reached:

```js
var idx = STATES.indexOf(id);
if (idx > traceProgress) traceProgress = idx;
STATES.forEach(function (s, k) {
  var node = traceNodes[s].node;
  node.classList.toggle('is-done', k < traceProgress);
  node.classList.toggle('is-active', k === traceProgress);
  node.classList.toggle('is-idle', k > traceProgress);
});
if (traceFill) {
  traceFill.style.width = (((traceProgress + 0.5) / STATES.length) * 100) + '%';
}
```
(`frontend/assets/js/mission.js:414-424`)

The comment states the intent:

> *"Advance the trace to the furthest state reached. This is driven by the same events that carry the
> real result, so the bar and the node states move only when the run actually reaches a stage — never
> on a timer. The fill spans from the left edge to the centre of the current node."*
> (`frontend/assets/js/mission.js:410-413`)

**The arithmetic, worked:**

```
STATES.length          = 9
final traceProgress    = 8        (RESPOND is the last index, and it is reached)
fill = ((8 + 0.5) / 9) * 100
     = (8.5 / 9) * 100
     = 94.4444444…%
```

So **94.4444 % is the fully-complete trace bar**, not a partial one: it is 8.5/9, and the missing
5.5556 % is the half-node at the right edge that the "centre of the current node" rule deliberately
leaves unfilled.

**Measured, live, 2026-09-25:**

> *"| Execution trace (progress bar) | **VERIFIED** | `.trace__fill` width is set from real event
> count (measured 94.4444% live, 2026-09-25) |"* (`docs/FINAL_DELIVERY_TODO.md` §1.4)

and confirmed across all three live passes:

> *"Common to all twenty-four: a real `run_*` id, `mock_nodes = 0`, live trace bar at 94.4444%, the HF
> link present in the DOM, and every `/api/*` call addressed to
> `<backend-host>` (`capabilities` → `assets` → `infer`; two `assets` calls for
> the pair tasks)."* (`DELIVERY_REPORT_2026-09-25.md` §3)

The implementation note records that the bar was *fixed* to reach this:

> *"`markState` now sets `traceFill.style.width` from the furthest state reached and toggles
> `is-done`/`is-active`/`is-idle`; `resetUI` resets both. CSS classes already existed
> (`system.css:654-660`)."* (`docs/FINAL_DELIVERY_TODO.md` §4, P5-T02)

and the reset clears it:

```js
traceProgress = -1;
if (traceFill) traceFill.style.width = '0';
```
(`frontend/assets/js/mission.js:440-441`)

> **94.4444 % is an artifact metric, not a system-level claim.** It measures one CSS width in one
> page. It says nothing about the pipeline's own progress reporting — the server returns a completed
> `ResultEnvelope` with no streaming, so the bar reflects the *client's* event timeline, which is
> itself reconstructed from a single request/response pair.

### 6.5 The bar is not a determinate progress bar, and the docs say so

> *"A determinate-looking progress bar would lie. Use an indeterminate state with a 'this can take up
> to a minute' hint."* (`docs/FRONTEND_INTEGRATION.md` §6)

The trace bar is a **stage** indicator — which states have been reached — not a percentage of elapsed
time. The 94.4444 % figure is the completed state of that stage indicator, and reading it as "94 % of
the work is done" would be a misreading the code does not invite.

### 6.6 The deterministic policy the preview uses

The preview's routing is a **rule list**, not the learned router, and the page says so in the intent
panel (`source`, `policyVersion: 'pc-3.2.1'`, `basis: 'rule match, no learned router'`).

```js
/* --- deterministic policy: mirrors the intended controller, rules only --- */
SQ.policy = function (query) {
  var q = (query || '').toLowerCase();
  var rules = [];
  function hit(re, name) { var m = re.test(q); rules.push({ rule: name, fired: m }); return m; }
  ...
```
(`frontend/assets/js/core.js:622-626`)

The B-08 defect lived here and in `mission.js`, and its fix is worth recording because it is the
clearest example of a *lexical* router failing in a way that produced a wrong answer rather than an
error:

> *"**Sibling defect**: `core.js` `SQ.policy` (the architecture page's mock router) still carried
> `built` in its change regex, so that page's **own shipped sample** — *"Where is the built-up
> area?"* — fired `intent.change` on `built` and `intent.quantify` on `area`, and was answered as
> `CHANGE_VQA` with a change-detector specialist. Same defect as `mission.js`, on a second surface.
> `where` is now evaluated first, `built` removed, and `new` counts only outside a `where` question;
> a small `record()` helper keeps the rule list's display order. Regression tests drive the
> **shipped** `SQ.policy` — **4 red before the fix, 6 green after**."*
> (`DELIVERY_REPORT_2026-09-25.md` §1.3)

The fixed rule order is visible in the code, with the reason in a comment:

```js
/* `where` is evaluated FIRST because the change rule below depends on it: a
   word that reads as "change" only OUTSIDE a location question must not turn
   a `where` question into a change request. */
var where = /where|locate|position|which part|bound|outline|coordinate/.test(q);

var sar = hit(/\bsar\b|radar|backscatter|sentinel-1|insar/, 'sensor.sar');
/* `built` was removed and `new` counts only outside a `where` question.
   "Where is the built-up area?" -- the architecture page's OWN sample --
   previously fired intent.change on "built", then intent.quantify on "area",
   and was answered as CHANGE_VQA with a CHANGE_DETECTOR specialist: a
   location question routed to a change question. */
var changeStem = /chang|differ|expand|grow|encroach|lost|removed/.test(q);
var newAsChange = /\bnew\b/.test(q) && !where;
var change = record('intent.change', changeStem || newAsChange);
```
(`frontend/assets/js/core.js:631-645`)

**B-08 is `CLOSED`**, with two documented residuals:

> *"Residuals: "What is the new runway?" still reads `change` (non-`where` + `new`); "How much
> built-up area was added?" now reads `vqa` (under-trigger) — both documented."*
> (`docs/FINAL_DELIVERY_TODO.md` §5, B-08)

### 6.7 The answer bank is prototype text, and it is labelled

`SQ.ANSWER_BANK` holds per-task sentence templates for the **preview** path only:

```js
SQ.ANSWER_BANK = {
  CHANGE_ANALYSIS: 'Significant change detected. {n} coherent regions totalling {area}; dominant transition is {dominant}. Registration residual {reg} px — the pair is usable for pixel comparison.',
  ...
};
```
(`frontend/assets/js/core.js:677-684`)

They are filled from `SQ.scene()`, whose own comment is the disclosure:

```js
/* SYNTHETIC DEMO CONSTANTS — the geo / temporal / metric fields below are
   illustrative placeholders, not real observations. They exist only so the
   prototype renders a populated instrument; the platform string above
   already marks the plate as PROTOTYPE SYNTHETIC. Any page that displays
   these values MUST disclose that they are synthetic (see the `synthetic`
   flag below) and MUST NOT present them as measured satellite data.
   core.js is shared across pages and is intentionally NOT removed here —
   only labelled. If a live backend ever supplies a real scene, it should
   set synthetic:false and override these fields. */
return {
  seed: seed,
  biome: biome,
  regions: regions,
  synthetic: true,               // every field below is a demo placeholder
  gsd: 10,                       // metres per pixel (placeholder)
  aoi: { lat: 31.204, lon: 72.816 },          // placeholder coordinate
  dates: { t0: '2024-03-14', t1: '2025-03-19' }, // placeholder epochs
  sensor: 'OPTICAL / MSI',
  platform: 'SENTINEL-2 · L2A (PROTOTYPE SYNTHETIC)',
  registrationRMSE: 0.42         // placeholder residual
};
```
(`frontend/assets/js/core.js:268-288`)

> **`SQ.ANSWER_BANK` and `SQ.scene()` are preview-only.** The live driver never reads them: it renders
> `result.answer` verbatim (`mission.js:712-714`). The `synthetic: true` flag and the `platform`
> string exist so a reader of the *preview* cannot mistake it for a measurement. This is the design
> the style guide's "never upgrade a status" rule requires, implemented in the data itself.

---

## 7. The live client — `frontend/assets/js/live.js`

### 7.1 The endpoints it calls

```js
/*: The orchestrator's proxied routes (deploy/render/main.py). These are NOT
   the Space's own `/v1/*` routes -- the browser never talks to the Space
   directly; the orchestrator is the only public door. */
SQ.ENDPOINTS = {
  assets: '/assets',
  infer: '/infer',
  capabilities: '/capabilities',
  health: '/health'
};
```
(`frontend/assets/js/live.js:52-60`)

### 7.2 Base-URL resolution, in three ordered steps

```js
/**
 * The orchestrator base URL, with no trailing slash.
 *
 * Resolution order is documented in the file header. Returning `/api` rather
 * than '' keeps the failure mode legible: a misconfigured deployment asks the
 * Pages host for `/api/infer` and gets a clean 404, instead of the page's
 * own index.html being fetched as JSON and producing a confusing parse error.
 */
SQ.live.baseUrl = function () {
  var injected = window.SATQUERY_API_BASE;
  if (injected) return _normalizeBase(String(injected));

  var meta = document.querySelector('meta[name="satquery-api-base"]');
  if (meta && meta.content) return _normalizeBase(String(meta.content));

  return '/api';
};
```
(`frontend/assets/js/live.js:91-107`)

| Order | Source | Why |
|---|---|---|
| 1 | `window.SATQUERY_API_BASE` | an inline config so a deployment points at its own backend without rebuilding the JS |
| 2 | `<meta name="satquery-api-base">` | the same idea, declarative |
| 3 | `/api` on the current origin | correct for a same-origin deployment and for the local dev proxy |

The normaliser exists because an absolute base naturally omits `/api`:

```js
/**
 * A configured base, with `/api` guaranteed for absolute origins.
 * ...
 * someone configuring an absolute URL naturally writes the
 * ORIGIN -- `https://host` -- and then `url()` produced `https://host/assets`
 * instead of `https://host/api/assets`. Every call 404s, and it is a silent
 * failure: the page reports a network error rather than a misconfiguration.
 */
function _normalizeBase(raw) {
  var base = String(raw).replace(/\/+$/, '');
  if (base.indexOf('://') === -1) return base;          // relative: as written
  var after = base.slice(base.indexOf('://') + 3);
  var slash = after.indexOf('/');
  var path = slash === -1 ? '' : after.slice(slash);
  if (path === '' || path === '/') return base + '/api';
  return base;
}
```
(`frontend/assets/js/live.js:109-131`)

> *"Rule 3 is why development needs no secret: a dev server that proxies `/api` to Render lets the
> browser talk to `http://localhost:8080/api/...` and the CORS allowlist is then a non-issue. Direct
> cross-origin calls also work, and that is what the localhost CORS entries in
> `deploy/render/main.py` exist for."* (`frontend/assets/js/live.js:40-43`)

### 7.3 The content-type map, derived from the extension

```js
/*: Extensions the Space's store accepts, mirroring the gateway content-type
   allowlist (gateway/policy.py `allowed_content_types`). The browser sets the
   Content-Type header from this map; a wrong type is a 422 from the store, so
   guessing it from the extension is more reliable than trusting the File's
   own `.type`, which browsers leave empty for GeoTIFF. */
SQ.CONTENT_TYPES = {
  tif: 'image/tiff',
  tiff: 'image/tiff',
  png: 'image/png',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg'
};
```
(`frontend/assets/js/live.js:62-73`)

Note the client's map has **four** types and omits `image/geotiff` and `application/octet-stream`,
which the server's allowlist of five includes. A `.geotiff` file therefore has no client-side mapping
and is refused by `uploadAsset` before any request:

```js
var contentType = SQ.contentTypeFor(file);
if (!contentType) {
  return Promise.reject(
    LiveError(
      'upload',
      'Unsupported file type: ' + (file && file.name ? file.name : '(unnamed)') +
        '. Use GeoTIFF, TIFF, PNG or JPEG.'
    )
  );
}
```
(`frontend/assets/js/live.js:204-213`)

> **This is a real client/server asymmetry.** The message says "Use GeoTIFF, TIFF, PNG or JPEG" while
> the map has no `geotiff` extension key, so a file named `scene.geotiff` is refused with a message
> that names its own format. The server would accept it as `image/geotiff`. Recorded as a defect in
> the client, not smoothed over.

### 7.4 Uploads are sequential, on purpose

```js
/**
 * Upload several Files, sequentially, preserving order.
 *
 * Sequential rather than parallel, and that is a considered choice: the
 * Codespace runs `cache_max_models: 1` and puts v1 execution in a single
 * process with sequential plans (gateway/assets.py). Firing five uploads at
 * once gains nothing and makes a partial failure harder to reason about --
 * the caller learns exactly which file failed, by index.
 */
```
(`frontend/assets/js/live.js:245-254`)

This is the client's implementation of the contract's *"Serialize requests"* obligation
(`docs/API_CONTRACT.md` §6, item 3).

### 7.5 The analysis request body is minimal, because `extra="forbid"`

```js
var body = { assets: ids, query: String(query || '') };
if (opts.forceTask) body.force_task = opts.forceTask;
```
(`frontend/assets/js/live.js:301-302`)

> *"`extra="forbid"` is why this function sends nothing else: an extra key is a 422, not an ignored
> field. `force_task` is omitted rather than sent as null, because both are accepted but omitting it
> keeps the payload minimal and lets the server's own router decide."* (`frontend/assets/js/live.js:287-291`)

### 7.6 The response is asserted at the boundary

```js
return resp.json().catch(function () { return null; }).then(function (parsed) {
  if (!resp.ok) throw describeFailure('infer', resp.status, parsed);
  if (!parsed || !parsed.result) {
    throw LiveError('infer', 'The service returned no result.', {
      status: resp.status,
      detail: JSON.stringify(parsed).slice(0, 400)
    });
  }
  return { envelope: parsed, state: state, transport: transport };
});
```
(`frontend/assets/js/live.js:318-327`)

and on the upload path:

```js
/* Assert the shape at the boundary. A 200 whose body lacks asset_id
   would otherwise travel into `/api/infer` as `undefined` and fail
   there, naming the wrong cause. */
if (!body || typeof body.asset_id !== 'string' || !body.asset_id) {
  throw LiveError('upload', 'Upload succeeded but returned no asset id.', { ... });
}
```
(`frontend/assets/js/live.js:224-231`)

### 7.7 The error object carries the contract's classification

```js
function LiveError(stage, message, opts) {
  opts = opts || {};
  var err = new Error(message);
  err.name = 'SQ.LiveError';
  err.stage = stage;
  err.status = opts.status || 0;
  err.code = opts.code || '';
  err.detail = opts.detail || '';
  err.recoverable = !!opts.recoverable;
  return err;
}
```
(`frontend/assets/js/live.js:152-162`)

> *"`stage` names the step ('upload' | 'infer'), `status` is the HTTP status if a response was
> received, and `code` is the contract's error code when the server supplied the v1 envelope. The
> server's own message is preserved rather than replaced -- a generic "something went wrong" would
> hide the difference between "your file is too large" and "the engine is waking"."*
> (`frontend/assets/js/live.js:142-150`)

### 7.8 The capabilities call exists so the page can refuse to promise

```js
/**
 * GET /api/capabilities, for the UI to show what the engine can actually do.
 *
 * Not part of the analysis flow; it exists so the page can refuse to promise
 * a task the deployment cannot serve, rather than failing after an upload.
 */
```
(`frontend/assets/js/live.js:368-373`)

`mission.js` consumes it at start-up:

```js
engine = SQ.run({ query: QUERY });
engine.on(onEvent);
resetUI();
runMock(QUERY);
loadCapabilities();
```
(`frontend/assets/js/mission.js:971-975`)

### 7.9 The pair-aware dispatch — the client honours `requires_pair`

> *"`/api/capabilities` declares `requires_pair` and `max_assets` per task, and the page chooses the
> task with the asset count in mind."* (`frontend/assets/js/mission.js:134-135`)

```js
/* Send ONLY the assets the dispatched task requires. A single-image task
   (vqa / grounding / caption) must NOT receive the optional T0 frame: the
   backend rejects a two-asset payload for a one-asset task with
   `invalid_request`. Temporal tasks (change / change_vqa) need T0+T1, and
   optical_sar needs the optical+SAR pair (T1 + the second modality in T0). */
var filesToSend = assetsForTask(forced, selectedT1, selectedT0);
```
(`frontend/assets/js/mission.js:624-629`)

and the substitution is disclosed before the request, not after:

```js
/* Say the substitution where the user is looking, before the request, so
   the result is not surprising. It is a fact about the request, not an
   error: one image genuinely cannot support change detection. */
if (choice.substituted) {
  obsNote.innerHTML = 'Analysing as <span class="mono">' + forced +
    '</span> — ' + choice.reason + '. Add a T0 frame to run <span class="mono">' +
    choice.wanted + '</span>.';
}
```
(`frontend/assets/js/mission.js:645-652`)

This was a **deployed defect** (P1 in the status board) and its fix is recorded:

> *"Acceptance: deployed `mission.js` contains `assetsForTask`; vqa-with-pair returns real result
> (only T1 uploaded)."* (`docs/FINAL_DELIVERY_TODO.md` §4, P4-T01)

### 7.10 Optical-SAR gets an early warning, because its pair is two modalities

```js
/* Optical-SAR is the one task whose pair is two MODALITIES, not two times.
   Warn early (before the upload) when the second file looks like a plain
   photo rather than a radar product, so the round-trip does not fail opaquely. */
if (forced === 'optical_sar' && pairNote) {
  var sarCheck = validateOpticalSar(selectedT1, selectedT0);
  pairNote.innerHTML = sarCheck.level !== 'ok' ? sarCheck.message : pairNoteDefault;
}
```
(`frontend/assets/js/mission.js:631-637`)

The contract's rule the validator implements:

> *"**Optical+SAR contract:** modality inferred from band count — `{1,2}` ⇒ SAR, `{3,4,8,11,12,13}` ⇒
> optical; both GeoTIFF, same W×H, uint8/uint16, rasterio-readable."*
> (session `HANDOFF_NEXT_AGENT.md` §4)

---

## 8. Cloudflare traps

Three measured behaviours of the Cloudflare Pages tier cost real time on this project. All three are
recorded because they are non-obvious and each one produced a wrong assumption.

### 8.1 `_headers` rules CONCATENATE — they do not override

This is the finding, and the file's own earlier comment was **false**:

```text
# Format: a path pattern, then indented Header: value lines. `*` matches any
# number of characters. A request that matches several rules inherits ALL of
# them, and a header set by more than one rule is JOINED with a comma in file
# order -- it is NOT overridden. Verified live 2026-09-25: a narrower
# Cache-Control rule did not replace the broader one, it appended to it. To
# remove a header contributed by a broader rule, detach it with a
# "! Header-Name" line (Cloudflare Pages supports `!` detach).
```
(`frontend/_headers:3-9`)

**The measurement.** A specific `max-age=0` rule placed under the broad `/assets/img/*`
`max-age=604800` rule produced this live response:

```
Cache-Control: public, max-age=604800, public, max-age=0, must-revalidate
```
(`DELIVERY_REPORT_2026-09-25.md` §5; `frontend/_headers:70`)

**The consequence, and why it is worse than a normalisation failure:** Chromium takes the **first**
`max-age` it finds, so the broad week-long value still won:

> *"Chromium honours the **first** `max-age`, so the returning browser kept the stale image and the
> carve-out was ineffective."* (`DELIVERY_REPORT_2026-09-25.md` §5)

The `_headers` file records the same conclusion and forbids reintroducing the pattern:

```text
# NOTE (2026-09-25): an earlier revision of this file tried to carve the
# delta-growth change-demo pair out of the week-long rule above with two literal
# path blocks carrying max-age=0. It did NOT work. Cloudflare does not override a
# header when a second rule sets it -- it JOINS the values in file order, and the
# live response was "public, max-age=604800, public, max-age=0, must-revalidate".
# Chromium takes the FIRST max-age it finds, so the broad week-long value still
# won and a returning browser kept the stale image. Do not reintroduce a
# Cache-Control carve-out here: any rule broad enough to matter also matches
# /assets/img/*, so the broad value is always present. The pair was instead given
# NEW URLs (delta-growth-t0-1975-720.jpg / delta-growth-t1-2025-720.jpg), which
# is the only cache-busting that does not depend on _headers semantics.
```
(`frontend/_headers:66-76`)

The status board records the corrected comment as a *documentation* fix, which is what it was:

> *"| Cache-busting | **VERIFIED (with a caveat)** | `_headers` revalidates JS/CSS; the EO pair was
> instead given NEW URLs because Cloudflare **concatenates** matching `_headers` rules — see §1.7
> item 9 |"* (`docs/FINAL_DELIVERY_TODO.md` §1.4)

and §1.7 item 9:

> *"**Cloudflare `_headers` CONCATENATES matching rules** instead of overriding them. A specific rule
> under a broad `/assets/img/*` rule produces `Cache-Control: public, max-age=604800, …, max-age=0,
> must-revalidate`, and Chromium honours the **first** `max-age` — so a per-path override cannot
> un-cache a long-lived asset. Measured live 2026-09-25. The cache-busting fix therefore **renames**
> the asset to a new URL rather than adding a `_headers` rule."*
> (`docs/FINAL_DELIVERY_TODO.md` §1.7 item 9)

**A consequence that persists:** the deleted old URLs still answer from the edge cache.

> *"Consequence: the old URLs still answer `200` from Cloudflare's **edge cache** (`CF-Cache-Status:
> HIT`) although the files are deleted; a cache-busted request returns `404`. Nothing references
> them."* (`DELIVERY_REPORT_2026-09-25.md` §5)

### 8.2 The `!` detach escape hatch

Cloudflare Pages supports `! Header-Name` to detach a header contributed by a broader rule. The
`_headers` file records it as the **correct** way to remove an inherited header
(`frontend/_headers:8-9`) — but the project chose URL renaming instead for the EO pair, because
*"any rule broad enough to matter also matches `/assets/img/*`, so the broad value is always
present"* and a detach would have removed `Cache-Control` for *every* image rather than for one.

### 8.3 The `308` redirect: `X.html` → `/X`

> *"**Cloudflare Pages 308-redirects `X.html` → `/X`.** Drive `https://satquery.pages.dev/mission`."*
> (session `HANDOFF_NEXT_AGENT.md` §4)

| Request | Response |
|---|---|
| `GET /mission.html` | `308` → `Location: /mission` |
| `GET /mission` | `200`, the page |

**Why it matters for a harness.** A headed-browser driver that navigates to
`https://satquery.pages.dev/mission.html` is redirected, and any assertion written against the
*pre-redirect* URL — or against a `location.pathname` that still ends in `.html` — sees a different
document URL than it expected. The rule the project adopted is to drive the **extensionless** path.

**Why it does not matter for the site itself.** `docs/DEPLOYMENT_DECISION.md` §6 records that no
`_redirects` file was created, and the reason:

> *"**`_redirects`** — every link in the site is already a literal `.html` path; there are no pretty
> URLs to map."*

So the site's own internal links are `.html` and Cloudflare's `308` is a *host-level* behaviour that
the site does not depend on. The two facts are consistent: the site never needs the redirect, and a
harness that types the URL directly must account for it.

### 8.4 What was deliberately NOT created

`docs/DEPLOYMENT_DECISION.md` §6 lists the files that were considered and refused, with reasons:

| Not created | Reason |
|---|---|
| `_redirects` | every link is already a literal `.html` path; there are no pretty URLs to map |
| `sitemap.xml` | *"needs a canonical production domain. Inventing one would publish a URL that does not resolve, so it is omitted until the Pages domain is fixed."* |
| `wrangler.toml` | optional for Pages; the deploy command carries the project name |
| a **Content-Security-Policy** | *"`atlas.html` carries one inline `style=""` attribute, so a strict CSP would need `'unsafe-inline'` anyway. A CSP permitting unsafe-inline is security theatre; add a real one after that attribute is moved into `pages.css`."* |

> **Note the correction.** `sitemap.xml` *does* exist in the tree (`frontend/sitemap.xml`, 741 B) and
> `robots.txt` carries a `Sitemap:` line, so the "needs a canonical domain" blocker was resolved
> after that decision record was written. The decision record is retained as the historical record;
> the tree is the current state.

---

## 9. Cache-busting

### 9.1 The rule that matters: JS and CSS must revalidate

```text
# CSS and JS are NOT content-hashed. They MUST revalidate on every request, or a
# deploy is masked by a cached asset for up to the max-age window — observed on
# 2026-09-25 when a returning browser served the pre-fix mission.js and kept
# hitting the old invalid_request. max-age=0 + must-revalidate makes the browser
# re-fetch (and Cloudflare re-validate) on every load, so a deploy is picked up
# immediately, exactly like the HTML above.
/assets/css/*
  Cache-Control: public, max-age=0, must-revalidate

/assets/js/*
  Cache-Control: public, max-age=0, must-revalidate
```
(`frontend/_headers:50-60`)

**The incident that produced the rule is named in the comment:** a returning browser served the
**pre-fix `mission.js`** and *"kept hitting the old `invalid_request`"*. The cache was masking a
correct deploy — the same failure class as the phantom defect chase that
session `HANDOFF_NEXT_AGENT.md` §4 warns about:

> *"**Stale browser cache can mask a correct deploy.** Verify server-side (GitHub API sha256) *and*
> client-side (CDP `Network.clearBrowserCache`), or you will chase a phantom."*

### 9.2 The full `_headers` policy

| Path pattern | `Cache-Control` | Why |
|---|---|---|
| `/*` | *(none set)* | the baseline block sets only security headers |
| `/` | `public, max-age=0, must-revalidate` | HTML revalidates every time |
| `/*.html` | `public, max-age=0, must-revalidate` | *"so a deploy is picked up immediately rather than being masked by a cached page that still points at yesterday's CSS"* |
| `/assets/video/*` | `public, max-age=604800` | the launch film, 22,710,313 B — *"the single largest asset on the site and the one worth not re-downloading"* |
| `/assets/fonts/*` | `public, max-age=31536000` | *"Stable, versioned by presence rather than by filename, so a long max-age is appropriate."* |
| `/assets/css/*` | `public, max-age=0, must-revalidate` | not content-hashed |
| `/assets/js/*` | `public, max-age=0, must-revalidate` | not content-hashed |
| `/assets/img/*` | `public, max-age=604800` | *"Copernicus / ESA / NASA reference imagery. Stable."* |

(`frontend/_headers:21-64`)

The baseline security headers, which apply to every path:

```text
/*
  X-Content-Type-Options: nosniff
  Referrer-Policy: strict-origin-when-cross-origin
  X-Frame-Options: DENY
  Cross-Origin-Opener-Policy: same-origin
```
(`frontend/_headers:21-25`)

> **No CSP, and the file says why.** *"No Content-Security-Policy is set, deliberately. NOTE
> (2026-09-25): the site is no longer fully static — mission.html loads live.js and POSTs to the
> Render orchestrator (the Analyze console), so a real CSP would also need a connect-src for that API
> origin. atlas.html additionally carries one inline `style=""` attribute, so any CSP would have to
> allow `'unsafe-inline'` anyway."* (`frontend/_headers:14-19`)

### 9.3 The status, with its caveat

> *"| Cache-busting | **VERIFIED (with a caveat)** | `_headers` revalidates JS/CSS; the EO pair was
> instead given NEW URLs because Cloudflare **concatenates** matching `_headers` rules"*
> (`docs/FINAL_DELIVERY_TODO.md` §1.4)

and the phase item is honest about which half was verified when:

> *"**P4-T02** — Cache-busting for JS/CSS | Status: **COMPLETE** (live confirmation pending P10-T01
> re-deploy) | Acceptance: returning users get fresh JS on next load. | Evidence: file edited
> (2026-09-25). Post-deploy `curl -I` to confirm header."* (`docs/FINAL_DELIVERY_TODO.md` §4)

> **`UNKNOWN — not established from the available evidence`:** the post-deploy `curl -I` output
> confirming the live `Cache-Control` on `/assets/js/*`. The file was edited and the phase marked
> complete; the recorded evidence is the file edit, not a captured response header.

---

## 10. The harness lesson — a headed-browser driver that records false passes

This is the most valuable operational finding in this chapter, because it is a **false pass**, not a
false failure.

### 10.1 The failure, as it was observed

> *"When I re-ran the suite to cover the final commit, the first case came back `run_id=0002`,
> `mock_nodes=9`, `answer="No answer yet"`, and only the `capabilities` call — i.e. the **mock** path.
> Diagnosis: the harness drove the query box with `fill_input()`, which types using **real CDP key
> events**, and Chrome **drops synthesized key events when the browser window does not hold OS
> focus**. Measured directly: with Chrome backgrounded, `press_key("Z")` left `#qtext.value`
> unchanged, while `type_text("Q")` (CDP `Input.insertText`, not focus-gated) inserted fine.
> `fill_input` has **no assertion**, so the harness happily clicked Run with the page's **default**
> query still in the box."* (`DELIVERY_REPORT_2026-09-25.md` §3)

**The shape of the false pass:** the query box kept the page's default (`What changed here?`), the
harness clicked Run anyway, the page produced a *result* — and that result was recorded as the
verdict for a case whose query was never entered. The harness had no way to tell the difference
between "the query was entered and the run used it" and "the query was never entered".

### 10.2 Why the earlier 8/8 run was *not* infected — the three discriminators

The report does not simply re-run and hope. It checks whether the earlier result was contaminated,
using evidence the harness recorded:

> *"I then checked whether the earlier 8/8 run was infected by the same silent failure. It was not:
>
> * its recorded intents are **query-specific** — A1 reads `taskvqa…temporalnone`, whereas the default
>   query *"What changed here?"* would read `taskchange…temporalrequired` (exactly what the failed run
>   showed);
> * its answers **embed the query text** — e.g. `[grounding] Located 6 candidate region(s) for 'Where
>   are the built-up areas in this image?'`;
> * A6 required two files (`optical 4/12 + SAR 2/2` channels), which only the uploaded pair supplies.
>
> So the 8/8 result is a valid measurement."* (`DELIVERY_REPORT_2026-09-25.md` §3)

The three discriminators generalise:

| Discriminator | What it proves |
|---|---|
| the recorded **intent** is query-specific | the query reached the router |
| the **answer** embeds the query text | the server received the intended query |
| a case **requires an artefact** only the setup supplies | the setup really happened |

### 10.3 The fix: deterministic query entry plus pre-dispatch assertions

> *"The harness has since been rebuilt (`run_all_postfix2.harness`) to set the query deterministically
> and to **assert the form state before clicking Run**, recording per case: `q_ok` (the box really
> held the query), `obs_ok` (`#obsTail == 'ready'` and one file on `#fileInput`), `t0_ok` (both frames
> for pair tasks), `no_mock_nodes`, and a computed `verdict`. A silent no-op can no longer be recorded
> as a pass."* (`DELIVERY_REPORT_2026-09-25.md` §3)

The three pre-dispatch assertions, and the rule they implement:

> *"**Do NOT use `fill_input()` or `press_key()` to enter the query.** They type with real CDP key
> events, which Chrome **silently drops when the browser window does not hold OS focus** — the box
> keeps its default text and the run silently exercises the wrong query. Use `js()` to set
> `#qtext.value` (plus `input`/`change` events) and/or `type_text()` (CDP `Input.insertText`, not
> focus-gated). **Always assert the form state before clicking Run** — `q_ok` (box holds the query),
> `obs_ok` (`#obsTail == 'ready'`), `t0_ok` (both frames for pair tasks) — or a no-op will be recorded
> as a pass. `upload_file()` is fine and flips `#obsTail` to `ready`."*
> (session `HANDOFF_NEXT_AGENT.md` §5.2)

| Assertion | Checks | Failure it prevents |
|---|---|---|
| `q_ok` | `#qtext.value` holds the intended query | the silent-drop false pass |
| `obs_ok` | `#obsTail == 'ready'` **and** one file on `#fileInput` | an upload that did not land |
| `t0_ok` | both frames present, for pair tasks | a pair task run on one asset |
| `no_mock_nodes` | `mock_nodes == 0` | the preview path being recorded as live |

`obs_ok`'s second half is a real DOM fact, because the page sets that tail from the upload:

`#obsTail` reads `none` in the markup (`mission.html:67`) and the live driver flips it to `ready`.

### 10.4 Two more harness bugs — both false *failures*

> *"Two further harness bugs surfaced while re-running — both produced **false failures**, never false
> passes, but they are easy to repeat:
>
> 1. **The answer tag is not universal.** The server prefixes the answer with `[task]` only for the
>    region tasks (`grounding`, `change`, `change_vqa`, `optical_sar`). vqa answers are bare
>    (`Grassland`) and caption answers are prose, so a tag-only discriminator wrongly fails them.
>    Fix: the **dispatched** task is `answer_tag` when present, else the intent panel's reading.
> 2. **The intent panel renders a concatenated string** — `task<name>modality<…>temporal<…>`.
>    Matching `task([a-z_]+)` greedily swallows the whole string; it must be `task([a-z_]+?)modality`."*
>    (`DELIVERY_REPORT_2026-09-25.md` §3)

The concatenation is a real property of the intent panel, which renders chips without separators:

```js
function renderIntent(intent, dispatched) {
  intentHost.innerHTML = '';
  var rows = [
    ['task', dispatched || intent.task], ['modality', intent.modality], ['temporal', intent.temporal],
    ['spatial', intent.spatial_output], ['evidence', intent.evidence], ['source', intent.source]
  ];
  if (dispatched) rows.push(['reading', intent.task]);
  rows.forEach(function (r) {
    var c = U.el('span', 'chip chip--plain');
    c.innerHTML = '<span class="k">' + r[0] + '</span>' + r[1];
    intentHost.appendChild(c);
  });
}
```
(`frontend/assets/js/mission.js:344-356`)

### 10.5 The reading-versus-dispatch distinction, which is a *feature* not a bug

> *"**Read the *dispatched* task from the answer's `[task]` tag when present, else from the intent
> panel's reading** — the panel shows the router's *reading*, and a quantifier upgrade legitimately
> makes the two differ (A5 reads `change`, dispatches `change_vqa`)."*
> (session `HANDOFF_NEXT_AGENT.md` §5.3)

The panel preserves both, on purpose:

```js
if (dispatched) rows.push(['reading', intent.task]);
```
(`frontend/assets/js/mission.js:352`)

with the reasoning in the docstring:

> *"The `task` chip then names what was SENT and a `reading` chip preserves what the router saw —
> showing only one of the two would either misreport the request or hide the router's input."*
> (`frontend/assets/js/mission.js:341-343`)

### 10.6 The safety property that made the harness bugs survivable

> *"Pass 3's raw harness output reports `SUMMARY 0/8` — because it was launched with the harness build
> that still had the two discriminator bugs. Its verdicts in `results_pass3.json` are recomputed from
> the recorded evidence by `recompute_verdicts.py`. This is exactly the intended safety property:
> **the recorded evidence is independent of the verdict computation**, so a harness bug never forces a
> 24-minute browser re-run — and never silently flips a real failure into a pass."*
> (`DELIVERY_REPORT_2026-09-25.md` §3)

That is the generalisable lesson: **record evidence, compute verdicts separately.** A harness that
computes its verdict inline has no way to re-derive it when the verdict logic turns out to be wrong.

### 10.7 The three live passes

| Pass | Target | Result | Raw output |
|---|---|---|---|
| 1 | `ff46eba42b18` + `d413d3672311` | 8/8 | `run_output.txt` |
| 2 | final HEAD `2d7ae53b482d`, asserting harness | 8/8 | `run_final2.txt` → `results_final.json` |
| 3 | final HEAD `2d7ae53b482d`, repeat | 8/8 | `run_final3.txt` → `results_pass3.json` |

> *"24 live runs, 24 correct dispatches, no run id repeated across passes."*
> (`DELIVERY_REPORT_2026-09-25.md` §3)

The eight cases and their pass-2 run ids:

| case | query | expected | dispatched | pass 2 run_id |
|---|---|---|---|---|
| A1 | What type of terrain dominates this scene? | vqa | vqa | `run_0843db184e32` |
| A2 | Describe the main visual characteristics of this scene. | caption | caption | `run_5b766f2d7df7` |
| A3 | Where are the visible buildings in this image? | grounding | grounding | `run_ea590b6fd70f` |
| A4 | What changed between the earlier and later image? | change | change | `run_65a4b2f9d912` |
| A5 | Did the coastline advance between the two observations? | change_vqa | change_vqa | `run_efe24b98d217` |
| A6 | …combining the optical and SAR observations? | optical_sar | optical_sar | `run_6375b80dcb8e` |
| **B1** | **Where are the built-up areas in this image?** | **grounding** | **grounding** | **`run_2a07dcdbae96`** |
| **B2** | **Where is the new airport?** | **grounding** | **grounding** | **`run_9134f40a258c`** |

(`DELIVERY_REPORT_2026-09-25.md` §3)

> **A6's query is elided in the source as `"…combining the optical and SAR observations?"`** — the
> leading words are not reproduced in the report, and this chapter does not invent them.

### 10.8 The two verdicts, kept separate

> *"1. **Deployment / integration: PASS** — the full pipeline works on unseen imagery and questions.
> 2. **Model quality: MIXED** — caption and grounding are meaningful; change/change_vqa are plausible;
>    VQA is weak-but-related; optical-SAR still returns a bare class index
>    (`class_18 (margin 1.000; optical channels 4/12, SAR channels 2/2)`), not a human label."*
> (`DELIVERY_REPORT_2026-09-25.md` §3)

This is the style guide's rule applied at the harness level: *"a mixed result is never 'all work
perfectly'."* The integration passes; the model quality does not, and the two are not merged.

### 10.9 The harness's hard constraints

| Constraint | Detail |
|---|---|
| `browser-use` block-buffers stdout | *"the output file sits at 0 bytes until the process exits — that looks exactly like a stall but is not"* |
| `grep` block-buffers when piped | *"piping the harness through `grep` swallows all output if the pipeline is killed — redirect to a file"* |
| sandbox proxy is dead | *"Every network call needs `--noproxy '*'` (curl) or `ProxyHandler({})` / `--no-proxy-server` (Python / browser)"* |
| the harness is a `.harness` script | piped to `browser-use.exe` via stdin; helpers are `goto_url`, `upload_file`, `fill_input`, `type_text`, `press_key`, `js`, `capture_screenshot`, `wait_for_element` |

(session `HANDOFF_NEXT_AGENT.md` §4, §5.1, §5.5)

---

## 11. What is NOT RUN, OPEN, SUPPORTED or BLOCKED for this topic

| Item | Status | Detail |
|---|---|---|
| The Analyze console's live path | **VERIFIED** | `mission.js` live driver; 8/8 × 3 passes; real `run_*` ids (`docs/FINAL_DELIVERY_TODO.md` §6 E-11, E-14) |
| The preview/mock path | **SUPPORTED** | *"`runMock` only when no file selected; emits empty payloads, marked `is-mock`; not in production path"* (`docs/FINAL_DELIVERY_TODO.md` §1.4) |
| The trace bar's fill | **VERIFIED (live)** | measured 94.4444 % (`docs/FINAL_DELIVERY_TODO.md` §1.4) |
| Benchmark page | **VERIFIED (section 03 only)** | §03's reliability curve is real from `artifacts/calibration_v001.json`; the remaining section-03 PR curves are *"still labelled illustrative"* (`docs/FINAL_DELIVERY_TODO.md` §4, P6-T01 note) |
| Research page | **VERIFIED** | *"entries now trace to real artifacts/reports with honest limitations"* (`docs/FINAL_DELIVERY_TODO.md` §4, P7-T01) |
| Journey/Lab page | **VERIFIED** | *"stages correspond to real phases/reports; implemented/verified/attempted/blocked distinguished"* (`docs/FINAL_DELIVERY_TODO.md` §4, P7-T02) |
| Anatomy of a Run | **VERIFIED (was SYNTHETIC)** | rebuilt around the real captured `run_d124d8b9adea`; *"synthetic `SEED=917`/`SQ-RUN-0917`/`a3f19c2`/`0.74` removed"* (`docs/FINAL_DELIVERY_TODO.md` §4, P8-T01/T02) |
| HF header link | **VERIFIED** | present in all 11 navs; live DOM-confirmed (`docs/FINAL_DELIVERY_TODO.md` §6 E-13) |
| GitHub header link | **VERIFIED** | target is the only public repo (`docs/FINAL_DELIVERY_TODO.md` §4, P9-T01) |
| Cache-busting | **VERIFIED (with a caveat)** | see §9.3 — the live `curl -I` confirmation is not in the recorded evidence |
| `frontend/.tools/shot.sh` rendering the live tree | **NOT RUN** | the script pointed `ROOT` at the **retired prototype**; the fix was specified and *"was **never started**"* (`HANDOFF_NEXT_AGENT.md` §0, §4.2 item E) |
| Five audited visual defects (contrast, occluded disclosure, `[hidden]`, 390 px overflow, `shot.sh`) | **NOT RUN** | *"I authorised all five and sent the spec, but the session was interrupted before any file was touched."* (`HANDOFF_NEXT_AGENT.md` §4.2) — `mission.html` mtime and the untouched `system.css` are the verification |
| The client/server `image/geotiff` asymmetry | **OPEN (defect, undocumented elsewhere)** | the client's `CONTENT_TYPES` map has no `geotiff` key while the server's allowlist has `image/geotiff` (§7.3) |
| The plan's seven viewer tabs vs the shipped four modes | **DIVERGENCE, recorded** | see §4.2 |
| **UNKNOWN — not established from the available evidence** | — | whether the deployed bundle's `_headers` is byte-identical to the working tree's; the live `Cache-Control` header on `/assets/js/*`; the measured rendering of the five authorised-but-unstarted visual fixes; the A6 query's leading words |

### 11.1 Two honesty constraints that outlive the sprint

> *"**Imagery honesty:** everything in `frontend/assets/img/eo/` is Copernicus / ESA / NASA reference
> material with `satquery_result: false` and `role: illustrative`. Nothing may imply it is SatQuery
> pipeline output. **Never replace a fake value with another fake value** — either measure it or label
> it with `.disclose`."* (`HANDOFF_NEXT_AGENT.md` §7, repo copy)

> *"**Never fabricate.** No invented confidence values, areas, RMSE, run IDs, acquisition dates,
> lat/lon, model outputs or execution times."* (`HANDOFF_NEXT_AGENT.md` §7, repo copy)

---

## 12. Where the evidence lives

| Claim class | File | What it establishes |
|---|---|---|
| the 8 events, the 9 states, the policy | `frontend/assets/js/core.js` | `SQ.EVENT_NAMES` (:616), `SQ.STAGES` (:605), `SQ.policy` (:623), `SQ.scene` (:207) |
| the live client | `frontend/assets/js/live.js` | endpoints (:55), base URL (:99), upload (:202), infer (:294), run (:345) |
| the console | `frontend/assets/js/mission.js` | `runMock` (:556), `runLive` (:601), `onEvent` (:502), `markState` (:404), the fill formula (:423) |
| the cache/security policy | `frontend/_headers` | the concatenation finding (:3-9), the EO note (:66-76), the JS/CSS rule (:50-60) |
| the page set and the header nav | `frontend/*.html` | 11 files; the two external links on each |
| the staging pipeline | `scripts/stage_pages.mjs` | the reference walk, the 25 MiB limit (:43), the exit codes (:25-32) |
| hermeticity, the film, what was not created | `docs/DEPLOYMENT_DECISION.md` | §3, §6, §7 |
| the Pages tier and its one live page | `docs/DEPLOYMENT_TOPOLOGY.md` | §3.1 and the header correction |
| the status board, the Cloudflare finding, B-08 | `docs/FINAL_DELIVERY_TODO.md` | §1.4, §1.7 item 9, §4, §5, §6 |
| the live validation and the harness trap | `DELIVERY_REPORT_2026-09-25.md` | §1, §3, §5 |
| the hard constraints, the harness rules | session `HANDOFF_NEXT_AGENT.md` | §4, §5 |
| the design law and the imagery-honesty rule | repo `HANDOFF_NEXT_AGENT.md` | §7 |

### 12.1 Cross-references

| For… | Read |
|---|---|
| the topology, the tiers, the tunnel | [02 — Deployment Topology](./02-deployment-topology.md) |
| the controller's nine states in full, and the server-side events | [03 — Request Lifecycle](./03-request-lifecycle.md) §36–§38 |
| the evidence records and the confidence rules the console renders | [06 — Evidence and Confidence](./06-evidence-and-confidence.md) |
| the endpoints the client calls, and their envelopes | [08 — The API Contract](./08-api-contract.md) |
| the health payload, the trace as an observability object, the runbook | [10 — Observability and Operations](./10-observability-and-ops.md) |
