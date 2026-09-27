# CLAUDE.md

Personal portfolio site for Patrick Rowe — ML engineer, computational chemistry/physics.
Astro 6 → GitHub Pages, built and deployed by `.github/workflows/deploy.yml` on every push
to `main`. Static-first, zero JS by default, markdown in git, Python for every data
pipeline.

## Read the spec before building

Three specs live in `docs/001-initial-spec-docs/`. Read the relevant one **before** writing
code — they contain decisions with reasons, and re-deriving them wastes a session.

| Task | Read first |
|---|---|
| Anything visual, any new component, any CSS | `docs/001-initial-spec-docs/01-architecture-and-design-system.md` §5–6 |
| Routes, IA, hosting, headers, performance budget | `docs/001-initial-spec-docs/01-...` §4, §7–8 |
| New post, new project entry, schemas, notebooks, bibliography, CV | `docs/001-initial-spec-docs/02-content-and-publishing.md` |
| Anything interactive, ONNX, WASM, RDKit, a hosted endpoint | `docs/001-initial-spec-docs/03-interactive-demos.md` |
| Any content change, the publications record, theme articles, the copy pipeline | `docs/superpowers/specs/2026-09-26-content-restructure-and-rewrite-design.md` (§3 structure, §4 voice, §5 pipeline) |

The three original specs and `docs/002-authoring-guide.md` predate the restructure. Where
they disagree with the restructure spec, the restructure spec wins; it notes each
disagreement. `docs/` is gitignored on purpose (the remote is public), so these files
exist only locally.

If the code and the spec disagree, **stop and say so**. Don't silently follow either one.

## Skills

**Before any change that affects what a page looks like, invoke the `frontend-design`
skill.** That includes: new components, CSS edits, layout changes, typography, spacing,
colour, motion, SVG figures, and any new page template. It applies to modifications, not
just greenfield work.

Do not skip it because a change seems small. The recurring failure mode on this project is
a "quick" style tweak that quietly reintroduces a generic look the design deliberately
avoids.

It is installed as a plugin (`frontend-design:frontend-design`), not vendored in the repo.
If the skill isn't available, say so rather than proceeding.

## Invariants

Do not break these without asking first.

- **Tokens only.** Every colour, font, size and spacing value comes from
  `src/styles/tokens.css`. A literal hex code, `font-family`, or hard-coded `px` font size
  in a component is a bug, even if it looks right.
- **`--lustre` marks state and anomaly, never decoration.** Hover, focus, current page,
  outliers in data, defects in a figure. If it's there to look nice, use `--graphite`.
- **Zero JS on `/`, `/work/`, `/writing/*`.** Demo pages are the only exception. If a task
  seems to require an island on these routes, it probably requires a rethink instead.
- **Wall-label vocabulary is `METHOD / SYSTEM / RESULT`** (or `PUBLISHED` in place of
  `RESULT`). Don't invent new field names to fit awkward content — that's a signal the
  content needs rewriting.
- **Nothing under `public/demos/` is fetched on page load.** Click-to-load only.
- **Every demo has a static fallback** that makes the same point without JS.
- **`prefers-reduced-motion: reduce` disables all animation.** Every time.
- **Four top-level routes.** Work, Writing, CV, About. Adding a fifth is a design decision,
  not an implementation detail.
- **Anchor-target `##` headings are permanent URLs.** They are the plain name of the work,
  sentence case, five words or fewer, unique in the article, exactly as the table in the
  restructure spec §3.6 gives them. No rewrite pass may change one.
- **Every `##` section of a theme article ends with `[Contents ↑](#contents)`.**
  `scripts/check_links.py` fails otherwise.
- **`→` is internal, `↗` is external.** Link text names the target, never "here"; a DOI is
  never the link text.
- **Copy goes through the pipeline** in restructure spec §5: dossier, draft, Opus stylist,
  separate Opus fact audit, Patrick. Blurbs, captions, alt text and wall-label fields are
  part of the article and go through the same audit. The voice guide is
  `docs/voice/derek-lowe.md` and the terminology sheet `docs/voice/terminology.md`.

## Conventions that differ from the obvious default

- **Python owns anything that transforms data**; JS only renders. Bibliography parsing,
  notebook conversion, figure generation and model export are scripts in `scripts/`. Do not
  add a JS BibTeX parser, a JS notebook converter, or a build plugin that does data work.
- **`src/data/publications.yaml` is the publication record**, hand-maintained and validated
  by the `publications` collection in `src/content.config.ts`. It feeds the Publications
  band on `/work/`, each theme article's closing list, the `Cite` opener and, from
  session 7, `/cv/`. An entry's `theme` is a reference to a `work` entry and its `anchor`
  is a `##` heading id in that article; `scripts/check_links.py` confirms every anchor
  exists after a build.
- **`draft` hides a page from production and `stub` marks placeholder prose on a live page.**
  A theme article is `draft: true` until Patrick approves it, then `draft: false` with
  `stub` and `stubNote` removed in the same commit. A theme also needs `shortTitle`, and its
  METHOD / SYSTEM / RESULT are each one noun phrase of at most 90 characters with no
  semicolon and no PUBLISHED; the schema enforces this.
- **Content bends to the schema, not the reverse.** If a project won't fit the Zod schema in
  `src/content.config.ts`, fix the content. Loosening the schema needs a reason.
- **Data figures are SVG produced by a committed script** in `scripts/figures/`: plots,
  statistics and diagrams, not hand-drawn, never a screenshot of a plot. Strip white
  backgrounds so `--plate` shows through. Molecular renderings follow the next bullet.
- **Two renderers, each with a job** (restructure spec §6.2). Single structures and small
  slabs are monochrome SVG from `scripts/figures/render_cluster.py`: every stroke resolves
  to `var(--ink)`, depth is opacity, inline it with `?raw` and
  `<Fragment set:html={...}>`, never `<Image>`. Grids of large boxes and one hero render
  per theme article are greyscale PNG from Blender via `molrender`: one grey per species,
  no hue, view transform Standard, world pure white so `multiply` onto `--plate` shows no
  box, placed with `<Image>` inside a `<Plate>`. Either way the producing script is
  committed under `scripts/figures/` and its input structures under
  `scripts/figures/data/`, so every figure regenerates. Never a screenshot of a viewer.
- **Prose is British English** (—ise, —isation). Identifiers and library APIs stay as the
  library spells them.
- **Internal links carry trailing slashes**: `/work/carbon/`.

## Commands

```bash
npm run dev              # localhost:4321
npm run build            # also the type/schema check — run before claiming done
npm run preview          # verify the built output, not just dev

npm test                                  # node --test: src/lib helpers
uv run scripts/check_links.py             # after npm run build: anchors, hrefs, redirects, return links, Fig. n
uv run pytest scripts/tests               # the link check's own tests
PLAYWRIGHT_BROWSERS_PATH=./.playwright uv run python scripts/shoot.py [outdir] [route ...]
MOLRENDER_PYTHON=/home/patrick/.local/share/mamba/envs/molrender/bin/python
$MOLRENDER_PYTHON scripts/figures/render_box_grid.py --manifest scripts/figures/data/graphitisation/manifest.json   # Blender panels; env is not the venv
uv run scripts/notebook_to_post.py ...    # notebook -> writing entry (not yet written)
uv run scripts/export_onnx.py             # checkpoint -> public/demos/<slug>/ (not yet written)
```

Inside a sandboxed session `uv` cannot write its cache; `.venv/bin/python` and
`.venv/bin/pytest` run the same things.

## Pitfalls

- **RDKit.js leaks.** Every `RDKit.get_mol()` needs a matching `mol.delete()` in a `finally`.
  WASM heap objects are not garbage collected; a few hundred unfreed molecules crash the tab.
- **Pin Astro to 6.x.** Astro 7 is not to be used here. Also pin the vendored
  `@rdkit/rdkit` version — upstream npm maintenance was in transition as of April 2026.
- **COOP/COEP headers are scoped to `/demos/*` in `public/_headers`.** Applying them
  site-wide breaks third-party embeds including Vimeo. On GitHub Pages this file is inert
  (restructure spec §11); a demo that needs COOP/COEP reopens the hosting question.
- **Self-host WASM.** Never hotlink a CDN for `onnxruntime-web` or RDKit — it breaks the
  demo when someone else's CDN does and it conflicts with the COEP headers.
- **Export the single forward pass to ONNX, not the sampling loop.** Control flow exports
  badly. Drive the loop from JS.
- **KaTeX CSS loads only on pages with `math: true`.** Don't move it into the base layout.

## Definition of done

A task is not finished until:

1. `npm run build` passes, then `uv run scripts/check_links.py` reports 0 failed and
   `npm test` passes.
2. Visual changes have been **checked in a browser at 1280px and 390px** — screenshot it,
   don't reason about the CSS.
3. Keyboard navigation and visible focus still work on anything interactive.
4. No new hard-coded colours, fonts or sizes (grep for `#` in `src/components/`).
5. New JS on a non-demo route is zero, or the reason is stated.

## Don't

- No Tailwind, no CSS framework, no component library, no icon package.
- No analytics beyond Cloudflare Web Analytics. No cookies, no banner.
- No dark mode, no comments system, no newsletter capture — all deliberately deferred.
- Don't commit model weights over ~50 MB; keep them in `public/demos/` only if small,
  otherwise fetch from a release asset.
- Don't add a dependency to solve something a 30-line script solves.
