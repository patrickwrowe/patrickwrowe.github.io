# CLAUDE.md

Personal portfolio site for Patrick Rowe — ML engineer, computational chemistry/physics.
Astro 6 → Cloudflare Pages. Static-first, zero JS by default, markdown in git, Python for
every data pipeline.

**Status: not yet scaffolded.** `patrickwrowe.github.io/` still holds the old hand-written
static site (`index.html`, `styles.css`, `css/`, `js/`) — there is no `package.json`, no
`src/`, no `scripts/`. Everything below describes the target, so the Invariants, Commands
and Definition of done sections are aspirational until the Astro project exists. Delete this
paragraph once it does.

## Read the spec before building

Three specs live in `docs/001-initial-spec-docs/`. Read the relevant one **before** writing
code — they contain decisions with reasons, and re-deriving them wastes a session.

| Task | Read first |
|---|---|
| Anything visual, any new component, any CSS | `docs/001-initial-spec-docs/01-architecture-and-design-system.md` §5–6 |
| Routes, IA, hosting, headers, performance budget | `docs/001-initial-spec-docs/01-...` §4, §7–8 |
| New post, new project entry, schemas, notebooks, bibliography, CV | `docs/001-initial-spec-docs/02-content-and-publishing.md` |
| Anything interactive, ONNX, WASM, RDKit, a hosted endpoint | `docs/001-initial-spec-docs/03-interactive-demos.md` |

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

## Conventions that differ from the obvious default

- **Python owns anything that transforms data**; JS only renders. Bibliography parsing,
  notebook conversion, figure generation and model export are scripts in `scripts/`. Do not
  add a JS BibTeX parser, a JS notebook converter, or a build plugin that does data work.
- **`src/data/publications.json` is generated.** Edit `scripts/publications.bib` and re-run
  `bib_to_json.py`. Same for anything else marked generated in a header comment.
- **Content bends to the schema, not the reverse.** If a project won't fit the Zod schema in
  `src/content.config.ts`, fix the content. Loosening the schema needs a reason.
- **Figures are SVG produced by a committed script** in `scripts/figures/`, not hand-drawn,
  never a screenshot of a plot. Strip white backgrounds so `--plate` shows through.
- **Every molecular rendering of carbon goes through `scripts/figures/render_cluster.py`.**
  It takes XYZ and emits a monochrome ball-and-stick SVG whose every stroke resolves to
  `var(--ink)`, with depth encoded as opacity. Extend it rather than reaching for VMD,
  Ovito or a fresh script — a screenshot of a viewer arrives with its own palette and
  fights the page. Structures live beside it in `scripts/figures/data/` so figures
  regenerate. Inline the output with `?raw` and `<Fragment set:html={...}>`, never
  `<Image>`, or the custom properties never resolve.
- **Prose is British English** (—ise, —isation). Identifiers and library APIs stay as the
  library spells them.
- **Internal links carry trailing slashes**: `/work/carbon-gap-20/`.

## Commands

```bash
npm run dev              # localhost:4321
npm run build            # also the type/schema check — run before claiming done
npm run preview          # verify the built output, not just dev

uv run scripts/bib_to_json.py            # publications.bib -> publications.json
uv run scripts/notebook_to_post.py ...   # notebook -> writing entry
uv run scripts/export_onnx.py            # checkpoint -> public/demos/<slug>/
```

## Pitfalls

- **RDKit.js leaks.** Every `RDKit.get_mol()` needs a matching `mol.delete()` in a `finally`.
  WASM heap objects are not garbage collected; a few hundred unfreed molecules crash the tab.
- **Pin Astro to 6.x.** Astro 7 is not to be used here. Also pin the vendored
  `@rdkit/rdkit` version — upstream npm maintenance was in transition as of April 2026.
- **COOP/COEP headers are scoped to `/demos/*` in `public/_headers`.** Applying them
  site-wide breaks third-party embeds including Vimeo.
- **Self-host WASM.** Never hotlink a CDN for `onnxruntime-web` or RDKit — it breaks the
  demo when someone else's CDN does and it conflicts with the COEP headers.
- **Export the single forward pass to ONNX, not the sampling loop.** Control flow exports
  badly. Drive the loop from JS.
- **KaTeX CSS loads only on pages with `math: true`.** Don't move it into the base layout.

## Definition of done

A task is not finished until:

1. `npm run build` passes.
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
