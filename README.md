# patrickwrowe.github.io

Personal site for Patrick Rowe — machine learning for molecules, materials and proteins.

Astro 6, static output, zero JS on every route except demos. Content is markdown in
git; Python owns every pipeline that touches data.

## Getting started

```bash
./setup.sh        # idempotent — Node deps to ./node_modules, Python to ./.venv
npm run dev       # localhost:4321
npm run build     # type/schema check + production build
npm run preview   # serve the built output
```

## Where things are

| Path | What |
|---|---|
| `docs/002-authoring-guide.md` | **How to add and edit pages.** Start here for content. |
| `docs/001-initial-spec-docs/` | The specs. Read before building — see `CLAUDE.md`. |
| `src/styles/tokens.css` | The only place colours, type and scale are defined. |
| `src/content/work/` | One `.mdx` per project. Schema in `src/content.config.ts`. |
| `src/content/writing/` | One `.md`/`.mdx` per post. |
| `src/data/cv.yaml` | Single source for the CV. |
| `scripts/` | Python. The content pipeline. |
| `public/_headers` | Cloudflare headers; COOP/COEP scoped to `/demos/*` only. |

## Status

Phase 0 of the plan in spec 01 §9: the landing page is ported from the prototype and
the four top-level routes exist. Project and post bodies are placeholders and say so
on the page — they carry a visible **Draft** notice, driven by `stub: true` in
frontmatter. Remove the flag when the real copy lands.

Not built yet: the publications block, the CV PDF, every interactive demo, Pagefind
search, and OG images.
