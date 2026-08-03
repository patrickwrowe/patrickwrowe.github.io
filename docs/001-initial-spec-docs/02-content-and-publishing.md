# Spec 02 — Content and publishing

**Companion to:** `01-architecture-and-design-system.md`
**Scope:** how content gets from your head (or a notebook, or a `.bib` file) onto the site.

---

## 1. Principle

**Git is the CMS, and Python owns every pipeline that touches data.**

You will not enjoy maintaining a headless CMS, and you don't need one for a single-author site. Markdown files in the repo mean: version history for free, no vendor, no auth, no runtime cost, and editing in the tool you already use. The cost is that publishing requires a commit — which for three-to-ten posts a year is not a cost.

The second half matters as much. Anything that transforms data — bibliography parsing, notebook conversion, figure generation, model export — is a Python script in `scripts/`, not a JavaScript build plugin. You stay fluent, the pieces are independently testable, and you avoid a class of fragile npm dependencies.

---

## 2. Content types

Three collections plus two data files.

| Type | Location | Format | Schema |
|---|---|---|---|
| Work | `src/content/work/*.mdx` | MDX | §2.1 |
| Writing | `src/content/writing/*.{md,mdx}` | MD or MDX | §2.2 |
| Publications | `src/data/publications.json` | JSON, **generated** | §4 |
| CV | `src/data/cv.yaml` | YAML, hand-edited | §5 |

### 2.1 `work` schema

```ts
// src/content.config.ts
import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const work = defineCollection({
  loader: glob({ pattern: "**/*.mdx", base: "./src/content/work" }),
  schema: ({ image }) => z.object({
    title:     z.string(),
    year:      z.string(),                    // "2020" | "2022–24" — string, not number
    order:     z.number().default(100),       // manual sort on the index
    featured:  z.boolean().default(false),    // appears on the landing page
    summary:   z.string().max(200),           // used in meta description + index

    // The wall-label fields. See doc 01 §6.1. Two to four, in this order.
    method:    z.string(),
    system:    z.string(),
    result:    z.string().optional(),
    published: z.string().optional(),         // venue + year; replaces `result` on paper entries

    tags:      z.array(z.string()).default([]),
    links:     z.array(z.object({
                 label: z.enum(["Paper", "Code", "Data", "Preprint", "Video", "Docs"]),
                 url:   z.string().url(),
               })).default([]),
    hero:      image().optional(),
    heroAlt:   z.string().optional(),
    demo:      z.string().optional(),         // demo slug — see doc 03
    draft:     z.boolean().default(false),
  }),
});
```

The schema is doing real work here: it forces every project to declare a `method` and a `system`, which is what keeps the catalogue comparable. If a project resists those fields, that's a signal the project page needs rethinking, not that the schema needs loosening.

Example frontmatter:

```yaml
---
title: GAP-20
year: "2020"
featured: true
order: 1
summary: A machine-learned interatomic potential for carbon, accurate across
  graphite, diamond and amorphous phases.
method: Gaussian approximation potential, trained on DFT reference data
system: Carbon — graphite, diamond, amorphous phases, surfaces
result: Adopted in academia and industry for structure search, phase-diagram
  prediction and graphitisation studies
tags: [interatomic-potentials, carbon, materials]
links:
  - { label: Code, url: "https://github.com/patrickwrowe/Carbon_GAP" }
hero: ./gap20-phase-diagram.svg
heroAlt: Predicted carbon phase diagram compared against experiment.
---
```

### 2.2 `writing` schema

```ts
const writing = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/writing" }),
  schema: z.object({
    title:    z.string(),
    date:     z.coerce.date(),
    kicker:   z.string().max(160),      // one line, shown on the index
    tags:     z.array(z.string()).default([]),
    draft:    z.boolean().default(false),
    math:     z.boolean().default(false),   // load KaTeX CSS only when true
    dateless: z.boolean().default(false),   // evergreen pieces hide their date
    source:   z.string().optional(),        // path to originating notebook, if any
  }),
});
```

`dateless` is worth having. A post explaining what a machine-learned potential learns is as true in three years as today; showing "May 2026" on it invites the reader to discount it. Reserve it for genuinely evergreen explainers.

### 2.3 Drafts

`draft: true` filters an entry out of production builds but keeps it visible in `astro dev`, so you can write in the real layout:

```ts
const posts = await getCollection("writing", ({ data }) =>
  import.meta.env.PROD ? !data.draft : true
);
```

---

## 3. Writing a post

### 3.1 MD or MDX

Use `.md` by default. Use `.mdx` only when a post needs to import a component — a chart, an interactive island, a custom figure. MDX has a real build cost and slightly worse editor support; don't pay it for prose.

The one thing MDX buys you is significant, though, and it's the Distill move: **an interactive figure sitting inline in an argument**, at the exact point where it explains something.

```mdx
---
title: What a machine-learned potential actually learns
date: 2026-07-14
kicker: Descriptors, smoothness, and why transferability is mostly a data problem.
math: true
dateless: true
---

import Plate from "@/components/Plate.astro";
import SOAPExplorer from "@/demos/soap-explorer/Island.tsx";

The descriptor is the whole game. A potential can only distinguish two
environments its descriptor distinguishes...

<SOAPExplorer client:visible />

Move the cutoff radius and watch the second-neighbour shell drop out of
the representation entirely.
```

Note `client:visible` — the island hydrates only when scrolled into view, so a post with three interactive figures still has a zero-JS first paint.

### 3.2 Math

`remark-math` + `rehype-katex`, rendered at build time. KaTeX over MathJax: faster, no runtime, and the output is static HTML you can style. (Clarity uses MathJax; for a static site KaTeX is the better default.)

```js
// astro.config.mjs
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

export default defineConfig({
  markdown: { remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] },
});
```

Load `katex.min.css` only on pages with `math: true` in frontmatter — it's ~23KB and most posts won't need it.

Inline `$E = \sum_i \varepsilon_i$` and display `$$...$$` both work.

### 3.3 Code

Shiki is built into Astro; no configuration needed beyond a theme. Pick a light theme that sits on `--paper` — `github-light` or `min-light` — and override the background to `--plate` so code blocks match the figure containers.

```js
markdown: { shikiConfig: { theme: "min-light", wrap: true } }
```

### 3.4 Figures

Every figure uses `Plate.astro` (doc 01 §6.2): a `Fig. n` marker, the figure on `--plate`, a hairline rule, a serif caption.

Format guidance:

- **SVG for anything generated** — plots, diagrams, structures. Sharp at any zoom, tiny, and themeable with CSS variables so it responds to the palette.
- Export matplotlib to SVG (`savefig("f.svg")`), then strip the white background so `--plate` shows through.
- **WebP/AVIF for photographs**, through Astro's `<Image>`, which handles responsive sizes and lazy loading.
- Never a PNG screenshot of a plot.

Generated figures should be produced by a committed script, not by hand. If a figure can be regenerated from a script in `scripts/figures/`, it can be restyled when the palette changes and it can be corrected when the data does.

### 3.5 Video

Vimeo embeds via a small `<Vimeo id="..."/>` wrapper that renders a click-to-play poster image rather than the iframe. This keeps ~800KB of third-party JS off the page until someone actually wants the video, and avoids needing to relax the COEP headers from doc 01 §7.

---

## 4. Publications, from BibTeX

You have a publication record and Google Scholar is its canonical home. The site's job is a well-typeset list that links out — not a competing database.

**Source of truth:** `scripts/publications.bib`, exported from Scholar or Zotero.

**Pipeline:** a Python script converts it to JSON at commit time. Do not parse BibTeX in the JS build — the libraries are fragile and you'll be debugging someone else's parser at the worst moment.

```python
# scripts/bib_to_json.py
"""publications.bib -> src/data/publications.json

Run after editing the .bib. Commit both files.
    uv run scripts/bib_to_json.py
"""
import json, pathlib
import bibtexparser          # pip install bibtexparser

ME = {"Rowe", "P. Rowe", "Patrick Rowe", "Rowe, Patrick"}
ROOT = pathlib.Path(__file__).resolve().parents[1]

def split_authors(field: str) -> list[str]:
    return [a.strip().replace("\n", " ") for a in field.split(" and ")]

def main() -> None:
    lib = bibtexparser.parse_file(ROOT / "scripts" / "publications.bib")

    entries = []
    for e in lib.entries:
        f = {k.key: k.value for k in e.fields}
        authors = split_authors(f.get("author", ""))
        entries.append({
            "key": e.key,
            "title": f.get("title", "").strip("{}"),
            "authors": authors,
            "mine": [i for i, a in enumerate(authors)
                     if any(m in a for m in ME)],      # so the template can bold you
            "venue": f.get("journal") or f.get("booktitle") or "Preprint",
            "year": int(f.get("year", 0)),
            "doi": f.get("doi"),
            "url": f.get("url"),
        })

    entries.sort(key=lambda x: (-x["year"], x["title"]))
    out = ROOT / "src" / "data" / "publications.json"
    out.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print(f"wrote {len(entries)} publications -> {out.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
```

Rendering: `PubList.astro` reads the JSON, bolds your name using the `mine` indices, and typesets each entry with the mono/serif split from doc 01. Group by year with a mono year marker in the left column — the same grid as the wall label, so the page feels of a piece.

Cross-link: a project page's `links` array points at its papers; the publications list doesn't need to point back.

---

## 5. The CV

**One source: `src/data/cv.yaml`.** Three renderings:

1. `/cv/` — HTML, generated from the YAML at build time. This is the canonical version and the one that gets indexed.
2. `public/patrick-rowe-cv.pdf` — for people who want to download something.
3. The landing page pulls its "currently / previously" line from the same file.

```yaml
# src/data/cv.yaml
current:
  role: Machine Learning Engineer
  org: SandboxAQ
  from: 2026-01
  focus: >
    Small-molecule property prediction, ML engineering for drug discovery,
    protein language modelling.

experience:
  - role: Research Scientist, Protein Engineering
    org: AbCellera Biologics
    location: Vancouver, Canada
    from: 2022-05
    to: 2024-10
    points:
      - AI and simulation-driven workflows for antibody selection and engineering,
        deployed across more than thirty clinical targets.
      - Data architecture and automated pipelines for lab-automation data.
      - Direct engagement with pharmaceutical partners, translating requirements
        into selection strategies and securing contract milestones.
# ... etc
```

The PDF is the awkward one. Options, in order of preference:

1. **Keep the existing PDF and hand-update it**, with a CI check that fails if `cv.yaml` changed and the PDF didn't. Lowest effort, and the PDF is the artefact you care most about looking right.
2. Generate it with Typst from the same YAML. Clean output, fast, but it's a second layout to maintain.
3. Print-stylesheet the `/cv/` page. Tempting and almost always disappointing.

Recommendation: **option 1.** Don't over-engineer the PDF.

---

## 6. Notebooks to posts

Your natural writing medium for technical posts is a notebook. The pipeline should accept that rather than fight it.

```python
# scripts/notebook_to_post.py
"""Convert a notebook into a writing entry.

    uv run scripts/notebook_to_post.py notebooks/smiles-tokenisation.ipynb \
        --slug smiles-is-a-strange-language \
        --title "SMILES is a strange language to model" \
        --kicker "Canonicalisation, invalid strings, and what tokenisation costs you."

Emits src/content/writing/<slug>.md plus public/figures/<slug>/ for the
extracted images. Re-running overwrites, so keep prose edits in the notebook
until the post is final, then stop re-running.
"""
```

Implementation notes:

- Shell out to `jupyter nbconvert --to markdown --output-dir <tmp>`, which writes the markdown plus an assets directory.
- Move assets to `public/figures/<slug>/` and rewrite the image paths.
- Prepend the frontmatter block from the CLI arguments, with `source:` set to the notebook path.
- Strip cells tagged `hide` (set `"tags": ["hide"]` in cell metadata) so setup and imports don't appear.
- Convert the trailing output of each code cell into a fenced block labelled `text`.

**Where Quarto fits.** For a genuinely Distill-shaped article — heavy cross-referencing, sidenotes, numbered equations, Observable figures — Quarto is a better tool than this pipeline, and it renders `.qmd` and notebooks natively. Use it as a *producer of static HTML* dropped into `public/articles/<slug>/`, linked from a `writing` entry, rather than trying to make it render the whole site. That keeps the option open without letting Bootstrap into the design system.

Don't reach for it until you have a piece that actually needs it.

---

## 7. Feeds, sitemap, search

**RSS** at `/rss.xml` via `@astrojs/rss`, covering `writing` only. Include the kicker, not the full content.

**Sitemap** via `@astrojs/sitemap`. Automatic.

**Search:** Pagefind, added at Phase 6. It runs after the Astro build, indexes the output HTML, and ships a static index the client queries — no server, no third party. Don't add it before there are ~15 pages to search; a search box over eight pages is furniture.

---

## 8. Authoring workflow

```bash
# new post
cp templates/post.md src/content/writing/my-post.md   # draft: true
npm run dev                                            # visible at localhost:4321

# from a notebook
uv run scripts/notebook_to_post.py notebooks/x.ipynb --slug x --title "..." --kicker "..."

# publish
# set draft: false, then:
git add -A && git commit -m "post: x" && git push
# Cloudflare Pages builds and deploys on push to main
```

Open a PR instead of pushing to `main` when you want a preview URL to read on a phone before publishing. Cloudflare gives every PR one automatically.

### Pre-publish checklist

- [ ] `kicker` reads well standing alone — it appears on the index and in the feed
- [ ] Every figure has a caption saying what it shows and how it was made
- [ ] Code blocks have a language tag
- [ ] `math: true` set if there's any math
- [ ] Internal links use trailing slashes (`/work/gap-20/`)
- [ ] `summary`/`kicker` under the character limit — the schema will fail the build otherwise

---

## 9. Content strategy

A brief, unsolicited note, because it determines whether the site works.

**The work catalogue carries the weight.** Four excellent project pages beat twenty stubs. A good project page answers: what was the problem, what did you build, what did it change, and how do I check. That last one is where most portfolios fail and where you have an advantage — you have papers, code and adoption to point at.

**Writing is optional and should stay optional.** A blog with three strong posts and no posting schedule is a better signal than a blog with a monthly cadence and nothing to say. The `dateless` flag exists so evergreen explainers don't rot.

**The most valuable thing you can write** is the piece nobody else can: what actually happens when a machine-learned potential meets a system outside its training distribution; what it takes to get a model trusted inside a screening pipeline; why generating valid SMILES is harder than the papers make it sound. You've done all three. Those posts are the ones a hiring manager forwards to someone else.
