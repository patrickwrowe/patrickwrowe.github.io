# Authoring guide

How to add and edit content on this site. Written for the person maintaining it, not
for a web developer.

The short version: **content is markdown files in `src/content/`, and the build
validates them.** If a date is malformed or a summary runs long, `npm run build` fails
with the file and field named. You cannot silently publish a broken page.

---

## 1. The loop

```bash
./setup.sh            # once, or after pulling changes
npm run dev           # http://localhost:4321, reloads on save
```

Edit a markdown file, watch it in the browser, and when you are happy:

```bash
npm run build         # this is the check — it must pass
```

`npm run build` runs the schema validation and the TypeScript check as well as
building. Treat a red build as "not finished", never as "works anyway".

To see what actually ships rather than the dev version:

```bash
npm run preview       # serves dist/ on the same port
```

Worth doing before anything goes public. Dev and build can differ — a stylesheet that
looks fine in dev can end up loading on every page in the build.

---

## 2. Where things live

| You want to change | Edit this |
|---|---|
| A project or paper page | `src/content/work/<slug>.mdx` |
| A blog post or note | `src/content/writing/<slug>.md` |
| The CV | `src/data/cv.yaml` |
| The About page | `src/pages/about.astro` |
| The landing page's opening paragraph | `src/pages/index.astro` |
| Contact / Scholar / GitHub / LinkedIn links | `src/data/cv.yaml` — used by the CV *and* the footer |

**The filename is the URL.** `src/content/work/gap-20.mdx` becomes `/work/gap-20/`.
Rename the file to change the URL; nothing else needs updating.

---

## 3. Adding a paper page

This is the main case. Copy an existing file and edit it — `gap-20.mdx` is the closest
model.

```bash
cp src/content/work/gap-20.mdx src/content/work/graphene-potential.mdx
```

### The frontmatter

Everything between the `---` fences. This is data, not prose — it feeds the wall label
on `/work/` and the landing page.

```yaml
---
title: Development of a machine learning potential for graphene
year: "2018"                      # string, not a number — "2022–24" is also valid
order: 5                          # lower sorts first on /work/
featured: false                   # true puts it on the landing page
summary: >-                       # max 200 characters, used in search results
  A Gaussian approximation potential for graphene, trained on DFT and validated
  against phonon dispersion and thermal properties.

# The wall label. Two to four rows, in this order.
method: Gaussian approximation potential trained on DFT reference data
system: Graphene — pristine and defective sheets
published: Physical Review B, 2018

tags: [interatomic-potentials, carbon, materials]
links:
  # Check the DOI and venue against the actual record — these are format examples,
  # not real references.
  - { label: Paper, url: "https://doi.org/<the-doi>" }
  - { label: Code, url: "https://github.com/patrickwrowe/<repo>" }

math: true                        # only if the body contains equations
stub: true                        # shows a Draft banner — delete when finished
stubNote: >-
  Abstract only so far; needs the transferability discussion.
---
```

### The wall-label vocabulary is fixed

`METHOD`, `SYSTEM`, and then either `RESULT` **or** `PUBLISHED` — never both. Set
`published:` on paper entries and it replaces `result:` automatically.

| Field | Answers | Example |
|---|---|---|
| `method` | What the technique *is* | Autoregressive transformer over SMILES |
| `system` | What it was applied *to* | Drug-like small molecules |
| `result` | What it *changed or enabled* | Deployed across 30+ clinical targets |
| `published` | Venue and year | Nature Communications, 2020 |

Do not invent new field names to fit awkward content. If a project genuinely resists
`method` and `system`, that is a signal the page needs rethinking — the whole point of
the fixed vocabulary is that entries are comparable at a glance.

One live example of this going wrong: `solid-liquid-interfaces.mdx` currently has
`result: With experimentalists at the National Graphene Institute`. That is a
collaboration credit, not an outcome. It should say what the work made possible.

### `links` labels are a closed set

`Paper`, `Code`, `Data`, `Preprint`, `Video`, `Docs`. Anything else fails the build.
Adding one means editing `src/content.config.ts`.

### The body

Plain markdown below the frontmatter. `##` for section headings — do not use `#`, the
page title is already an `<h1>`.

The four questions a good project page answers, in order:

1. **What was the problem?** — why this was hard, for someone outside your field
2. **What did you build?**
3. **What did it change?**
4. **How do I check?** — the papers, the code, the numbers

That last one is where most portfolios fail and where you have the advantage.

---

## 4. Adding a post

```bash
cp src/content/writing/smiles-is-a-strange-language.md src/content/writing/my-post.md
```

```yaml
---
title: What tokenisation costs you
date: 2026-08-14
kicker: One line, max 160 characters — shown on the index and in the RSS feed.
tags: [generative-models]
math: false
dateless: false     # true hides the date on evergreen explainers
stub: false
---
```

Use `.md` by default. Use `.mdx` only when the post needs to import a component — a
figure, an interactive island. MDX has a real build cost; don't pay it for prose.

`dateless: true` is for genuinely evergreen pieces. An explainer that will still be
true in three years shouldn't invite the reader to discount it because it says
"Jul 2026". It leaves the date column blank rather than substituting a label.

---

## 5. Two different "draft" flags

This trips people up. They do different things.

| Flag | Effect |
|---|---|
| `stub: true` | Page is **live and linked**, with a visible Draft banner. Use while filling a page in. |
| `draft: true` | Page is **hidden from the built site** but visible in `npm run dev`. Use for something not ready to exist publicly. |

Every placeholder page on the site right now uses `stub`, which is why the links from
the landing page all resolve. `stubNote` sets the banner text; omit it for a generic
message.

**To finish a page: delete `stub` and `stubNote`.** The banner disappears with them.

---

## 6. Figures

Figures are SVG generated by a committed script, never a screenshot of a plot. The
point is that they can be restyled when the palette changes and corrected when the
data does.

```mdx
import Plate from "@/components/Plate.astro";
import PhononDispersion from "@/components/figures/PhononDispersion.astro";

<Plate n={2}>
  <PhononDispersion />
  <Fragment slot="caption">
    Phonon dispersion for graphene, <b>GAP vs DFT</b>. Computed with the finite
    displacement method; the acoustic branches agree to within 3 cm⁻¹.
  </Fragment>
</Plate>
```

Captions are real captions: what is shown, how it was produced, why it is here.

To make a new figure: write the generator in `scripts/figures/`, export SVG from
matplotlib with `savefig("f.svg")`, strip the white background so the `--plate` colour
shows through, and save the result as an `.astro` component under
`src/components/figures/`.

Note the SVG gotcha that bit the hero figure: **a `<style>` block inside the SVG gets
stripped by the build.** Put the CSS in the component's own `<style>` block instead —
see `AmorphousCarbon.astro`. And if you use a `<mask>`, its gradient stops must be
white; black stops mask everything out regardless of opacity.

---

## 7. Maths

Set `math: true` in frontmatter, then `$E = mc^2$` inline and `$$...$$` for display.
Rendered at build time — no JavaScript reaches the reader.

Without the flag the equations render as unstyled markup. The flag exists so the 330KB
of KaTeX fonts only loads on pages that need it.

---

## 8. Editing the CV

`src/data/cv.yaml` is the single source. `/cv/` is generated from it, and the footer's
contact links come from the same file, so they cannot drift apart.

Dates are free strings (`period: May 2022 – Oct 2024`), matching the mixed precision of
the source CV. Write them however the record actually has them.

The PDF at `public/patrick-rowe-cv.pdf` does not exist yet. When it does, it must be
regenerated whenever this file changes — that is the failure mode this structure exists
to prevent.

---

## 9. Rules that will bite you

- **Internal links need trailing slashes.** `/work/gap-20/`, not `/work/gap-20`.
- **British English in prose** — "—ise", "—isation". Library names stay as spelled.
- **Never write a colour, font or size directly.** Everything comes from
  `src/styles/tokens.css`. A `#4B4A7C` in a component is a bug even when it looks right.
- **`--lustre` marks state and anomaly, never decoration.** Hover, focus, current page,
  defects in a figure. If it is there to look nice, use `--graphite`.
- **No JavaScript on `/`, `/work/`, `/writing/*`.** Demo pages are the only exception.
- **`summary` is capped at 200 characters, `kicker` at 160.** The build enforces both.

---

## 10. Publishing

```bash
npm run build                     # must pass
git add -A
git commit -m "Add the graphene potential paper page."
git push
```

Cloudflare Pages builds and deploys on push to `main`, and gives every pull request a
preview URL — useful for reading a post on a phone before it is public.

For visual changes, screenshot both widths rather than reasoning about the CSS:

```bash
npm run preview &
PLAYWRIGHT_BROWSERS_PATH=./.playwright uv run python scripts/shoot.py screenshots/
```

That writes every route at 1280px and 390px.

---

## 11. Not built yet

So you know where the edges are:

- **Publications list.** `scripts/bib_to_json.py` and `PubList.astro` are specified but
  not written. Paper pages are currently the only place publications appear.
- **The CV PDF**, and a check that it stays in step with `cv.yaml`.
- **All interactive demos.** `molecular-generation.mdx` sets `demo:`, which puts a
  "Live demo" flag on the landing page for something that does not exist. Fix before
  the site goes public.
- **`notebook_to_post.py`**, for turning a notebook into a post.
- **Search**, worth adding at roughly 15 pages. Not before.
- **A photo for the About page.**
