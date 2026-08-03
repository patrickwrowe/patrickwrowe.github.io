# Spec 01 — Architecture and design system

**Project:** patrickrowe.\<tld\> — personal portfolio, writing, CV, interactive demos
**Status:** draft for implementation
**Companion docs:** `02-content-and-publishing.md`, `03-interactive-demos.md`
**Reference implementation:** `landing-page.html` (self-contained; open it in a browser)

---

## 1. What this site is for

One sentence: **convince a technically literate stranger, within about ten seconds, that you build models that do real scientific work — then give them the fastest possible route to the evidence.**

Three audiences, in priority order:

| Audience | What they want | What they need to find in <30s |
|---|---|---|
| Hiring managers / founders at AI-for-science companies | Evidence of shipped, load-bearing ML | Selected work, CV, GitHub |
| Research collaborators | What you actually did and whether it's reproducible | Papers, code, methods |
| People arriving from a paper or a link | Context on one specific thing | That project's page, and a reason to stay |

Two design consequences follow from this and should not be traded away:

1. **No page should require JavaScript to convey its meaning.** Interactive demos are additive; the page must still make its point if the demo never loads.
2. **Every item is presented as evidence, not as a card in a grid.** See the wall-label pattern in §6.

### Non-goals

Not a CMS. Not a company site. No newsletter capture, no cookie banner, no analytics that need one, no chat widget, no "let's connect" CTA. The gallery convention this design borrows from is quiet on purpose.

---

## 2. Stack decision

### Recommendation

| Layer | Choice | Why |
|---|---|---|
| Framework | **Astro 6** | Content-first, ships zero JS by default, type-safe content collections, islands for the demos |
| Language | TypeScript (light) | Mostly `.astro` files; TS only where it earns its keep |
| Styling | **Plain CSS + custom properties**, using Astro's per-component scoped styles | See §2.3 |
| Content | Markdown / MDX in-repo | Git is the CMS |
| Math | `remark-math` + `rehype-katex` | Static render, no runtime MathJax |
| Code | Shiki (built into Astro) | Build-time highlighting, zero client JS |
| Search | **Pagefind** | Static index, no server. Add at Phase 5, not before |
| Host | **Cloudflare Pages** | Free, custom headers via `_headers`, preview deploys, and Workers available if a demo ever needs a proxy |
| Repo | GitHub, deploy on push to `main` | |

Astro 6.0 shipped stable on 10 March 2026 and the line is now at 6.2. <cite index="43-4">The release brought a built-in Fonts API that handles downloading and caching fonts for self-hosting and generates optimised fallbacks</cite>, which removes the usual fiddliest part of a type-led design. <cite index="42-1">Content Security Policy support is also stable in Astro 6.</cite>

**Pin to Astro 6.x.** <cite index="43-3">Astro 7 was in alpha as of mid-2026</cite> — not for this.

One piece of context worth knowing: <cite index="4-1">Cloudflare acquired the Astro Technology Company in January 2026, with the framework staying MIT-licensed and open source.</cite> Practically this means the Astro + Cloudflare Pages combination is the best-supported path, and it's the one to take.

### 2.1 Why Astro over the alternatives

| Option | Verdict |
|---|---|
| **Astro** | ✅ Zero-JS baseline suits a text-and-image site; islands give you exactly the escape hatch the demos need; content collections give schema-validated frontmatter so a typo in a date fails the build instead of the page |
| Hugo | Fast, but Go templates are unpleasant to extend and the interactivity story is poor. You will want islands. |
| Jekyll / al-folio | The academic default. Ruby toolchain, dated ergonomics, and hard to make look like anything other than an academic template — which is precisely what you said you don't want. |
| Quarto | Genuinely tempting: renders `.qmd` and notebooks natively, good for Distill-style articles. But theming is Bootstrap-constrained and you'd be fighting it for this aesthetic. **Use it as a tool inside the pipeline, not as the site** — see `02-content-and-publishing.md` §4. |
| Next.js | Overkill. You'd ship a React runtime to render a CV. |
| Streamlit / Gradio / Dash as the whole site | No. See §2.4. |

### 2.2 A note on your Dash experience

It transfers less than you'd hope, and that's fine. Dash, Streamlit and Gradio all follow the same model: a Python process holds state, the browser is a thin renderer, and a server must be running. A portfolio site is the opposite: no server, all state in the URL, and the HTML is the product.

The genuinely reusable part of your background is everything upstream of the UI — packaging models, defining clean interfaces, thinking about latency and payload size. `03-interactive-demos.md` is built around putting that to work.

### 2.3 Why plain CSS and not Tailwind

This is a close call and reasonable people differ. The reasoning here:

- The design is small — roughly ten components — and its whole character lives in precise spacing, hairline rules and type detail. Those are easier to tune and review in one token file than spread across utility strings.
- **Astro scopes component styles automatically**, which removes the main historical argument for utilities (global CSS bleed).
- The prototype's CSS lifts into the Astro project essentially unchanged.
- Fewer moving parts for someone who is not a full-time web developer.

If you later decide you want Tailwind (v4's CSS-first `@theme` config maps the tokens in §5 across almost one-to-one), that's a contained migration. Don't start there.

### 2.4 What *not* to do

Do not build the site in Streamlit, Gradio or Dash. Slow first paint, poor SEO, no design control, and a server that must stay warm to show someone your CV. If a specific demo is naturally Gradio-shaped, embed it in an iframe on one project page — see `03-interactive-demos.md` §6.3.

---

## 3. Repository layout

```
patrickrowe.dev/
├── astro.config.mjs
├── package.json
├── public/
│   ├── patrick-rowe-cv.pdf
│   ├── demos/                    # large demo assets: .onnx, .wasm, precomputed .json
│   ├── figures/                  # static SVG/PNG plates
│   └── _headers                  # Cloudflare: caching + COOP/COEP for demo routes
├── scripts/                      # Python. The content pipeline lives here.
│   ├── bib_to_json.py            # publications.bib  -> src/data/publications.json
│   ├── notebook_to_post.py       # .ipynb            -> src/content/writing/
│   └── export_onnx.py            # PyTorch checkpoint -> public/demos/<slug>/model.onnx
├── src/
│   ├── content.config.ts         # collection schemas (Zod)
│   ├── content/
│   │   ├── work/                 # one .md/.mdx per project
│   │   └── writing/              # one .md/.mdx per post
│   ├── data/
│   │   ├── publications.json     # generated — do not hand-edit
│   │   └── cv.yaml               # single source of truth for the CV
│   ├── components/
│   │   ├── WallLabel.astro       # the signature component (§6.1)
│   │   ├── Plate.astro           # figure + caption
│   │   ├── SiteHead.astro
│   │   ├── SiteFoot.astro
│   │   └── Demo.astro            # click-to-load demo frame (see doc 03)
│   ├── demos/                    # demo source; one dir per demo
│   ├── layouts/
│   │   ├── Base.astro
│   │   └── Prose.astro           # long-form: math, figures, footnotes
│   ├── pages/
│   │   ├── index.astro
│   │   ├── work/index.astro
│   │   ├── work/[...slug].astro
│   │   ├── writing/index.astro
│   │   ├── writing/[...slug].astro
│   │   ├── cv.astro
│   │   ├── about.astro
│   │   ├── 404.astro
│   │   └── rss.xml.ts
│   └── styles/
│       ├── tokens.css            # §5. The only place colours and type are defined.
│       └── global.css            # reset + base elements
└── README.md
```

**Rule:** Python owns the content pipeline (`scripts/`), JavaScript owns the rendering. This keeps you in a language you're fluent in for everything that involves data, and confines the JS to templating.

---

## 4. Information architecture

Four top-level routes. Resist adding a fifth.

```
/                     Landing. Statement, one plate, ~4 selected works, ~3 recent notes.
/work/                Full catalogue. Filterable by tag (no JS: tags are links to /work/tag/x/).
/work/[slug]/         Project page. Long-form, figures, links out, optionally a demo.
/writing/             Reverse-chronological notes.
/writing/[slug]/      Post. Prose layout: math, code, figures, optional islands.
/cv/                  HTML CV generated from cv.yaml, plus a PDF link.
/about/               Short bio, photo, contact, the climbing/mountaineering paragraph.
/rss.xml              Feed for /writing.
/404
```

### Deliberate decisions

- **Demos do not get their own top-level section.** A demo is a property a project has, not a category. `/demos/` would be a graveyard the moment you have one demo and four projects. If you eventually have six demos, add `/work/?filter=demo` and revisit.
- **Publications do not get their own top-level section either.** They render as a block on `/cv/` and are cross-linked from the relevant project pages. Google Scholar is the canonical list; don't compete with it.
- **`/work/` is the catalogue and `/writing/` is the notebook.** A project is finished and load-bearing; a note is a thought. Keeping them separate stops half-formed posts diluting the work.

### Navigation

Header: `Work · Writing · CV`. That's it. About is reached from the footer and from your name. The nav is not sticky — the gallery convention is that chrome gets out of the way.

---

## 5. Design system

The direction: **the exhibition catalogue, translated into scientific vernacular.** Gormley and Carl Rowe's sites work because the apparatus is tiny and the work is large. Clarity and Distill work because the metadata around a figure is precise. The synthesis is a site where every item — project, paper, post, demo — is typeset as a **wall label**, with the artist's *Medium / Dimensions / Collection* replaced by *Method / System / Result*.

### 5.1 Colour

Derived from graphite: cool paper, near-black ink, and the faint blue-violet lustre of a graphite surface as the only accent.

```css
:root{
  --paper:    #F7F7F4;  /* page */
  --plate:    #EFEFEA;  /* figure containers, table zebra */
  --ink:      #15161A;  /* primary text, strong rules */
  --graphite: #63666E;  /* secondary text, all metadata */
  --hairline: #DEDDD7;  /* rules, borders */
  --lustre:   #4B4A7C;  /* the only accent */
}
```

**The accent rule, which is the whole discipline of the palette:** `--lustre` marks *state* (hover, focus, current page) and *anomaly* (ring defects in the hero figure, outliers in a chart). It is never used to make something look nice. If you find yourself reaching for it decoratively, use `--graphite` instead.

There is no second accent. There is no gradient. Figures inside prose may use their own colour scales — that's data, and different rules apply.

Dark mode: **defer.** It roughly doubles the design surface and this palette's whole character is paper. Revisit only if you actually want it.

### 5.2 Type

Three roles, three faces. All free, all self-hostable via Astro's Fonts API.

| Role | Face | Used for |
|---|---|---|
| Display / UI | **Instrument Sans** (variable, 400/500/600) | Headings, statements, titles, nav |
| Long-form | **Source Serif 4** (variable, opsz axis) | Body prose, captions, wall-label values |
| Utility | **IBM Plex Mono** (400/500) | Every piece of metadata, code, figure numbers, dates |

The split carries meaning and should be enforced: **sans labels, serif reads, mono is data.** Mono is what makes a wall label read as a specification rather than a description — it's the single most important typographic decision on the site.

```css
--sans:  "Instrument Sans", system-ui, sans-serif;
--serif: "Source Serif 4", Georgia, serif;
--mono:  "IBM Plex Mono", ui-monospace, monospace;
```

**The label primitive.** Every metadata field on the site uses exactly this:

```css
.label{
  font-family: var(--mono);
  font-size: 0.6875rem;      /* 11px */
  font-weight: 500;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--graphite);
}
```

Scale (fluid where it matters):

| Token | Size | Use |
|---|---|---|
| `--t-statement` | `clamp(2rem, 5.1vw, 3.35rem)` / lh 1.06 / ls −0.028em | Hero only. One per page, maximum. |
| `--t-h2` | 1.5rem | Section headings inside prose |
| `--t-title` | 1.375rem | Wall-label titles |
| `--t-body` | 1.0625rem / lh 1.65 | Serif prose |
| `--t-small` | 0.9375rem | Wall-label values, footer |
| `--t-label` | 0.6875rem | The label primitive |

Measure: prose caps at **62ch**, hero paragraph at **56ch**, statement at **19ch**.

### 5.3 Layout

```
--shell:  1180px;
--gutter: clamp(1.25rem, 5vw, 3.5rem);
```

Spacing is a 4px base. Section rhythm is `clamp(4rem, 9vw, 6.5rem)` block padding. Rules do the structural work: `1px solid var(--hairline)` between items, `1px solid var(--ink)` under a section heading and above the footer. **No boxes, no shadows, no border-radius above 2px, no cards.**

### 5.4 Motion

One orchestrated moment on page load: the hero plate is scanned in left-to-right over 1.25s, then the ring defects register 0.95s in. Everything else is a 180ms colour transition on hover. That's the entire motion budget.

`prefers-reduced-motion: reduce` disables all of it — already implemented in the prototype.

---

## 6. Component inventory

### 6.1 `WallLabel.astro` — the signature component

Used on the landing page, the work index, and as the header block of every project page. This is what makes the site cohere.

```
2020                  GAP-20                                    →
                      METHOD   Gaussian approximation potential…
                      SYSTEM   Carbon — graphite, diamond, amorphous…
                      RESULT   Adopted in academia and industry for…
```

Props:

```ts
interface Props {
  year:   string;                 // "2020" | "2022–24"
  title:  string;
  href:   string;
  fields: { label: string; value: string }[];   // 2–4 rows
  flag?:  "Live demo" | "Preprint" | "Code";
}
```

Field vocabulary — keep it to these, so entries are comparable at a glance:

| Field | Contains | Example |
|---|---|---|
| `METHOD` | What the model/technique is | Autoregressive transformer over SMILES |
| `SYSTEM` | What it was applied to | Drug-like small molecules |
| `RESULT` | What it changed or enabled | Deployed across 30+ clinical targets |
| `PUBLISHED` | Venue and year (optional, replaces RESULT on paper entries) | Nature Communications, 2020 |

Behaviour: whole row is the link. Hover raises the background by ~3%, turns the year `--lustre`, and nudges the arrow 3px. **Fields are always visible** — hiding them until hover would be a nice museum gesture and a bad portfolio.

### 6.2 `Plate.astro` — figure and caption

A figure on `--plate` background with a mono `Fig. n` marker and a serif caption, separated by a hairline rule. Optional edge-fade mask so a field of material reads as continuing past the frame rather than being boxed.

Captions are real captions: state what is shown, how it was produced, and why it's here. The hero caption in the prototype is the model.

### 6.3 Others

| Component | Notes |
|---|---|
| `SiteHead` / `SiteFoot` | Static. `aria-current` on the active nav item. |
| `Note.astro` | Compact writing-index row: date, title, one-line kicker. |
| `Prose.astro` | Long-form layout: 62ch measure, KaTeX, Shiki, footnotes, figure styling. |
| `Demo.astro` | Click-to-load frame for interactive demos. Fully specified in doc 03. |
| `PubList.astro` | Renders `publications.json`. Author name emphasised; DOI + PDF links. |

---

## 7. Hosting and deployment

**Cloudflare Pages**, connected to GitHub, deploy on push to `main`, preview deploys on PRs.

### Domain

Register a real domain. `patrickwrowe.github.io` reads as a placeholder and you can't set headers on it — which will block the demos in doc 03. Options in rough order of preference: `patrickrowe.dev`, `patrickwrowe.com`, `patrickrowe.science`. Keep the github.io repo and set it to redirect.

### `public/_headers`

```
/demos/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
  Cache-Control: public, max-age=31536000, immutable

/*.onnx
  Cache-Control: public, max-age=31536000, immutable
/*.wasm
  Cache-Control: public, max-age=31536000, immutable
```

COOP/COEP are needed for `SharedArrayBuffer`, which multi-threaded WASM runtimes want. Scope them to `/demos/*` only — applying them site-wide will break third-party embeds including Vimeo.

### Analytics

Cloudflare Web Analytics: free, cookieless, no banner required. Do not add Google Analytics.

---

## 8. Quality floor

Non-negotiable, checked before each deploy:

- Lighthouse ≥ 95 on performance, accessibility, best practices, SEO for `/`, `/work/`, a project page and a post.
- Total JS on `/` and `/writing/*`: **0 KB** (demo pages excepted).
- Largest Contentful Paint < 1.2s on a simulated 4G connection.
- Keyboard navigable throughout, visible focus rings (`--lustre`, 2px, 3px offset).
- Text contrast ≥ 4.5:1. Note: `--graphite` on `--paper` is ~5.4:1 — fine for body, but do not lighten it further.
- Every image has meaningful `alt`; every figure has a caption.
- Works with JavaScript disabled, demos excepted.
- `prefers-reduced-motion` respected.

Add per-page OG images (Astro can generate them at build time via Satori) at Phase 5, not before.

---

## 9. Build phases

| Phase | Deliverable | Rough effort |
|---|---|---|
| **0** | Scaffold Astro 6, port `tokens.css` and the landing page from the prototype, register domain, deploy to Cloudflare Pages. **Ship this before writing any content.** | half a day |
| **1** | `work` collection + schema, `/work/` index, project page template, `WallLabel`, `Plate`. Three real project pages. | 1–2 days |
| **2** | `/cv/` from `cv.yaml`, PDF link, `bib_to_json.py` + publications block. | half a day |
| **3** | `writing` collection, prose layout, KaTeX + Shiki, RSS, `/about/`. Two real posts. | 1 day |
| **4** | `Demo.astro` framework + first Tier-0 demo (precomputed molecule browser). See doc 03. | 1–2 days |
| **5** | Tier-1 live molecule generation (ONNX + RDKit.js). See doc 03. | 2–4 days |
| **6** | Pagefind search, OG images, `notebook_to_post.py`. | 1 day |

Phase 0 exists to get a real URL live on day one. Everything after is additive.

---

## 10. Risks and open decisions

| Risk | Mitigation |
|---|---|
| Design drifts as content is added and stops looking designed | The wall-label vocabulary in §6.1 is the constraint. New content must fit the existing components; if it genuinely can't, that's a deliberate design decision, not an ad-hoc one. |
| Demo assets balloon page weight | Click-to-load is mandatory. Doc 03 §5. |
| Blog goes stale and makes the site look abandoned | Undated notes, or drop visible dates on evergreen pieces. Three good posts beat twelve thin ones. |
| Astro 7 lands and tempts an upgrade | Stay on 6.x until 7 has been stable for a quarter. Nothing here needs it. |
| The CV exists in three places and diverges | `cv.yaml` is the single source. The PDF is generated from it or, if hand-made, is regenerated whenever the YAML changes. |

### Decisions to revisit later, deliberately deferred

- Dark mode
- Comments (recommendation: never)
- A `/notes/` scratchpad distinct from `/writing/`
- Tailwind migration

---

## Appendix — what the prototype demonstrates

`landing-page.html` is a single self-contained file, no build step. Open it directly. It exists to pin down the visual language, and every rule above is already implemented in it.

Note the hero figure specifically. It is not stock art: it's a 265-atom three-coordinate carbon network, built by taking a hexagonal graphene sheet, applying 26 Stone–Wales bond rotations, and relaxing the geometry under harmonic bond and 1–3 terms — which is why the ring census comes out hexagon-dominated with paired 5- and 7-membered rings, and bond lengths within 2.2% of each other. The generator is ~150 lines of numpy (`gen_network.py` in the working notes) and should live in `scripts/`.

That matters beyond this one figure: **the site's imagery should be generated from the science, not sourced.** It's the thing that will most reliably distinguish this from every other minimal ML portfolio, and it's the part you're best equipped to do.
