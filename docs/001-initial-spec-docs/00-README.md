# Personal site — planning set

Five files. Read them in this order.

| File | What it is |
|---|---|
| **`landing-page.html`** | Working prototype. Open it in a browser first — it's the fastest way to judge the direction, and everything in doc 01 is already implemented in it. Self-contained, no build step. |
| **`01-architecture-and-design-system.md`** | Stack decision and why, information architecture, content model, and the full design system: tokens, type, components. The main spec. |
| **`02-content-and-publishing.md`** | How content reaches the site. Collection schemas, the blog, notebooks → posts, publications from BibTeX, the CV as one source. |
| **`03-interactive-demos.md`** | The Python-to-web bridge. Three tiers with a decision rule, the demo contract, and the molecular generation demo specified end to end. |
| **`gen_network.py`** | The generator behind the hero figure. Amorphous graphene via Stone–Wales rotations on a hexagonal sheet, relaxed. ~150 lines of numpy, no dependencies beyond it. Belongs in `scripts/`. |

## The short version

**Stack:** Astro 6 → Cloudflare Pages. Content as markdown in git. Plain CSS with a tight token system, using Astro's scoped component styles. Python owns every pipeline that touches data; JavaScript only renders.

**Style:** the exhibition catalogue translated into scientific vernacular. Every project, paper and demo is typeset as a **wall label** — the artist's *Medium / Dimensions / Collection* becomes *Method / System / Result*. Graphite palette, one accent used only for state and anomaly, three typefaces with strict roles: sans labels, serif reads, mono is data.

**Demos:** precompute wherever possible; run in the browser via ONNX + RDKit.js where interaction genuinely needs computation; host a Python API only when a GPU or private data forces it. Nothing loads until the visitor asks for it, and every demo has a static fallback.

## Do these three things first

1. **Open `landing-page.html`** and decide whether the direction is right. If it isn't, that's a cheap thing to find out now — say what's off and it can be redirected before any of the spec becomes commitments.
2. **Register a domain.** `patrickwrowe.github.io` reads as a placeholder and can't set the HTTP headers the demos need. This blocks nothing else, but it has a lead time.
3. **Ship Phase 0** — scaffold, tokens, landing page, deploy. Half a day, and it gets a real URL live before any content exists. Everything after that is additive.

## Two things worth flagging

**Content in the prototype is drawn from your CV, and publication venues are placeholders.** Check them before anything goes live.

**The hero figure isn't decoration, and that's the point.** It's a real 265-atom carbon network with honest ring statistics, generated from a script. The recommendation in doc 01 that the site's imagery be *generated from the science rather than sourced* is the single choice most likely to distinguish this from every other minimal ML portfolio — and it's the part you're best placed to do.
