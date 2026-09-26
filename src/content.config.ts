import { defineCollection, reference } from "astro:content";
import { glob, file } from "astro/loaders";
// Astro 6 deprecates re-exporting `z` from `astro:content`; import zod directly.
import { z } from "zod";

// Schemas are from spec 02 §2.1 and §2.2, amended by the restructure spec §3.4 and §3.5
// (docs/superpowers/specs/2026-09-26-content-restructure-and-rewrite-design.md). The
// schema is doing real work: it forces every entry to declare a `method` and a `system`,
// which is what keeps the catalogue comparable. If an entry resists those fields, that is
// a signal the page needs rethinking, not that the schema needs loosening.

const LINK_LABELS = ["Paper", "Code", "Data", "Preprint", "Video", "Docs", "Poster", "Thesis"] as const;

const work = defineCollection({
  loader: glob({ pattern: "**/*.mdx", base: "./src/content/work" }),
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      // Used in "Read in <shortTitle> →" links from the publications list. Themes set it.
      shortTitle: z.string().max(24).optional(),
      year: z.string(), // "2020" | "2022–24" | "2016–26" — string, not number
      order: z.number().default(100),
      featured: z.boolean().default(false),
      summary: z.string().max(200),

      // theme: a thematic article covering several publications (restructure spec §3.3).
      // project: a standalone page. paper: a per-paper page; retired in session 8 of the
      // restructure, when this value and `authors`, `doi`, `published` are removed.
      kind: z.enum(["theme", "project", "paper"]).default("project"),
      authors: z.string().optional(), // full author list, as published (paper pages only)
      doi: z.string().optional(),

      // Figures reproduced from a published paper need their source and licence
      // stated. No credit, no reproduction.
      heroCredit: z.string().optional(),
      // Conference posters: a full-size image linked from a thumbnail. A plain link
      // is click-to-enlarge without a byte of JavaScript.
      poster: z.string().optional(),
      posterAlt: z.string().optional(),

      // The wall-label fields. See spec 01 §6.1. Two to four, in this order.
      method: z.string(),
      system: z.string(),
      result: z.string().optional(),
      published: z.string().optional(), // venue + year; replaces `result` on paper entries

      tags: z.array(z.string()).default([]),
      links: z
        .array(
          z.object({
            label: z.enum(LINK_LABELS),
            url: z.string().url(),
          }),
        )
        .default([]),
      hero: image().optional(),
      heroAlt: z.string().optional(),
      demo: z.string().optional(),
      // Hidden from production builds, visible in `astro dev`. A theme article carries
      // draft: true from creation until Patrick approves it.
      draft: z.boolean().default(false),
      // Loads the KaTeX stylesheet on this page only, same as `writing`. Without it
      // maths in the body renders as unstyled markup.
      math: z.boolean().default(false),

      // Marks an entry whose prose is placeholder pending real copy, so the page can
      // say so plainly instead of reading as finished.
      stub: z.boolean().default(false),
      // Rendered as a DraftNotice above the body. Same mechanism as `writing`, so
      // there is one way to mark a page provisional across both collections.
      stubNote: z.string().optional(),
    }),
});

const writing = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/writing" }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    kicker: z.string().max(160),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
    math: z.boolean().default(false),
    dateless: z.boolean().default(false),
    source: z.string().optional(),
    stub: z.boolean().default(false),
    // Rendered as a DraftNotice above the post. Lives in frontmatter so plain .md
    // files get the same visible marking as .mdx ones, which can import components.
    stubNote: z.string().optional(),
  }),
});

// The publication record — restructure spec §3.4. One file feeds the Publications band
// on /work/, each theme article's closing list, the Cite opener at the top of every
// anchor-target section, and the block on /cv/. Validation here is the point: a typo in
// `theme` or a missing external link fails the build instead of shipping a dead link.
// Entries are keyed by `id`, which is the slug of the paper page the entry replaces.
const publications = defineCollection({
  loader: file("./src/data/publications.yaml"),
  schema: z
    .object({
      kind: z.enum(["paper", "poster", "thesis", "preprint", "manuscript"]),
      title: z.string(),
      // Initials form, "P. Rowe". lib/publications.ts bolds Patrick by exact match.
      authors: z.array(z.string()).min(1),
      // Absent on a manuscript; its kind marker reads IN PREPARATION instead.
      venue: z.string().optional(),
      year: z.number().int(),
      doi: z.string().optional(),
      // Site-relative, for posters and the thesis: "/posters/abcellera-sitc-2023-tce-platform.pdf".
      pdf: z.string().optional(),
      theme: reference("work"),
      // The id of the `##` heading in the theme article. Astro slugs heading text, so
      // this is lowercase ASCII words joined by hyphens. check_links.py confirms it exists.
      anchor: z.string().regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/),
      // Three sentences, 45 to 65 words (restructure spec §4.4). Optional until the
      // theme's session writes it; check_links.py fails a live theme whose entries lack one.
      blurb: z.string().max(420).optional(),
      links: z
        .array(
          z.object({
            label: z.enum(["Preprint", "Data", "Code", "Docs"]),
            url: z.string().url(),
          }),
        )
        .default([]),
    })
    .refine((entry) => !(entry.doi && entry.pdf), { message: "doi and pdf are exclusive" })
    .refine((entry) => (entry.kind === "manuscript") !== Boolean(entry.doi || entry.pdf), {
      message: "a manuscript has no doi or pdf; every other kind has exactly one",
    }),
});

export const collections = { work, writing, publications };
