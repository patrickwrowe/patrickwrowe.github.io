import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
// Astro 6 deprecates re-exporting `z` from `astro:content`; import zod directly.
import { z } from "zod";

// Schemas are from spec 02 §2.1 and §2.2. The schema is doing real work: it forces
// every project to declare a `method` and a `system`, which is what keeps the
// catalogue comparable. If a project resists those fields, that is a signal the
// project page needs rethinking, not that the schema needs loosening.

const work = defineCollection({
  loader: glob({ pattern: "**/*.mdx", base: "./src/content/work" }),
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      year: z.string(), // "2020" | "2022–24" — string, not number
      order: z.number().default(100),
      featured: z.boolean().default(false),
      summary: z.string().max(200),

      // The wall-label fields. See spec 01 §6.1. Two to four, in this order.
      method: z.string(),
      system: z.string(),
      result: z.string().optional(),
      published: z.string().optional(), // venue + year; replaces `result` on paper entries

      tags: z.array(z.string()).default([]),
      links: z
        .array(
          z.object({
            label: z.enum(["Paper", "Code", "Data", "Preprint", "Video", "Docs"]),
            url: z.string().url(),
          }),
        )
        .default([]),
      hero: image().optional(),
      heroAlt: z.string().optional(),
      demo: z.string().optional(),
      draft: z.boolean().default(false),

      // Not in the spec. Marks an entry whose prose is placeholder pending real
      // copy, so the page can say so plainly instead of reading as finished.
      stub: z.boolean().default(false),
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

export const collections = { work, writing };
