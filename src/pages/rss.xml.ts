import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import type { APIContext } from "astro";
import { isPublished } from "@/lib/content";

// Covers `writing` only, and includes the kicker rather than the full content
// (spec 02 §7).
export async function GET(context: APIContext) {
  const notes = (await getCollection("writing", isPublished)).sort(
    (a, b) => b.data.date.valueOf() - a.data.date.valueOf(),
  );

  return rss({
    title: "Patrick Rowe — Writing",
    description:
      "Notes on machine-learned potentials, generative models for chemistry, and getting models used.",
    site: context.site!,
    items: notes.map((note) => ({
      title: note.data.title,
      description: note.data.kicker,
      pubDate: note.data.date,
      link: `/writing/${note.id}/`,
    })),
  });
}
