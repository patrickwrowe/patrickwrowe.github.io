import type { CollectionEntry } from "astro:content";

/**
 * Drafts are filtered out of production builds but stay visible in `astro dev`, so
 * you can write in the real layout. Spec 02 §2.3.
 */
export function isPublished({ data }: { data: { draft: boolean } }): boolean {
  return import.meta.env.PROD ? !data.draft : true;
}

/**
 * The wall-label field vocabulary is fixed: METHOD / SYSTEM / RESULT, with PUBLISHED
 * replacing RESULT on paper entries. Spec 01 §6.1 — do not invent new field names to
 * fit awkward content, that is a signal the content needs rewriting.
 */
export function wallLabelFields(data: CollectionEntry<"work">["data"]) {
  const fields = [
    { label: "Method", value: data.method },
    { label: "System", value: data.system },
  ];
  if (data.published) fields.push({ label: "Published", value: data.published });
  else if (data.result) fields.push({ label: "Result", value: data.result });
  return fields;
}

/** Manual sort on the index, then year descending as a tiebreak. */
export function workSort(a: CollectionEntry<"work">, b: CollectionEntry<"work">) {
  if (a.data.order !== b.data.order) return a.data.order - b.data.order;
  return b.data.year.localeCompare(a.data.year);
}

// timeZone: "UTC" matches how frontmatter dates are parsed. Without it, a post dated
// on the first of a month renders as the month before in any zone behind UTC.
const MONTH_YEAR = new Intl.DateTimeFormat("en-GB", {
  month: "short",
  year: "numeric",
  timeZone: "UTC",
});

/**
 * Evergreen pieces hide their date — spec 02 §2.2, which names
 * "what a machine-learned potential learns" as exactly this case.
 *
 * NOTE: the landing-page prototype shows a date on all three sample notes, so spec 02
 * and the prototype disagree here. Following spec 02, since it gives a reason: showing
 * "Jul 2026" on an evergreen explainer invites the reader to discount it. The date
 * column is left empty rather than filled with a substitute label.
 */
export function noteDate(data: CollectionEntry<"writing">["data"]): string {
  return data.dateless ? "" : MONTH_YEAR.format(data.date);
}
