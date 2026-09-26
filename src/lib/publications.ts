/**
 * Pure formatting helpers for the publication record — restructure spec §3.2 to §3.4.
 *
 * Nothing here imports from Astro, so `npm test` (node --test) exercises it directly and
 * the Astro components stay thin. Author strings are in initials form, "P. Rowe".
 */

export const ME = "P. Rowe";

export type PublicationKind = "paper" | "poster" | "thesis" | "preprint" | "manuscript";

export interface AuthorPart {
  text: string;
  me: boolean;
}

const ELLIPSIS: AuthorPart = { text: "…", me: false };

/** An initial, with or without a hyphenated second initial: "P." or "J.-M.". */
const INITIAL = /^\p{Lu}\.(?:-\p{Lu}\.)*$/u;

/** "V. L. Deringer" -> "Deringer"; "V. de Puyraimond" -> "de Puyraimond". */
export function surname(name: string): string {
  const parts = name.trim().split(/\s+/);
  let index = 0;
  while (index < parts.length - 1 && INITIAL.test(parts[index])) index += 1;
  return parts.slice(index).join(" ");
}

/** Surnames for a citation line: up to five joined with commas and "and", "et al." beyond. */
export function citationAuthors(authors: string[]): string {
  const names = authors.map(surname);
  if (names.length > 5) return `${names[0]} et al.`;
  if (names.length === 1) return names[0];
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

/** "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "101st". */
export function ordinal(n: number): string {
  const lastTwo = n % 100;
  if (lastTwo >= 11 && lastTwo <= 13) return `${n}th`;
  const suffixes: Record<number, string> = { 1: "st", 2: "nd", 3: "rd" };
  return `${n}${suffixes[n % 10] ?? "th"}`;
}

/**
 * Author list for the Publications band. Up to ten authors in full. Beyond that: the
 * first three, an ellipsis, Patrick with his position if he is not already shown, and
 * the last author. The component joins the parts with ", " and bolds `me`.
 */
export function listAuthors(authors: string[], me: string = ME): AuthorPart[] {
  const part = (text: string): AuthorPart => ({ text, me: text === me });
  if (authors.length <= 10) return authors.map(part);

  const last = authors.length - 1;
  const position = authors.indexOf(me);
  const parts = authors.slice(0, 3).map(part);
  if (position > 2 && position < last) {
    if (position > 3) parts.push(ELLIPSIS);
    parts.push({ text: `${me} (${ordinal(position + 1)} of ${authors.length})`, me: true });
    if (position < last - 1) parts.push(ELLIPSIS);
  } else {
    parts.push(ELLIPSIS);
  }
  parts.push(part(authors[last]));
  return parts;
}

const KIND_MARKERS: Record<PublicationKind, string | null> = {
  paper: null,
  poster: "Poster",
  thesis: "Thesis",
  preprint: "Preprint",
  manuscript: "In preparation",
};

/** The mono marker shown after the venue for anything that is not a journal paper. */
export function kindMarker(kind: PublicationKind): string | null {
  return KIND_MARKERS[kind];
}

const EXTERNAL_LABELS: Record<Exclude<PublicationKind, "manuscript">, string> = {
  paper: "Paper",
  poster: "Poster",
  thesis: "Thesis",
  preprint: "Preprint",
};

/** The one external link per entry, labelled by kind. A manuscript has none. */
export function externalLink(pub: {
  kind: PublicationKind;
  doi?: string;
  pdf?: string;
}): { label: string; href: string } | null {
  if (pub.kind === "manuscript") return null;
  const href = pub.doi ? `https://doi.org/${pub.doi}` : pub.pdf;
  if (!href) return null;
  return { label: EXTERNAL_LABELS[pub.kind], href };
}

/**
 * Reading time for the meta row of a theme article, from the raw MDX body: import
 * lines, tags and JSX expressions are dropped before counting. 230 words per minute,
 * never reported as less than one minute.
 */
export function readingMinutes(body: string, wordsPerMinute = 230): number {
  const text = body
    .replace(/^import .*$/gm, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/\{[^}]*\}/g, " ");
  const words = text.split(/\s+/).filter((token) => /[A-Za-z0-9]/.test(token)).length;
  return Math.max(1, Math.round(words / wordsPerMinute));
}

/** Newest first, then by title, for every list of publications on the site. */
export function byYearDesc<T extends { data: { year: number; title: string } }>(a: T, b: T): number {
  return b.data.year - a.data.year || a.data.title.localeCompare(b.data.title);
}
