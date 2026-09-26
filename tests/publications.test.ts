import test from "node:test";
import assert from "node:assert/strict";
import {
  surname,
  citationAuthors,
  ordinal,
  listAuthors,
  kindMarker,
  externalLink,
  readingMinutes,
  byYearDesc,
} from "../src/lib/publications.ts";

test("surname strips initials and keeps particles and double surnames", () => {
  assert.equal(surname("V. L. Deringer"), "Deringer");
  assert.equal(surname("V. de Puyraimond"), "de Puyraimond");
  assert.equal(surname("J.-M. Leyssale"), "Leyssale");
  assert.equal(surname("P. Balakrishna Pillai"), "Balakrishna Pillai");
  assert.equal(surname("P. Rowe"), "Rowe");
});

test("citationAuthors gives up to five surnames in full and et al. beyond", () => {
  assert.equal(
    citationAuthors(["P. Rowe", "V. L. Deringer", "P. Gasparotto", "G. Csányi", "A. Michaelides"]),
    "Rowe, Deringer, Gasparotto, Csányi and Michaelides",
  );
  assert.equal(
    citationAuthors(["C. Schran", "F. L. Thiemann", "P. Rowe", "E. A. Müller", "O. Marsalek", "A. Michaelides"]),
    "Schran et al.",
  );
  assert.equal(citationAuthors(["R. P. Fornari", "P. Rowe"]), "Fornari and Rowe");
  assert.equal(citationAuthors(["P. Rowe"]), "Rowe");
});

test("ordinal handles the teens and the twenties", () => {
  const cases = [
    [1, "1st"], [2, "2nd"], [3, "3rd"], [4, "4th"], [11, "11th"], [12, "12th"], [13, "13th"],
    [21, "21st"], [22, "22nd"], [23, "23rd"], [25, "25th"], [101, "101st"], [111, "111th"],
  ] as const;
  for (const [number, expected] of cases) assert.equal(ordinal(number), expected);
});

test("listAuthors keeps up to ten authors in full and marks Patrick", () => {
  const parts = listAuthors(["P. Rowe", "G. Csányi", "D. Alfè", "A. Michaelides"]);
  assert.deepEqual(parts.map((part) => part.text), ["P. Rowe", "G. Csányi", "D. Alfè", "A. Michaelides"]);
  assert.deepEqual(parts.map((part) => part.me), [true, false, false, false]);
});

test("listAuthors shortens a long list around Patrick's position", () => {
  const authors = Array.from({ length: 31 }, (_, index) => `A${index + 1}. Surname${index + 1}`);
  authors[24] = "P. Rowe";
  const parts = listAuthors(authors);
  assert.deepEqual(parts.map((part) => part.text), [
    "A1. Surname1", "A2. Surname2", "A3. Surname3", "…", "P. Rowe (25th of 31)", "…", "A31. Surname31",
  ]);
  assert.equal(parts.filter((part) => part.me).length, 1);
});

test("listAuthors does not repeat Patrick when he is already in the first three", () => {
  const authors = Array.from({ length: 47 }, (_, index) => `A${index + 1}. S${index + 1}`);
  authors[1] = "P. Rowe";
  assert.deepEqual(listAuthors(authors).map((part) => part.text), ["A1. S1", "P. Rowe", "A3. S3", "…", "A47. S47"]);
});

test("listAuthors puts no ellipsis between Patrick and a neighbouring shown author", () => {
  const authors = Array.from({ length: 12 }, (_, index) => `A${index + 1}. S${index + 1}`);
  authors[3] = "P. Rowe";
  assert.deepEqual(listAuthors(authors).map((part) => part.text), [
    "A1. S1", "A2. S2", "A3. S3", "P. Rowe (4th of 12)", "…", "A12. S12",
  ]);
});

test("kindMarker is empty for a paper and named for everything else", () => {
  assert.equal(kindMarker("paper"), null);
  assert.equal(kindMarker("poster"), "Poster");
  assert.equal(kindMarker("thesis"), "Thesis");
  assert.equal(kindMarker("preprint"), "Preprint");
  assert.equal(kindMarker("manuscript"), "In preparation");
});

test("externalLink labels by kind and resolves a DOI", () => {
  assert.deepEqual(externalLink({ kind: "paper", doi: "10.1063/5.0005084" }), {
    label: "Paper",
    href: "https://doi.org/10.1063/5.0005084",
  });
  assert.deepEqual(externalLink({ kind: "poster", pdf: "/posters/x.pdf" }), { label: "Poster", href: "/posters/x.pdf" });
  assert.deepEqual(externalLink({ kind: "thesis", pdf: "/thesis.pdf" }), { label: "Thesis", href: "/thesis.pdf" });
  assert.equal(externalLink({ kind: "manuscript" }), null);
});

test("readingMinutes counts prose words, not imports or tags, and never reports zero", () => {
  const body =
    'import Plate from "@/components/Plate.astro";\n\n' +
    "word ".repeat(460) +
    '\n<Plate n={1}><Image src={Figure} alt="A figure" /></Plate>\n';
  assert.equal(readingMinutes(body), 2);
  assert.equal(readingMinutes(""), 1);
});

test("byYearDesc sorts newest first, then by title", () => {
  const entries = [
    { data: { year: 2020, title: "B" } },
    { data: { year: 2023, title: "Z" } },
    { data: { year: 2020, title: "A" } },
  ];
  assert.deepEqual(entries.sort(byYearDesc).map((entry) => entry.data.title), ["Z", "A", "B"]);
});
