# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""Check the built site for dead anchors, hrefs and redirects, and for article hygiene.

Restructure spec §3.6, "The link check". Runs after ``npm run build`` and reads files
only: the built HTML under ``dist/``, the publication record, the theme articles' MDX,
and the redirect map if it exists. Nothing is fetched.

Checks, each reported per subject as OK, SKIPPED or FAIL:

* ``anchor``: every publication's ``/work/<theme>/#<anchor>`` exists as an id in the
  built page. SKIPPED while the theme is ``draft: true``, since a draft has no built page.
* ``blurb``: every publication of a live theme has a blurb. SKIPPED while draft.
* ``redirect``: every entry of ``src/data/redirects.json`` has a built page at the old
  path and a target whose page and fragment exist.
* ``href``: every site-relative ``href`` in ``dist/`` resolves to a built page or file,
  fragment included.
* ``contents-return``: every ``##`` section of a theme article ends with the line
  ``[Contents ↑](#contents)``.
* ``figures``: the ``Fig. n`` plate markers on each page run 1 to N in order.

Usage:
    uv run scripts/check_links.py                  # from the site root, after npm run build
    uv run scripts/check_links.py --dist dist --src src

Exit status is 1 if anything FAILs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---", re.S)
ID_ATTR = re.compile(r'(?<![\w-])id="([^"]+)"')
HREF_ATTR = re.compile(r'(?<![\w-])href="([^"]+)"')
FIGURE_MARK = re.compile(r'class="label plate__fig">Fig\.\s*(\d+)<')
SECTION_START = re.compile(r"^## ", re.M)
RETURN_LINE = "[Contents ↑](#contents)"


@dataclass(frozen=True)
class Finding:
    """One check applied to one subject."""

    check: str
    subject: str
    status: str  # OK | SKIPPED | FAIL
    detail: str = ""


def parse_frontmatter(text: str) -> dict:
    """Return the YAML frontmatter of an MDX file as a dict, or {} if there is none."""
    match = FRONTMATTER.match(text)
    return yaml.safe_load(match.group(1)) or {} if match else {}


def body_after_frontmatter(text: str) -> str:
    """Return the MDX text after its frontmatter block."""
    match = FRONTMATTER.match(text)
    return text[match.end() :] if match else text


def work_entries(work_dir: Path) -> dict[str, dict]:
    """Map every work slug under src/content/work to its frontmatter."""
    return {
        path.stem: parse_frontmatter(path.read_text()) for path in sorted(work_dir.glob("*.mdx"))
    }


def load_publications(path: Path) -> list[dict]:
    """Load the publication record; an empty file is an empty record."""
    return yaml.safe_load(path.read_text()) or []


def load_redirects(path: Path) -> dict[str, str]:
    """Load the redirect map, or {} when session 8 has not created it yet."""
    return json.loads(path.read_text()) if path.exists() else {}


def route_of(html_path: Path, dist: Path) -> str:
    """'dist/work/carbon/index.html' -> '/work/carbon/'; 'dist/404.html' -> '/404.html'."""
    relative = html_path.relative_to(dist)
    if relative.name == "index.html":
        parent = relative.parent.as_posix()
        return "/" if parent == "." else f"/{parent}/"
    return f"/{relative.as_posix()}"


def built_pages(dist: Path) -> dict[str, str]:
    """Map every built route to its HTML."""
    return {route_of(path, dist): path.read_text() for path in sorted(dist.rglob("*.html"))}


def split_target(target: str) -> tuple[str, str | None]:
    """'/work/carbon/#gap-20?x' -> ('/work/carbon/', 'gap-20')."""
    path, _, fragment = target.partition("#")
    fragment = fragment.split("?", 1)[0]
    return path.split("?", 1)[0], fragment or None


def target_exists(
    path: str, fragment: str | None, pages: dict[str, str], dist: Path
) -> tuple[bool, str]:
    """Whether a site-relative path (and fragment) resolves in the built output."""
    if path in pages:
        if fragment and fragment not in set(ID_ATTR.findall(pages[path])):
            return False, f"page exists but has no id {fragment!r}"
        return True, ""
    if (dist / path.lstrip("/")).is_file():
        return True, ""
    return False, "no such page in dist"


def check_publication_anchors(
    publications: list[dict], work: dict[str, dict], pages: dict[str, str], dist: Path
) -> list[Finding]:
    """Every publication points at a heading that exists in its theme's built page."""
    findings: list[Finding] = []
    for pub in publications:
        theme = pub["theme"]
        path = f"/work/{theme}/"
        subject = f"{pub['id']} -> {path}#{pub['anchor']}"
        if theme not in work:
            findings.append(Finding("anchor", subject, "FAIL", "theme not in src/content/work"))
            continue
        if work[theme].get("draft", False) and path not in pages:
            findings.append(Finding("anchor", subject, "SKIPPED", "theme is draft: true"))
            continue
        ok, detail = target_exists(path, pub["anchor"], pages, dist)
        findings.append(Finding("anchor", subject, "OK" if ok else "FAIL", detail))
    return findings


def check_blurbs(publications: list[dict], work: dict[str, dict]) -> list[Finding]:
    """A live theme's publications all carry a blurb."""
    findings: list[Finding] = []
    for pub in publications:
        subject = pub["id"]
        if work.get(pub["theme"], {}).get("draft", False):
            findings.append(Finding("blurb", subject, "SKIPPED", "theme is draft: true"))
        elif pub.get("blurb"):
            findings.append(Finding("blurb", subject, "OK"))
        else:
            findings.append(
                Finding("blurb", subject, "FAIL", "no blurb on a live theme's publication")
            )
    return findings


def check_redirects(redirects: dict[str, str], pages: dict[str, str], dist: Path) -> list[Finding]:
    """Each redirect has a built page at the old path and a resolvable target."""
    findings: list[Finding] = []
    for old, target in redirects.items():
        subject = f"{old} -> {target}"
        if old not in pages:
            findings.append(
                Finding("redirect", subject, "FAIL", "no redirect page built at the old path")
            )
            continue
        path, fragment = split_target(target)
        ok, detail = target_exists(path, fragment, pages, dist)
        findings.append(Finding("redirect", subject, "OK" if ok else "FAIL", detail))
    return findings


def check_internal_hrefs(pages: dict[str, str], dist: Path) -> list[Finding]:
    """Every site-relative href in the built HTML resolves; only failures are listed."""
    findings: list[Finding] = []
    for route, html in pages.items():
        for href in sorted(set(HREF_ATTR.findall(html))):
            if not href.startswith("/") or href.startswith("//"):
                continue
            path, fragment = split_target(href)
            ok, detail = target_exists(path, fragment, pages, dist)
            if not ok:
                findings.append(Finding("href", f"{route} -> {href}", "FAIL", detail))
    return findings


def check_contents_returns(work_dir: Path, work: dict[str, dict]) -> list[Finding]:
    """Every ## section of a theme article ends with the return line."""
    findings: list[Finding] = []
    for slug, data in work.items():
        if data.get("kind") != "theme":
            continue
        body = body_after_frontmatter((work_dir / f"{slug}.mdx").read_text())
        for section in SECTION_START.split(body)[1:]:
            heading = section.split("\n", 1)[0].strip()
            lines = [line.strip() for line in section.strip().splitlines() if line.strip()]
            ok = bool(lines) and lines[-1] == RETURN_LINE
            findings.append(
                Finding(
                    "contents-return",
                    f"{slug} ## {heading}",
                    "OK" if ok else "FAIL",
                    "" if ok else f"last line must be {RETURN_LINE}",
                )
            )
    return findings


def check_figure_numbers(pages: dict[str, str]) -> list[Finding]:
    """Plate markers on each page run Fig. 1 to Fig. N in order."""
    findings: list[Finding] = []
    for route, html in pages.items():
        numbers = [int(number) for number in FIGURE_MARK.findall(html)]
        if not numbers:
            continue
        expected = list(range(1, len(numbers) + 1))
        ok = numbers == expected
        findings.append(
            Finding(
                "figures",
                route,
                "OK" if ok else "FAIL",
                "" if ok else f"markers are {numbers}, expected {expected}",
            )
        )
    return findings


def run_all(src: Path, dist: Path) -> list[Finding]:
    """Run every check and return the findings in report order."""
    work_dir = src / "content" / "work"
    work = work_entries(work_dir)
    publications = load_publications(src / "data" / "publications.yaml")
    redirects = load_redirects(src / "data" / "redirects.json")
    pages = built_pages(dist)
    return [
        *check_publication_anchors(publications, work, pages, dist),
        *check_blurbs(publications, work),
        *check_redirects(redirects, pages, dist),
        *check_internal_hrefs(pages, dist),
        *check_contents_returns(work_dir, work),
        *check_figure_numbers(pages),
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the checks and print a report; return 1 if anything FAILs."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--src", type=Path, default=Path("src"))
    args = parser.parse_args(argv)
    if not args.dist.is_dir():
        print(f"no build at {args.dist}; run npm run build first")
        return 1

    findings = run_all(args.src, args.dist)
    counts = {
        status: sum(1 for finding in findings if finding.status == status)
        for status in ("OK", "SKIPPED", "FAIL")
    }
    for finding in findings:
        if finding.status != "OK":
            print(f"{finding.status:8} {finding.check:16} {finding.subject}  {finding.detail}")
    print(f"link check: {counts['OK']} ok, {counts['SKIPPED']} skipped, {counts['FAIL']} failed")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
