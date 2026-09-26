"""Tests for scripts/check_links.py against a tiny fabricated site tree."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_links  # noqa: E402

THEME_BODY = textwrap.dedent(
    """\
    ## GAP-20

    <Cite id="carbon-gap-20" />

    [Contents ↑](#contents)

    ## CHO-GAP

    [Contents ↑](#contents)
    """
)


def make_site(
    tmp_path: Path,
    *,
    draft: bool = False,
    blurb: bool = True,
    page_ids: tuple[str, ...] = ("contents", "gap-20", "cho-gap"),
    body: str = THEME_BODY,
    extra_html: str = "",
    redirects: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Build src/ and dist/ trees for one theme article and one publication."""
    src = tmp_path / "src"
    dist = tmp_path / "dist"
    work = src / "content" / "work"
    work.mkdir(parents=True)
    (src / "data").mkdir()
    (work / "carbon.mdx").write_text(
        f"---\ntitle: Carbon\nkind: theme\ndraft: {'true' if draft else 'false'}\n---\n\n{body}"
    )
    blurb_line = "  blurb: One sentence. Two sentences. Three sentences.\n" if blurb else ""
    (src / "data" / "publications.yaml").write_text(
        "- id: carbon-gap-20\n  kind: paper\n  title: T\n  authors: [P. Rowe]\n  year: 2020\n"
        f"  doi: 10.1/x\n  theme: carbon\n  anchor: gap-20\n{blurb_line}"
    )
    if redirects is not None:
        import json

        (src / "data" / "redirects.json").write_text(json.dumps(redirects))
    if not draft:
        page = dist / "work" / "carbon"
        page.mkdir(parents=True)
        ids = "".join(f'<h2 id="{page_id}">x</h2>' for page_id in page_ids)
        page.joinpath("index.html").write_text(f"<html><body>{ids}{extra_html}</body></html>")
    (dist / "work").mkdir(parents=True, exist_ok=True)
    (dist / "work" / "index.html").write_text('<a href="/work/carbon/">c</a>')
    return src, dist


def statuses(findings: list[check_links.Finding], check: str) -> list[str]:
    return [finding.status for finding in findings if finding.check == check]


def test_anchor_ok_when_built_page_has_the_id(tmp_path: Path) -> None:
    src, dist = make_site(tmp_path)
    findings = check_links.run_all(src, dist)
    assert statuses(findings, "anchor") == ["OK"]


def test_anchor_fails_when_id_is_missing(tmp_path: Path) -> None:
    src, dist = make_site(tmp_path, page_ids=("contents", "cho-gap"))
    findings = check_links.run_all(src, dist)
    assert statuses(findings, "anchor") == ["FAIL"]


def test_anchor_fails_when_theme_is_not_in_work(tmp_path: Path) -> None:
    src, dist = make_site(tmp_path)
    (src / "data" / "publications.yaml").write_text(
        "- id: carbon-gap-20\n  kind: paper\n  title: T\n  authors: [P. Rowe]\n  year: 2020\n"
        "  doi: 10.1/x\n  theme: carbn\n  anchor: gap-20\n  blurb: One. Two. Three.\n"
    )
    findings = check_links.run_all(src, dist)
    anchor = next(finding for finding in findings if finding.check == "anchor")
    assert anchor.status == "FAIL"
    assert "not in src/content/work" in anchor.detail


def test_anchor_is_skipped_for_a_draft_theme(tmp_path: Path) -> None:
    src, dist = make_site(tmp_path, draft=True)
    findings = check_links.run_all(src, dist)
    assert statuses(findings, "anchor") == ["SKIPPED"]
    assert statuses(findings, "blurb") == ["SKIPPED"]


def test_missing_blurb_fails_on_a_live_theme(tmp_path: Path) -> None:
    src, dist = make_site(tmp_path, blurb=False)
    findings = check_links.run_all(src, dist)
    assert statuses(findings, "blurb") == ["FAIL"]


def test_redirect_target_must_exist_and_old_page_must_be_built(tmp_path: Path) -> None:
    src, dist = make_site(
        tmp_path,
        redirects={
            "/work/carbon-gap-20/": "/work/carbon/#gap-20",
            "/work/old/": "/work/carbon/#nope",
        },
    )
    old = dist / "work" / "carbon-gap-20"
    old.mkdir()
    old.joinpath("index.html").write_text(
        '<meta http-equiv="refresh" content="0;url=/work/carbon/#gap-20">'
    )
    findings = check_links.run_all(src, dist)
    by_subject = {
        finding.subject: finding.status for finding in findings if finding.check == "redirect"
    }
    assert by_subject["/work/carbon-gap-20/ -> /work/carbon/#gap-20"] == "OK"
    assert by_subject["/work/old/ -> /work/carbon/#nope"] == "FAIL"


def test_redirect_fails_when_old_page_is_built_but_target_is_missing(tmp_path: Path) -> None:
    src, dist = make_site(
        tmp_path,
        redirects={"/work/old-a/": "/work/carbon/#nope", "/work/old-b/": "/work/gone/"},
    )
    for old_slug, target in (("old-a", "/work/carbon/#nope"), ("old-b", "/work/gone/")):
        old = dist / "work" / old_slug
        old.mkdir()
        old.joinpath("index.html").write_text(
            f'<meta http-equiv="refresh" content="0;url={target}">'
        )
    findings = check_links.run_all(src, dist)
    by_subject = {finding.subject: finding for finding in findings if finding.check == "redirect"}
    fragment_case = by_subject["/work/old-a/ -> /work/carbon/#nope"]
    page_case = by_subject["/work/old-b/ -> /work/gone/"]
    assert fragment_case.status == "FAIL" and "no id" in fragment_case.detail
    assert page_case.status == "FAIL" and "no such page" in page_case.detail


def test_internal_href_to_a_missing_page_fails_and_a_file_passes(tmp_path: Path) -> None:
    src, dist = make_site(
        tmp_path,
        extra_html='<a href="/work/gone/">g</a><a href="/posters/x.pdf">p</a><a href="https://x.org/">e</a>',
    )
    (dist / "posters").mkdir()
    (dist / "posters" / "x.pdf").write_bytes(b"%PDF")
    findings = check_links.run_all(src, dist)
    failed = [
        finding.subject
        for finding in findings
        if finding.check == "href" and finding.status == "FAIL"
    ]
    assert failed == ["/work/carbon/ -> /work/gone/"]


def test_every_theme_section_must_end_with_the_return_line(tmp_path: Path) -> None:
    body = THEME_BODY.replace(
        "## CHO-GAP\n\n[Contents ↑](#contents)\n", "## CHO-GAP\n\nSome prose.\n"
    )
    src, dist = make_site(tmp_path, body=body)
    findings = check_links.run_all(src, dist)
    assert statuses(findings, "contents-return") == ["OK", "FAIL"]


def test_figure_markers_must_run_from_one(tmp_path: Path) -> None:
    good = (
        '<span class="label plate__fig">Fig. 1</span><span class="label plate__fig">Fig. 2</span>'
    )
    src, dist = make_site(tmp_path, extra_html=good)
    assert statuses(check_links.run_all(src, dist), "figures") == ["OK"]
    bad = '<span class="label plate__fig">Fig. 1</span><span class="label plate__fig">Fig. 3</span>'
    src, dist = make_site(tmp_path / "bad", extra_html=bad)
    assert "FAIL" in statuses(check_links.run_all(src, dist), "figures")


def test_main_exit_code_reflects_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src, dist = make_site(tmp_path)
    assert check_links.main(["--src", str(src), "--dist", str(dist)]) == 0
    src, dist = make_site(tmp_path / "bad", page_ids=("contents",))
    assert check_links.main(["--src", str(src), "--dist", str(dist)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_attribute_names_ending_in_id_are_not_ids(tmp_path: Path) -> None:
    src, dist = make_site(
        tmp_path, page_ids=("contents", "cho-gap"), extra_html='<div data-id="gap-20"></div>'
    )
    findings = check_links.run_all(src, dist)
    assert statuses(findings, "anchor") == ["FAIL"]
