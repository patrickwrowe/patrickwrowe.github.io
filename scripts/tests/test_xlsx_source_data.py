"""Tests for scripts/tests/xlsx_source_data.py, on workbooks built here (a few cells each)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from xlsx_source_data import read_sheet

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
RELATIONSHIPS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
SHARED_HEADER = '<c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c>'
INLINE_HEADER = (
    '<c r="A1" t="inlineStr"><is><t>Cos</t></is></c><c r="B1" t="inlineStr"><is><t>P</t></is></c>'
)
BODY = '<c r="A2"><v>0.5</v></c><c r="B2"><v>1.25e-3</v></c>'


def workbook(path: Path, row_one: str, row_two: str, shared: bool) -> Path:
    """A one-sheet workbook ("Fig. 1") with two rows, with or without shared strings."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{MAIN}" xmlns:r="{RELATIONSHIPS}">'
            '<sheets><sheet name="Fig. 1" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        if shared:
            archive.writestr(
                "xl/sharedStrings.xml",
                f'<sst xmlns="{MAIN}"><si><t>Cos</t></si><si><r><t>P</t></r></si></sst>',
            )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{MAIN}"><sheetData><row r="1">{row_one}</row>'
            f'<row r="2">{row_two}</row></sheetData></worksheet>',
        )
    return path


@pytest.mark.parametrize(
    ("header", "shared"), [(SHARED_HEADER, True), (INLINE_HEADER, False)], ids=["shared", "inline"]
)
def test_a_header_of_shared_or_inline_strings_is_read_with_its_numbers(tmp_path, header, shared):
    names, body = read_sheet(workbook(tmp_path / "w.xlsx", header, BODY, shared), "Fig. 1")
    assert names == ["Cos", "P"]
    assert body == pytest.approx(np.array([[0.5, 1.25e-3]]))


def test_a_missing_body_cell_is_refused_naming_the_workbook_and_sheet(tmp_path):
    path = workbook(tmp_path / "w.xlsx", INLINE_HEADER, '<c r="A2"><v>0.5</v></c>', False)
    with pytest.raises(ValueError, match=r"w\.xlsx Fig\. 1: a body cell is missing"):
        read_sheet(path, "Fig. 1")
