"""Read one sheet of a Source Data workbook with the standard library.

The project environment has neither openpyxl nor pandas, and the sheets of a journal's
Source Data hold a header row over plain numbers. An xlsx is a zip of XML, so this reads that
much of it: the sheet's name-to-file mapping, the shared strings and each cell's value.
Formulas, styles and merged cells are not read. Header cells may be shared or inline strings.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RELATIONSHIP = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def read_sheet(workbook: Path, sheet: str) -> tuple[list[str], np.ndarray]:
    """The header row and the numeric body of one sheet.

    Args:
        workbook: Path to the .xlsx file.
        sheet: The sheet's name, exactly as the workbook spells it (e.g. "Fig.4b").

    Returns:
        (the first row's cells as strings, the remaining rows as a float array with one
        column per header cell).

    Raises:
        KeyError: If the workbook has no sheet of that name.
        ValueError: If a body cell is not a number or is missing.
    """
    with zipfile.ZipFile(workbook) as archive:
        shared = (
            [
                "".join(text.text or "" for text in item.iter(f"{MAIN}t"))
                for item in ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            ]
            if "xl/sharedStrings.xml" in archive.namelist()
            else []
        )
        sheets = ElementTree.fromstring(archive.read("xl/workbook.xml")).find(f"{MAIN}sheets")
        targets = {
            relationship.get("Id"): relationship.get("Target")
            for relationship in ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        }
        by_name = {entry.get("name"): targets[entry.get(RELATIONSHIP)] for entry in sheets}
        root = ElementTree.fromstring(archive.read("xl/" + by_name[sheet].removeprefix("/xl/")))

    rows: dict[int, dict[int, str]] = {}
    for cell in root.iter(f"{MAIN}c"):
        reference = re.fullmatch(r"([A-Z]+)(\d+)", cell.get("r"))
        column = 0
        for letter in reference.group(1):
            column = column * 26 + ord(letter) - ord("A") + 1
        value = cell.find(f"{MAIN}v")
        if cell.get("t") == "inlineStr":
            text = "".join(piece.text or "" for piece in cell.iter(f"{MAIN}t"))
        elif value is not None:
            text = shared[int(value.text)] if cell.get("t") == "s" else value.text
        else:
            continue
        rows.setdefault(int(reference.group(2)), {})[column - 1] = text

    header_row, *body_rows = (rows[number] for number in sorted(rows))
    width = len(header_row)
    try:
        body = np.array([[float(row[column]) for column in range(width)] for row in body_rows])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{workbook} {sheet}: a body cell is missing or not a number") from error
    return [header_row[column] for column in range(width)], body
