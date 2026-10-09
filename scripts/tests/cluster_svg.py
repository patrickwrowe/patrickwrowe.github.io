"""Run render_cluster.py's command line from a test, and read its drawings back."""

from __future__ import annotations

import re
import sys

import numpy as np
import render_cluster


def run_cli(monkeypatch, capsys, *argv: str) -> str:
    """Run `render_cluster.main()` with this command line.

    Args:
        monkeypatch: pytest's monkeypatch fixture, to set `sys.argv`.
        capsys: pytest's capsys fixture, to collect what the script prints.
        *argv: The arguments after the script's name, `--output` included.

    Returns:
        What the script printed to standard output.
    """
    monkeypatch.setattr(sys, "argv", ["render_cluster.py", *argv])
    render_cluster.main()
    return capsys.readouterr().out


def circle_centres(svg: str, species: str | None = None) -> np.ndarray:
    """Every drawn disc and ring in drawing order: (cx, -cy) in angstrom, the figure's frame.

    The figure is drawn with y down, so the sign of cy is flipped back: the result is in the
    frame the rotated structure is in, with y up and a single row's centre at y = 0.

    Args:
        svg: A render_cluster.py figure.
        species: Element symbol; only the circles painted in its `--species-*` token, or None
            for every circle.

    Returns:
        Array of shape (n_circles, 2), in angstrom.
    """
    token = rf"var\(--species-{species.lower()}\)" if species else ""
    return np.array(
        [
            [float(centre_x), -float(centre_y)]
            for centre_x, centre_y in re.findall(
                rf'<circle cx="([-\d.]+)" cy="([-\d.]+)"[^>]*{token}[^>]*/>', svg
            )
        ]
    ).reshape(-1, 2)


def panel_centres(svg: str) -> list[float]:
    """The x of each panel's centre in a labelled series, in angstrom, left to right.

    A label is centred on its panel, so its x is the panel's centre.
    """
    return [float(label_x) for label_x in re.findall(r'<text x="([-\d.]+)"', svg)]
