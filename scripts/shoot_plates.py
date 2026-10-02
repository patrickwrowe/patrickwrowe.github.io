"""Capture every figure plate that holds a panel grid or an inline chart, at five
viewport and pixel-density settings, with the layout numbers a design review needs.

    npm run dev -- --port 4321 &
    PLAYWRIGHT_BROWSERS_PATH=./.playwright uv run python scripts/shoot_plates.py \
        [outdir] [route] [--base-url http://localhost:4321]

Writes <outdir>/<setting>-fig<n>.png per plate (n as the page numbers its figures) and
prints one JSON line per setting with, for each grid: whether it is shown (a grid inside
a closed <details> is not), its computed display, the track count, its width and scroll
width, the first thumbnail's rendered width, the srcset variant the browser picked and
the `sizes` it was given. The restructure spec's phone rule (section 6.2: every panel at
least 100 CSS px, every label at least 11 px) is checked against these numbers, not
reasoned about from the CSS, and its panel half is enforced: the script exits 1 if the
first thumbnail of any shown grid is under PHONE_FLOOR_PX at the phone setting. `shoot.py`
takes the full-page screenshots the definition of done asks for; this is the close-up.
"""

import argparse
import json
import pathlib
import sys

from playwright.sync_api import sync_playwright
from shoot import DEFAULT_BASE_URL, scroll_through_lazy_figures

# (name, viewport width in CSS px, device pixel ratio): a 1x monitor, a 2x laptop, the
# plate's breakout threshold, the 700-1000 px band where grids may scroll, and a phone.
SETTINGS = [
    ("w1920-dpr1", 1920, 1),
    ("w1440-dpr2", 1440, 2),
    ("w1100-dpr1", 1100, 1),
    ("w900-dpr1", 900, 1),
    ("w390-dpr3", 390, 3),
]
PHONE_SETTING = "w390-dpr3"
PHONE_FLOOR_PX = 100
GRID_METRICS = """() => Array.from(document.querySelectorAll('.grid')).map((grid) => {
  const img = grid.querySelector('img');
  const style = getComputedStyle(grid);
  return {
    shown: grid.checkVisibility(),
    display: style.display,
    tracks: style.gridTemplateColumns.split(' ').length,
    gridWidth: grid.clientWidth,
    scrollWidth: grid.scrollWidth,
    thumbWidth: img ? img.clientWidth : null,
    variant: img ? (img.currentSrc || '').match(/w=(\\d+)/)?.[1] : null,
    sizes: img ? img.sizes : null,
  };
})"""


def main(argv: list[str] | None = None) -> int:
    """Capture the plates of one route at every setting into `outdir`.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.

    Returns:
        Process exit status: 1 if a shown grid's first thumbnail is under PHONE_FLOOR_PX
        at PHONE_SETTING, else 0.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "outdir", nargs="?", type=pathlib.Path, default=pathlib.Path("screenshots/plates")
    )
    parser.add_argument("route", nargs="?", default="/work/carbon/")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="the running site")
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    too_narrow: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for name, width_px, device_scale in SETTINGS:
            context = browser.new_context(
                viewport={"width": width_px, "height": 1000}, device_scale_factor=device_scale
            )
            page = context.new_page()
            page.goto(f"{args.base_url}{args.route}", wait_until="networkidle")
            scroll_through_lazy_figures(page)  # every lazy figure loads before the close-ups
            plates = page.locator("figure.plate")
            for index in range(plates.count()):
                plate = plates.nth(index)
                if plate.locator(".grid, .chart").count() == 0:
                    continue
                plate.scroll_into_view_if_needed()
                page.wait_for_timeout(300)
                plate.screenshot(path=str(args.outdir / f"{name}-fig{index + 1}.png"))
            grids = page.evaluate(GRID_METRICS)
            print(name, json.dumps(grids))
            if name == PHONE_SETTING:
                too_narrow += [
                    f"grid {index + 1}: first thumbnail {grid['thumbWidth']} px"
                    for index, grid in enumerate(grids)
                    if grid["shown"] and (grid["thumbWidth"] or 0) < PHONE_FLOOR_PX
                ]
            context.close()
        browser.close()
    if too_narrow:
        print(
            f"phone rule broken at {PHONE_SETTING} (under {PHONE_FLOOR_PX} px): {too_narrow}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
