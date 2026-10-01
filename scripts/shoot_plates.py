"""Capture every figure plate that holds a panel grid or an inline chart, at five
viewport and pixel-density settings, with the layout numbers a design review needs.

    npm run dev -- --port 4321 &
    PLAYWRIGHT_BROWSERS_PATH=./.playwright uv run python scripts/shoot_plates.py [outdir] [route]

Writes <outdir>/<setting>-fig<n>.png per plate (n as the page numbers its figures) and
prints one JSON line per setting with, for each grid: its computed display, the track
count, its width and scroll width, the first thumbnail's rendered width, the srcset
variant the browser picked and the `sizes` it was given. The restructure spec's phone
rule (section 6.2: every panel at least 100 CSS px, every label at least 11 px) is
checked against these numbers, not reasoned about from the CSS. `shoot.py` takes the
full-page screenshots the definition of done asks for; this script is the close-up.
"""

import json
import pathlib
import sys

from playwright.sync_api import sync_playwright

from shoot import BASE

# (name, viewport width in CSS px, device pixel ratio): a 1x monitor, a 2x laptop, the
# plate's breakout threshold, the 700-1000 px band where grids may scroll, and a phone.
SETTINGS = [
    ("w1920-dpr1", 1920, 1),
    ("w1440-dpr2", 1440, 2),
    ("w1100-dpr1", 1100, 1),
    ("w900-dpr1", 900, 1),
    ("w390-dpr3", 390, 3),
]
GRID_METRICS = """() => Array.from(document.querySelectorAll('.grid')).map((grid) => {
  const img = grid.querySelector('img');
  const style = getComputedStyle(grid);
  return {
    display: style.display,
    tracks: style.gridTemplateColumns.split(' ').length,
    gridWidth: grid.clientWidth,
    scrollWidth: grid.scrollWidth,
    thumbWidth: img ? img.clientWidth : null,
    variant: img ? (img.currentSrc || '').match(/w=(\\d+)/)?.[1] : null,
    sizes: img ? img.sizes : null,
  };
})"""


def main() -> None:
    """Capture the plates of one route at every setting into `outdir`."""
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "screenshots/plates")
    out.mkdir(parents=True, exist_ok=True)
    route = sys.argv[2] if len(sys.argv) > 2 else "/work/carbon/"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for name, width_px, device_scale in SETTINGS:
            context = browser.new_context(
                viewport={"width": width_px, "height": 1000}, device_scale_factor=device_scale
            )
            page = context.new_page()
            page.goto(f"{BASE}{route}", wait_until="networkidle")
            # Scroll pass so every lazy figure has loaded before the close-ups.
            height_px = page.evaluate("document.body.scrollHeight")
            for scroll_px in range(0, height_px, 800):
                page.evaluate(f"window.scrollTo(0, {scroll_px})")
                page.wait_for_timeout(120)
            page.wait_for_load_state("networkidle")
            plates = page.locator("figure.plate")
            for index in range(plates.count()):
                plate = plates.nth(index)
                if plate.locator(".grid, .chart").count() == 0:
                    continue
                plate.scroll_into_view_if_needed()
                page.wait_for_timeout(300)
                plate.screenshot(path=str(out / f"{name}-fig{index + 1}.png"))
            print(name, json.dumps(page.evaluate(GRID_METRICS)))
            context.close()
        browser.close()


if __name__ == "__main__":
    main()
