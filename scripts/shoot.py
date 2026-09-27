"""Screenshot the running site at the two widths CLAUDE.md requires.

    npm run preview &
    PLAYWRIGHT_BROWSERS_PATH=./.playwright uv run python scripts/shoot.py [outdir]

Definition of done item 2: visual changes are checked in a browser at 1280px and
390px. Screenshot it, don't reason about the CSS.
"""

import pathlib
import sys

from playwright.sync_api import Page, sync_playwright

BASE = "http://localhost:4321"
WIDTHS = {"desktop": 1280, "mobile": 390}
ROUTES = {
    "landing": "/",
    "work": "/work/",
    "work-entry": "/work/carbon-gap-20/",
    "writing": "/writing/",
    "post": "/writing/smiles-is-a-strange-language/",
    "cv": "/cv/",
    "about": "/about/",
    "404": "/404",
}


SCROLL_STEP_WAIT_MS = 250  # pause per viewport-height step, long enough to start a lazy fetch


def _scroll_through_lazy_figures(page: Page) -> None:
    """Walk the page top to bottom so `loading="lazy"` figures come into view and load.

    Steps in viewport-sized increments with a short wait at each, then scrolls back to
    the top (the recorded scroll position) and waits for the network to go idle so every
    triggered image fetch has finished before the screenshot.

    Args:
        page: The Playwright page to scroll.
    """
    page.evaluate(
        """
        async (stepWaitMs) => {
            const step = window.innerHeight;
            const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
            let scrolled = 0;
            const scrollHeight = document.documentElement.scrollHeight;
            while (scrolled < scrollHeight) {
                scrolled += step;
                window.scrollTo(0, scrolled);
                await sleep(stepWaitMs);
            }
            window.scrollTo(0, 0);
        }
        """,
        SCROLL_STEP_WAIT_MS,
    )
    page.wait_for_load_state("networkidle")


def main() -> None:
    """Screenshot ROUTES, or the routes given after the output directory, at both widths.

    Usage:
        python scripts/shoot.py [outdir] [route ...]

    Names for routes given on the command line come from the route itself, so
    ``/work/carbon/`` is saved as ``work-carbon-desktop.png``.
    """
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "screenshots")
    out.mkdir(parents=True, exist_ok=True)
    requested = sys.argv[2:]
    routes = {
        route.strip("/").replace("/", "-") or "landing": route for route in requested
    } or ROUTES

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for label, width in WIDTHS.items():
            page = browser.new_page(viewport={"width": width, "height": 900})
            for name, route in routes.items():
                page.goto(f"{BASE}{route}", wait_until="networkidle")
                page.wait_for_timeout(1400)  # let the plate scan-in settle
                _scroll_through_lazy_figures(page)
                path = out / f"{name}-{label}.png"
                page.screenshot(path=path, full_page=True)
                print(f"{path}  ({width}px)")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
