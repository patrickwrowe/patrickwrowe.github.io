"""Screenshot the running site at the two widths CLAUDE.md requires.

    npm run preview &
    PLAYWRIGHT_BROWSERS_PATH=./.playwright uv run python scripts/shoot.py \
        [outdir] [route ...] [--base-url http://localhost:4321]

Definition of done item 2: visual changes are checked in a browser at 1280px and
390px. Screenshot it, don't reason about the CSS.
"""

import argparse
import pathlib

from playwright.sync_api import Page, sync_playwright

DEFAULT_BASE_URL = "http://localhost:4321"
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


def scroll_through_lazy_figures(page: Page) -> None:
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


def main(argv: list[str] | None = None) -> None:
    """Screenshot ROUTES, or the routes given after the output directory, at both widths.

    Names for routes given on the command line come from the route itself, so
    ``/work/carbon/`` is saved as ``work-carbon-desktop.png``.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outdir", nargs="?", type=pathlib.Path, default=pathlib.Path("screenshots"))
    parser.add_argument("routes", nargs="*", help="routes to shoot (default: every page type)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="the running site")
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    routes = {
        route.strip("/").replace("/", "-") or "landing": route for route in args.routes
    } or ROUTES

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for label, width in WIDTHS.items():
            page = browser.new_page(viewport={"width": width, "height": 900})
            for name, route in routes.items():
                page.goto(f"{args.base_url}{route}", wait_until="networkidle")
                page.wait_for_timeout(1400)  # let the plate scan-in settle
                scroll_through_lazy_figures(page)
                path = args.outdir / f"{name}-{label}.png"
                page.screenshot(path=path, full_page=True)
                print(f"{path}  ({width}px)")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
