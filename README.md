# patrickwrowe.github.io

Patrick Rowe's portfolio site: research articles on machine learning for molecules,
materials and proteins, projects, writing and the publication record. Astro 6, static, zero
JavaScript by default, built and deployed to GitHub Pages by `.github/workflows/deploy.yml`
on every push to `main`. Python owns every data pipeline: each chart and molecular rendering
is made by a committed script from committed inputs.

## Layout

```
src/                  pages, layouts, components, styles/tokens.css, content/work (articles), data/
public/               served as is: posters, video, demos (fetched on click only)
scripts/              check_links.py (the post-build link check), shoot*.py (screenshots), gen_network.py
scripts/figures/      figure scripts: SVG charts, render_cluster.py, the Blender driver render_box_grid.py
scripts/figures/data/ every figure's inputs, each directory with its provenance.json
scripts/tests/        the pytest suite; tests/ the node tests of src/lib; archive/ the retired pages
```

## Setting up and running

`./setup.sh` installs the Node dependencies into `node_modules` and the Python ones into
`.venv`. Every command (dev server, build, link check, tests, screenshots, figure
regeneration) is in the **Commands** block of [`CLAUDE.md`](CLAUDE.md), with the rules the
site is built to.

## Environment variables

None is needed to build the site or run the checks. Tool locations are never written into a
tracked file, so the commands that need one take it from the environment:

| Variable | Needed by |
|---|---|
| `MOLRENDER_PYTHON` | `scripts/figures/render_box_grid.py`, the Blender panels: the Python of the environment that has molrender |
| `FFMPEG` | `scripts/figures/cut_graphitisation_clip.sh`, the graphitisation clip |
| `PAPERS_DIR` | optional: the directory of the papers, so the tests also check the figures' numbers against them (skipped without it) |

## The six checks

A change is done when all six exit 0:

```bash
npm run build                          # also the type and content-schema check
.venv/bin/python scripts/check_links.py
npm test
.venv/bin/pytest scripts/tests -q
.venv/bin/ruff check scripts
.venv/bin/ruff format --check scripts
```
