#!/usr/bin/env bash
#
# Set up the local toolchain for this site. Safe to re-run — everything here is
# idempotent, and nothing is installed outside this directory.
#
#   ./setup.sh
#
# Node deps land in ./node_modules, Python deps in ./.venv. Both are gitignored.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REQUIRED_NODE_MAJOR=22
REQUIRED_NODE_MINOR=12

info() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
fail() { printf '\n\033[31mError: %s\033[0m\n' "$1" >&2; exit 1; }

# --- Node -------------------------------------------------------------------
info "Checking Node"

command -v node >/dev/null 2>&1 || fail \
  "node not found. Astro 6 needs Node >= ${REQUIRED_NODE_MAJOR}.${REQUIRED_NODE_MINOR}.0 — install it (nvm install 22) and re-run."

node_version=$(node --version | sed 's/^v//')
node_major=${node_version%%.*}
node_rest=${node_version#*.}
node_minor=${node_rest%%.*}

if [ "$node_major" -lt "$REQUIRED_NODE_MAJOR" ] ||
   { [ "$node_major" -eq "$REQUIRED_NODE_MAJOR" ] && [ "$node_minor" -lt "$REQUIRED_NODE_MINOR" ]; }; then
  fail "Node ${node_version} is too old. Astro 6 needs >= ${REQUIRED_NODE_MAJOR}.${REQUIRED_NODE_MINOR}.0."
fi
echo "node ${node_version}, npm $(npm --version)"

# --- JS dependencies --------------------------------------------------------
# npm ci is the reproducible path but requires the lockfile to already agree with
# package.json; npm install is what generates/updates it. Prefer ci, fall back.
info "Installing JS dependencies"
if [ -f package-lock.json ] && npm ci --no-audit --fund=false 2>/dev/null; then
  echo "installed from package-lock.json"
else
  npm install --no-audit --fund=false
fi

# --- Vendored KaTeX ---------------------------------------------------------
# Copied into public/ rather than imported as a module: an import would be merged
# into the shared stylesheet and ship on every route, but spec 02 §3.2 wants it only
# on pages with `math: true`. Committed, following the same convention spec 03 §3.3
# sets for the RDKit wasm. Only woff2 is copied — the woff/ttf fallbacks are ~900KB
# and nothing that can run this site needs them.
info "Refreshing vendored KaTeX"
mkdir -p public/vendor/katex/fonts
cp node_modules/katex/dist/katex.min.css public/vendor/katex/
cp node_modules/katex/dist/fonts/*.woff2 public/vendor/katex/fonts/
echo "public/vendor/katex ($(du -sh public/vendor/katex | cut -f1))"

# --- uv ---------------------------------------------------------------------
info "Checking uv"
command -v uv >/dev/null 2>&1 || fail \
  "uv not found. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
echo "$(uv --version)"

# --- Python environment -----------------------------------------------------
# uv sync creates ./.venv, fetches a matching interpreter if the system one is too
# old, resolves against uv.lock, and prunes anything no longer declared.
info "Syncing Python environment"
uv sync

# --- Verify -----------------------------------------------------------------
info "Verifying"
npx astro --version
uv run python -c "import numpy, bibtexparser, matplotlib, yaml; print('python pipeline imports OK')"

cat <<'EOF'

Setup complete.

  npm run dev       localhost:4321
  npm run build     type/schema check + production build
  npm run preview   serve the built output

  uv run python scripts/gen_network.py    regenerate the hero network
                                          (writes network.json to the CWD)

Not installed on purpose: torch, onnxruntime, RDKit. Those belong to the demo
tiers (spec 03, Phases 4-5) and are several GB. Add them to pyproject.toml when
scripts/export_onnx.py actually exists.
EOF
