# Web app

Vite + React + TypeScript. Playable against **classic bots** (Minimax / MCTS) and **AlphaZero
levels** (int8 ONNX in a Web Worker with PUCT), with live evaluation, undo, shareable positions,
dark mode and two-player mode.

## Run it

```bash
npm install
npm run dev        # http://localhost:5173/Connect-4-Game/
```

Screenshots: `../docs/figures/app-*.png`.

## Look at the UI directions

Open `mockups/directions.html` directly in a browser (no build or server needed):

```
C:\Users\swarsin\Connect-4-Game\web\mockups\directions.html
```

Same position, same sidebar, three token sets (A modern toy, B editorial, C arcade). The chosen
one becomes the token source for the real app. Rendered PNGs: `../docs/figures/mock-*.png`.

## Screenshots (visual QA)

```bash
npm install
npx playwright install chromium
node scripts/screenshots.mjs          # -> ../docs/figures/mock-{a,b,c}[-mobile].png
```

On WSL, headless Chromium needs `libnspr4`/`libnss3`. Without root, fetch them locally:

```bash
cd /tmp && apt-get download libnspr4 libnss3
mkdir -p ~/.local/chromedeps
for f in libnspr4_*.deb libnss3_*.deb; do dpkg -x "$f" ~/.local/chromedeps; done
export LD_LIBRARY_PATH=~/.local/chromedeps/usr/lib/x86_64-linux-gnu
```

## Models

`public/models/{beginner,easy,master}.onnx` are static-int8 exports (3.05 MB each) with
`levels.json` carrying the measured Elo and the per-level simulation budget. Do not edit the Elo by
hand — regenerate with `python -m c4.eval.az_report`.

ONNX Runtime Web is loaded lazily, only when an AlphaZero level is chosen. Its 14.8 MB wasm runtime
is copied to `public/ort` by `scripts/copy-ort.mjs`; `vite.config.ts` sets the
`onnxruntime-web-use-extern-wasm` condition so Rollup does **not** inline a second copy. Removing
that condition silently doubles the bundle and bypasses `public/ort`.

Measured on a desktop Chromium (wasm, 1 thread): Medium (network only) ~60 ms per move, Hard
(200 simulations) ~1.2 s, Master (800 simulations) ~3.2 s. The first AlphaZero analysis also loads
the model (~0.7 s) before it appears.

## Deploy

Pushes to `main` run `.github/workflows/deploy.yml` and publish `web/dist` to GitHub Pages.
The site is served from `/Connect-4-Game/` (Vite `base`); set `VITE_BASE=/` for a custom domain
or a user/org page. Pages must be configured once with **Source: GitHub Actions**.

End-to-end tests run against a production build through `vite preview`:

```bash
npm run build
npx playwright test        # or: npm run test:e2e
```

CI runs the same suite plus lint, typecheck and the parity tests on every pull request.

## Commands

```bash
npm run dev        # copies ORT wasm to public/ort, then Vite
npm run build      # tsc + vite build
npm test           # Vitest (engine parity, search parity)
npm run test:e2e   # Playwright
```

## Conventions

* Search runs in a Web Worker; the main thread never blocks.
* Neural input/output layout and move notation are fixed in `../docs/conventions.md`.
* Fixtures: `../data/fixtures/engine_cases.json`, `net_cases.json`, `search_cases.json`.
* Vite `base` is `/Connect-4-Game/` for GitHub Pages; override with `VITE_BASE`.
