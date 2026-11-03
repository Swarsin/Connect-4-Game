// Capture UI screenshots for visual QA.
//   node scripts/screenshots.mjs [url] [outDir]
// Default: the direction mockups page -> ../docs/figures
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const url = process.argv[2] ?? pathToFileURL(resolve(here, "../mockups/directions.html")).href;
const out = resolve(process.argv[3] ?? resolve(here, "../../docs/figures"));
mkdirSync(out, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 2 });
await page.goto(url, { waitUntil: "networkidle" });
await page.evaluate(() => document.fonts.ready);

for (const dir of ["a", "b", "c"]) {
  const el = page.locator(`section[data-dir="${dir}"]`);
  await el.screenshot({ path: `${out}/mock-${dir}.png` });
}

await page.setViewportSize({ width: 390, height: 844 });
await page.reload({ waitUntil: "networkidle" });
await page.evaluate(() => document.fonts.ready);
for (const dir of ["a", "b", "c"]) {
  await page.locator(`section[data-dir="${dir}"]`).screenshot({ path: `${out}/mock-${dir}-mobile.png` });
}

await page.setViewportSize({ width: 1440, height: 1000 });
await page.screenshot({ path: `${out}/mock-all.png`, fullPage: true });
await browser.close();
console.log(`wrote screenshots to ${out}`);
