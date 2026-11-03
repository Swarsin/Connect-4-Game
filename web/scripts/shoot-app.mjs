// Ad-hoc app screenshots for visual QA (dev server must be running).
//   node scripts/shoot-app.mjs http://127.0.0.1:5173/Connect-4-Game/
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

const url = process.argv[2] ?? "http://127.0.0.1:5173/Connect-4-Game/";
const out = resolve(process.argv[3] ?? "../docs/figures");
mkdirSync(out, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 2 });
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(String(e)));

await page.goto(url, { waitUntil: "domcontentloaded" });
await page.waitForSelector(".board");
await page.evaluate(() => document.fonts.ready);
await page.waitForTimeout(600);
await page.screenshot({ path: `${out}/app-empty.png`, fullPage: true });

// play column 4 and wait for the AI to reply
await page.locator(".cell").nth(3).click();
await page.waitForTimeout(120);
await page.screenshot({ path: `${out}/app-thinking.png`, fullPage: true });
await page.waitForFunction(() => document.querySelectorAll(".history span").length >= 2, null, {
  timeout: 20_000,
});
await page.waitForTimeout(600);
await page.screenshot({ path: `${out}/app-played.png`, fullPage: true });

// dark theme
await page.locator(".icon-btn").first().click();
await page.waitForTimeout(300);
await page.screenshot({ path: `${out}/app-dark.png`, fullPage: true });
await page.locator(".icon-btn").first().click();

// mobile
await page.setViewportSize({ width: 390, height: 844 });
await page.reload({ waitUntil: "domcontentloaded" });
await page.waitForSelector(".board");
await page.evaluate(() => document.fonts.ready);
await page.locator(".cell").nth(3).click();
await page.waitForTimeout(1500);
await page.screenshot({ path: `${out}/app-mobile.png`, fullPage: true });

// AlphaZero level
await page.setViewportSize({ width: 1440, height: 1000 });
await page.goto(`${url}#/play?bot=az:hard&mode=ai&human=1`, { waitUntil: "domcontentloaded" });
await page.waitForSelector(".board");
await page.waitForFunction(() => document.querySelector(".eval-num")?.textContent !== "—", null, {
  timeout: 20_000,
});
await page.evaluate(() => document.fonts.ready);
await page.locator(".cell").nth(3).click();
await page.waitForFunction(() => document.querySelectorAll(".history span").length >= 2, null, {
  timeout: 30_000,
});
await page.waitForTimeout(800);
await page.screenshot({ path: `${out}/app-az.png`, fullPage: true });

// winning position (player 1 has three in column 1)
await page.setViewportSize({ width: 1440, height: 1000 });
await page.goto(`${url}#/play?m=121212&bot=minimax:d=2&mode=ai&human=1`, { waitUntil: "domcontentloaded" });
await page.waitForSelector(".board");
await page.evaluate(() => document.fonts.ready);
await page.locator(".cell").nth(35).click();
await page.waitForTimeout(500);
await page.screenshot({ path: `${out}/app-win.png`, fullPage: true });

await browser.close();
console.log(errors.length ? `console errors:\n${errors.join("\n")}` : "no console errors");
