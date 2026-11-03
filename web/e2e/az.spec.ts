import { expect, test } from "@playwright/test";

test.describe("AlphaZero in the browser", () => {
  test("hard level loads the model, values the position and replies", async ({ page }) => {
    test.slow(); // wasm + 3 MB model + 200 simulations
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));

    await page.goto("./#/play?bot=az:hard&mode=ai&human=1");
    await expect(page.locator(".board")).toBeVisible();

    // the calibrated value head replaces the "—" placeholder once the model is ready
    await expect(page.locator(".eval-num")).not.toHaveText("—", { timeout: 45_000 });
    await expect(page.locator(".card .small").first()).toContainText("network value");

    await page.locator(".cell").nth(38).click();
    await expect(page.locator(".history span")).toHaveCount(2, { timeout: 45_000 });
    await expect(page.locator(".status-line")).toContainText("Your move");

    expect(errors).toEqual([]);
  });

  test("the level picker lists AlphaZero and switching restarts the game", async ({ page }) => {
    await page.goto("./");
    await expect(page.locator("#bot optgroup")).toHaveCount(2);
    await page.selectOption("#bot", "az:medium");
    await expect(page.locator(".pill")).toContainText("ALPHAZERO", { ignoreCase: true });
    await expect(page.locator(".board")).toBeVisible();
  });
});
