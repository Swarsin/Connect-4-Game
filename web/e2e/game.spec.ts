import { expect, test } from "@playwright/test";

test.describe("classic play", () => {
  test("loads, plays a move, the AI replies and undo works", async ({ page }) => {
    await page.goto("./");
    await expect(page.locator(".board")).toBeVisible();

    // bottom row, centre column (cells are rendered top row first)
    await page.locator(".cell").nth(38).click();
    // the classic bot replies in a few milliseconds, so only the settled state is observable
    await expect(page.locator(".history span")).toHaveCount(2, { timeout: 30_000 });
    await expect(page.locator(".status-line")).toContainText("Your move");
    await expect(page.locator(".eval-num")).not.toHaveText("—");

    await page.getByRole("button", { name: "Undo" }).click();
    await expect(page.locator(".history span")).toHaveCount(0);
  });

  test("a winning move ends the game and highlights four cells", async ({ page }) => {
    await page.goto("./#/play?m=121212&bot=minimax:d=2&mode=ai&human=1");
    await expect(page.locator(".board")).toBeVisible();

    // player 1 has three in column 1 and to move: bottom-left cell wins
    await page.locator(".cell").nth(35).click();
    await expect(page.locator(".banner strong")).toHaveText("You win");
    await expect(page.locator(".cell.win")).toHaveCount(4);
    await expect(page.getByRole("button", { name: "Play again" })).toBeVisible();
  });

  test("hot-seat mode alternates players", async ({ page }) => {
    await page.goto("./#/play?mode=local");
    await expect(page.locator(".status-line")).toContainText("Player 1 to move");
    await page.locator(".cell").nth(3).click();
    await expect(page.locator(".status-line")).toContainText("Player 2 to move");
    await page.locator(".cell").nth(4).click();
    await expect(page.locator(".status-line")).toContainText("Player 1 to move");
  });
});

test.describe("mobile", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test("board fits without horizontal scrolling", async ({ page }) => {
    await page.goto("./");
    await expect(page.locator(".board")).toBeVisible();
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - window.innerWidth,
    );
    expect(overflow).toBeLessThanOrEqual(1);
    await expect(page.locator(".side")).toBeVisible();
  });
});
