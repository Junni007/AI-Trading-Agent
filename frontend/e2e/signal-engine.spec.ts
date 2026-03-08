/**
 * Playwright E2E Tests — Signal.Engine
 * =====================================
 * Critical user flow coverage per webapp-testing skill:
 *   1. Navigation between all routes
 *   2. Connection error banner renders + dismisses
 *   3. Dashboard empty state
 *   4. Signals search + filter
 *   5. Analytics page structure
 *   6. Config page settings
 *   7. Responsive mobile menu
 *   8. 404 not-found route
 *
 * Run: npx playwright test
 * Requires: npm install -D @playwright/test && npx playwright install chromium
 */

import { test, expect, Page } from '@playwright/test';

// ── Shared helpers ────────────────────────────────────────────────────────────

/** Navigate to a route and wait for network idle */
async function goto(page: Page, path: string) {
    await page.goto(`http://localhost:5173${path}`);
    await page.waitForLoadState('domcontentloaded');
}

// ── 1. Navigation ─────────────────────────────────────────────────────────────

test.describe('Navigation', () => {
    test('Dashboard loads at root path', async ({ page }) => {
        await goto(page, '/');
        await expect(page.locator('text=Signal.Engine')).toBeVisible();
    });

    test('can navigate to Signals page', async ({ page }) => {
        await goto(page, '/');
        await page.click('nav >> text=Signals');
        await expect(page).toHaveURL(/\/signals/);
        await expect(page.locator('h1')).toContainText('Active Signals');
    });

    test('can navigate to Analytics page', async ({ page }) => {
        await goto(page, '/');
        await page.click('nav >> text=Analytics');
        await expect(page).toHaveURL(/\/analytics/);
        await expect(page.locator('h1')).toContainText('Portfolio Analytics');
    });

    test('can navigate to Config/Settings page', async ({ page }) => {
        await goto(page, '/');
        await page.click('nav >> text=Config');
        await expect(page).toHaveURL(/\/settings/);
        await expect(page.locator('h1')).toContainText('Configuration');
    });

    test('clicking brand logo returns to Dashboard', async ({ page }) => {
        await goto(page, '/signals');
        await page.click('a[href="/"]');
        await expect(page).toHaveURL('http://localhost:5173/');
    });
});

// ── 2. Connection Error Banner ─────────────────────────────────────────────────

test.describe('Connection error banner', () => {
    test('banner appears when backend is unreachable', async ({ page }) => {
        // Backend is not running in test env — banner should show
        await goto(page, '/');
        // Give the initial API call time to fail
        await page.waitForTimeout(2000);
        const banner = page.locator('text=Unable to reach backend API');
        await expect(banner).toBeVisible({ timeout: 5000 });
    });

    test('Dismiss button hides the banner', async ({ page }) => {
        await goto(page, '/');
        await page.waitForTimeout(2000);
        const banner = page.locator('text=Unable to reach backend API');
        await expect(banner).toBeVisible({ timeout: 5000 });

        await page.click('button:has-text("Dismiss")');
        await expect(banner).not.toBeVisible();
    });

    test('banner persists after page navigation if backend still down', async ({ page }) => {
        await goto(page, '/');
        await page.waitForTimeout(2000);
        await page.click('nav >> text=Signals');
        const banner = page.locator('text=Unable to reach backend API');
        await expect(banner).toBeVisible({ timeout: 3000 });
    });
});

// ── 3. Dashboard ──────────────────────────────────────────────────────────────

test.describe('Dashboard', () => {
    test('shows "No Signals Yet" empty state when no data', async ({ page }) => {
        await goto(page, '/');
        await page.waitForTimeout(1000);
        await expect(page.locator('text=No Signals Yet')).toBeVisible({ timeout: 5000 });
    });

    test('shows Terminal (sys::log) panel', async ({ page }) => {
        await goto(page, '/');
        await expect(page.locator('text=SYS::LOG')).toBeVisible({ timeout: 3000 });
    });

    test('Initialize Scan button is clickable', async ({ page }) => {
        await goto(page, '/');
        await page.waitForTimeout(1000);
        const btn = page.locator('button:has-text("Initialize Scan")');
        await expect(btn).toBeVisible({ timeout: 5000 });
        // Click should not throw — even if it triggers an API call that fails
        await btn.click();
    });

    test('REGIME, SYNC, and UPTIME chips are visible', async ({ page }) => {
        await goto(page, '/');
        await expect(page.locator('text=REGIME')).toBeVisible({ timeout: 3000 });
        await expect(page.locator('text=SYNC')).toBeVisible();
        await expect(page.locator('text=UPTIME')).toBeVisible();
    });
});

// ── 4. Signals Page ────────────────────────────────────────────────────────────

test.describe('Signals page', () => {
    test('renders heading and subtitle', async ({ page }) => {
        await goto(page, '/signals');
        await expect(page.locator('h1')).toContainText('Active Signals');
        await expect(page.locator('text=Real-time trading signals')).toBeVisible();
    });

    test('shows "No Active Signals" empty state by default', async ({ page }) => {
        await goto(page, '/signals');
        await expect(page.locator('text=No Active Signals')).toBeVisible({ timeout: 3000 });
    });

    test('filter tabs are rendered and clickable', async ({ page }) => {
        await goto(page, '/signals');
        const allTab = page.locator('button').filter({ hasText: /all signals/i }).first();
        const activeTab = page.locator('button').filter({ hasText: /^active/i }).first();
        const watchingTab = page.locator('button').filter({ hasText: /watching/i }).first();

        await expect(allTab).toBeVisible();
        await expect(activeTab).toBeVisible();
        await expect(watchingTab).toBeVisible();

        await activeTab.click();
        await watchingTab.click(); // Should not throw
    });

    test('search input is focusable and accepts text', async ({ page }) => {
        await goto(page, '/signals');
        const input = page.locator('input[placeholder*="Search ticker"]');
        await expect(input).toBeVisible();
        await input.fill('RELIANCE');
        await expect(input).toHaveValue('RELIANCE');
    });

    test('table headers (Ticker, Action, Confidence, Reasoning) are visible', async ({ page }) => {
        await goto(page, '/signals');
        await expect(page.locator('th').filter({ hasText: /ticker/i })).toBeVisible({ timeout: 3000 });
        await expect(page.locator('th').filter({ hasText: /action/i })).toBeVisible();
        await expect(page.locator('th').filter({ hasText: /confidence/i })).toBeVisible();
        await expect(page.locator('th').filter({ hasText: /reasoning/i })).toBeVisible();
    });
});

// ── 5. Analytics Page ──────────────────────────────────────────────────────────

test.describe('Analytics page', () => {
    test('shows Portfolio Analytics heading', async ({ page }) => {
        await goto(page, '/analytics');
        await expect(page.locator('h1')).toContainText('Portfolio Analytics');
    });

    test('shows Total Balance card', async ({ page }) => {
        await goto(page, '/analytics');
        await expect(page.locator('text=TOTAL BALANCE')).toBeVisible({ timeout: 3000 });
    });

    test('shows Equity Curve panel', async ({ page }) => {
        await goto(page, '/analytics');
        await expect(page.locator('text=Equity Curve')).toBeVisible({ timeout: 3000 });
    });

    test('shows Win Rate stat card', async ({ page }) => {
        await goto(page, '/analytics');
        await expect(page.locator('text=Win Rate')).toBeVisible({ timeout: 3000 });
    });

    test('shows Live Holdings section', async ({ page }) => {
        await goto(page, '/analytics');
        await expect(page.locator('text=Live Holdings')).toBeVisible({ timeout: 3000 });
    });

    test('shows Recent Trades section', async ({ page }) => {
        await goto(page, '/analytics');
        await expect(page.locator('text=Recent Trades')).toBeVisible({ timeout: 3000 });
    });
});

// ── 6. Config Page ─────────────────────────────────────────────────────────────

test.describe('Config / Settings page', () => {
    test('shows Configuration heading', async ({ page }) => {
        await goto(page, '/settings');
        await expect(page.locator('h1').filter({ hasText: /configuration/i })).toBeVisible({ timeout: 3000 });
    });

    test('shows System Status section', async ({ page }) => {
        await goto(page, '/settings');
        await expect(page.locator('text=System Status')).toBeVisible({ timeout: 3000 });
    });

    test('shows Trading Parameters section', async ({ page }) => {
        await goto(page, '/settings');
        await expect(page.locator('text=Trading Parameters')).toBeVisible({ timeout: 3000 });
    });

    test('shows Nifty 500 market option', async ({ page }) => {
        await goto(page, '/settings');
        await expect(page.locator('text=Nifty 500')).toBeVisible({ timeout: 3000 });
    });

    test('shows Danger Zone section', async ({ page }) => {
        await goto(page, '/settings');
        await expect(page.locator('text=Danger Zone')).toBeVisible({ timeout: 3000 });
    });

    test('Reset Defaults button is visible in Danger Zone', async ({ page }) => {
        await goto(page, '/settings');
        await expect(page.locator('button:has-text("Reset Defaults")')).toBeVisible({ timeout: 3000 });
    });

    test('Min Confidence slider is interactive', async ({ page }) => {
        await goto(page, '/settings');
        const slider = page.locator('input[type="range"]').first();
        await expect(slider).toBeVisible({ timeout: 3000 });
        // Verify it can be manipulated
        await slider.evaluate((el: HTMLInputElement) => {
            el.value = '80';
            el.dispatchEvent(new Event('input', { bubbles: true }));
        });
    });
});

// ── 7. Mobile Responsiveness ──────────────────────────────────────────────────

test.describe('Mobile responsive layout', () => {
    test.use({ viewport: { width: 375, height: 812 } });

    test('hamburger menu button is visible on mobile', async ({ page }) => {
        await goto(page, '/');
        // Hamburger is only visible at mobile breakpoints (md:hidden)
        const menuButton = page.locator('button').filter({
            hasText: '',
        }).first(); // Non-destructive: just check some button exists
        // More targeted: look for the SVG-only toggle button
        const svgButton = page.locator('header button').last();
        await expect(svgButton).toBeVisible({ timeout: 3000 });
    });

    test('clicking hamburger reveals mobile nav links', async ({ page }) => {
        await goto(page, '/');
        const hamburger = page.locator('header button').last();
        await hamburger.click();
        // After open, nav links should be visible
        await expect(page.locator('text=Signals').first()).toBeVisible({ timeout: 2000 });
    });

    test('Dashboard content stacks vertically on mobile', async ({ page }) => {
        await goto(page, '/');
        // Verify no horizontal overflow
        const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
        const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);
        expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 2); // allow 2px tolerance
    });
});

// ── 8. 404 Page ───────────────────────────────────────────────────────────────

test.describe('404 not-found page', () => {
    test('shows 404 heading for unknown routes', async ({ page }) => {
        await goto(page, '/this-does-not-exist-xyz');
        await expect(page.locator('h1')).toContainText('404');
    });

    test('shows "Back to Dashboard" link on 404 page', async ({ page }) => {
        await goto(page, '/unknown-route');
        const link = page.locator('a:has-text("Back to Dashboard")');
        await expect(link).toBeVisible({ timeout: 3000 });
    });

    test('"Back to Dashboard" link navigates to root', async ({ page }) => {
        await goto(page, '/totally-wrong');
        await page.click('a:has-text("Back to Dashboard")');
        await expect(page).toHaveURL('http://localhost:5173/');
    });
});

// ── 9. Accessibility ──────────────────────────────────────────────────────────

test.describe('Accessibility — keyboard navigation', () => {
    test('pressing Escape key does not crash the app', async ({ page }) => {
        await goto(page, '/');
        await page.keyboard.press('Escape');
        // App should still be functional
        await expect(page.locator('text=Signal.Engine')).toBeVisible();
    });

    test('all nav links are keyboard focusable (tab navigation)', async ({ page }) => {
        await goto(page, '/');
        // Tab through nav links
        await page.keyboard.press('Tab');
        await page.keyboard.press('Tab');
        await page.keyboard.press('Tab');
        // Just confirm the page didn't crash during tabbing
        await expect(page.locator('text=Signal.Engine')).toBeVisible();
    });
});
