import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright E2E configuration — Signal.Engine
 * ---------------------------------------------
 * Skill: webapp-testing (E2E, CI integration)
 * Run: npx playwright test
 * Install: npm install -D @playwright/test && npx playwright install chromium
 */
export default defineConfig({
    testDir: './e2e',
    timeout: 30_000,
    retries: process.env.CI ? 2 : 1,
    workers: process.env.CI ? 1 : undefined,
    reporter: process.env.CI ? 'github' : 'html',

    use: {
        baseURL: 'http://localhost:5173',
        headless: true,
        screenshot: 'only-on-failure',
        video: 'retain-on-failure',
        trace: 'on-first-retry',
    },

    projects: [
        {
            name: 'chromium',
            use: { ...devices['Desktop Chrome'] },
        },
        {
            name: 'Mobile Chrome',
            use: { ...devices['Pixel 5'] },
        },
    ],

    // Auto-start Vite dev server before running tests
    webServer: {
        command: 'npm run dev -- --port 5173',
        url: 'http://localhost:5173',
        reuseExistingServer: true,
        timeout: 15_000,
    },
});
