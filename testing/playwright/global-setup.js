const { chromium } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

module.exports = async (config) => {
  const baseURL = config.projects[0]?.use?.baseURL || 'http://localhost:9099/';
  const email = process.env.LOGIN_EMAIL || 'admin@owi.com';
  const password = process.env.LOGIN_PASSWORD || 'admin';

  const authDir = path.join(__dirname, '.auth');
  const statePath = path.join(authDir, 'state.json');

  fs.mkdirSync(authDir, { recursive: true });

  // If a prior state exists, reuse it. This keeps local dev fast.
  if (fs.existsSync(statePath) && !process.env.FORCE_RELOGIN) {
    return;
  }

  const browser = await chromium.launch({ headless: !!process.env.CI });
  const context = await browser.newContext();
  const page = await context.newPage();

  await page.goto(baseURL);
  await page.waitForLoadState('domcontentloaded');

  const existingToken = await page.evaluate(() => localStorage.getItem('userToken'));
  if (!existingToken) {
    await page.waitForSelector('#email', { timeout: 30_000 });

    await page.waitForLoadState('load');
    await page.waitForLoadState('networkidle');

    await page.waitForFunction(
      () => typeof window.__sveltekit_dev !== 'undefined' || typeof window.__sveltekit !== 'undefined',
      { timeout: 30_000 }
    );

    await page.fill('#email', email);
    await page.fill('#password', password);

    await Promise.all([
      page.waitForResponse(
        (r) => r.url().includes('/creator/login') && r.request().method() === 'POST',
        { timeout: 30_000 }
      ),
      page.click('form > button')
    ]);

    await page.waitForFunction(() => !!localStorage.getItem('userToken'), { timeout: 15_000 });
  }

  const tokenAfter = await page.evaluate(() => localStorage.getItem('userToken'));
  if (!tokenAfter) {
    await browser.close();
    throw new Error(
      'Login did not produce localStorage.userToken. Check UI selectors or credentials (LOGIN_EMAIL/LOGIN_PASSWORD).'
    );
  }

  await context.storageState({ path: statePath });
  await browser.close();
};
