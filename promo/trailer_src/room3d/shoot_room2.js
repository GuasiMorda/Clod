const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [, , outDir, from, to, colorsPath] = process.argv;
  const colors = JSON.parse(fs.readFileSync(colorsPath, 'utf8'));
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  page.on('pageerror', e => console.log('ERR', e.message));
  await page.goto('http://127.0.0.1:8765/room2.html');
  await page.waitForFunction('window.ready === true', null, { timeout: 120000 });
  const corners = {};
  const list = from.includes(',') ? from.split(',').map(Number) : null;
  const frames = list || Array.from({ length: +to - +from }, (_, k) => +from + k);
  for (const f of frames) {
    const c = colors[Math.min(f, colors.length - 1)] || [0.1, 0.05, 0.2];
    corners[f] = await page.evaluate(([t, r, g, b]) => window.renderAt(t, r, g, b), [f / 30, c[0], c[1], c[2]]);
    await page.locator('canvas').screenshot({ path: `${outDir}/r_${String(f).padStart(4, '0')}.jpg`, type: 'jpeg', quality: 94 });
  }
  fs.writeFileSync(`${outDir}/corners_${frames[0]}.json`, JSON.stringify(corners));
  await browser.close();
})();
