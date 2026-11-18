const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [,, outDir, from, to, step] = process.argv;
  const times = [];
  if (step === 'list') times.push(...from.split(',').map(Number));
  else for (let f = +from; f < +to; f++) times.push(f / 30);
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  page.on('console', m => console.log('PAGE', m.text()));
  page.on('pageerror', e => console.log('ERR', e.message));
  await page.goto('http://127.0.0.1:8765/index.html');
  await page.waitForFunction('window.ready === true', null, { timeout: 120000 });
  const t0 = Date.now();
  for (let i = 0; i < times.length; i++) {
    const t = times[i];
    await page.evaluate(t => window.renderAt(t), t);
    const name = step === 'list' ? `t_${t.toFixed(2)}.jpg` : `f_${String(Math.round(t * 30)).padStart(4, '0')}.jpg`;
    await page.locator('canvas').screenshot({ path: `${outDir}/${name}`, type: 'jpeg', quality: 93 });
    if (i % 30 === 0) console.log('frame', i, ((Date.now() - t0) / (i + 1)).toFixed(0), 'ms/f');
  }
  await browser.close();
})();
