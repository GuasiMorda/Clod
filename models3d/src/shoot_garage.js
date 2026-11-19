// usage: node shoot_garage.js <outDir> <car> <mode: stills|turn|glb> [frames]
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [, , outDir, carName, mode, nFrames] = process.argv;
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  page.on('pageerror', e => console.log('ERR', e.message));
  page.on('console', m => { if (m.type() === 'error') console.log('CONSOLE', m.text()); });
  await page.goto('http://127.0.0.1:8765/garage.html');
  await page.waitForFunction('window.ready === true', null, { timeout: 120000 });
  console.log('car', await page.evaluate(n => window.setCar(n), carName));
  if (mode === 'stills') {
    for (const m of ['front', 'rear', 'side', 'low']) {
      await page.evaluate(m => window.renderShot(m, 0), m);
      await page.locator('canvas').screenshot({ path: `${outDir}/${carName}_${m}.png` });
    }
  } else if (mode === 'turn') {
    const n = +nFrames || 180;
    for (let i = 0; i < n; i++) {
      await page.evaluate(t => window.renderShot('turn', t), i / n);
      await page.locator('canvas').screenshot({ path: `${outDir}/${carName}_t${String(i).padStart(4, '0')}.jpg`, type: 'jpeg', quality: 93 });
    }
  } else if (mode === 'glb') {
    const b64 = await page.evaluate(() => window.exportGLB());
    fs.writeFileSync(`${outDir}/${carName}.glb`, Buffer.from(b64, 'base64'));
  }
  await browser.close();
})();
