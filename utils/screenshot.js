const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function takeScreenshot(url, savePath) {
    const browser = await puppeteer.launch({ headless: true });
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 720 });
    await page.goto(url, { waitUntil: 'networkidle2' });
    await page.screenshot({ path: savePath });
    await browser.close();
}

const [url, savePath] = process.argv.slice(2);

takeScreenshot(url, savePath)
    .then(() => console.log(`Captura guardada en ${savePath}`))
    .catch(error => console.error('Error al tomar la captura:', error));