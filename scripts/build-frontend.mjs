import { cp, mkdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const projectRoot = process.cwd();
const sourceDir = path.join(projectRoot, 'frontend');
const outputDir = path.join(projectRoot, 'dist');
const configuredApiUrl = process.env.VITE_API_URL?.trim();
const isNetlifyBuild = process.env.NETLIFY === 'true';

if (isNetlifyBuild && !configuredApiUrl) {
    throw new Error(
        'VITE_API_URL is required for Netlify builds. Set it to the public HTTPS backend origin.'
    );
}

const apiUrl = new URL(configuredApiUrl || 'http://localhost:8765');
if (!['http:', 'https:'].includes(apiUrl.protocol)) {
    throw new Error('VITE_API_URL must use http:// or https://');
}
if (isNetlifyBuild && apiUrl.protocol !== 'https:') {
    throw new Error('VITE_API_URL must use HTTPS on Netlify.');
}

const normalizedApiUrl = apiUrl.toString().replace(/\/$/, '');
const runtimeConfig = [
    '// Generated at build time. Do not edit the deployed copy.',
    'window.__CONFIDENCE_ENGINE_CONFIG__ = Object.freeze({',
    `    API_URL: ${JSON.stringify(normalizedApiUrl)}`,
    '});',
    ''
].join('\n');

await rm(outputDir, { recursive: true, force: true });
await mkdir(outputDir, { recursive: true });
await cp(sourceDir, outputDir, { recursive: true });
await writeFile(path.join(outputDir, 'runtime-config.js'), runtimeConfig, 'utf8');

console.log(`Frontend built with API_URL=${normalizedApiUrl}`);
