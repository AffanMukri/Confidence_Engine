import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';

const config = await readFile(new URL('../dist/runtime-config.js', import.meta.url), 'utf8');
const expected = process.env.VITE_API_URL || 'http://localhost:8765';

assert.match(config, new RegExp(expected.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
assert.doesNotMatch(config, /confidence-engine\.netlify\.app:8765/);
console.log('Runtime API configuration verified.');
