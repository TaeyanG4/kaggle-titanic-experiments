/** Render local DOT sources into PNG/SVG documentation assets. No training or network. */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { instance } from '@viz-js/viz';
import sharp from 'sharp';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const sources = resolve(root, 'docs/diagrams');
const assets = resolve(root, 'docs/assets');
const names = ['workflow', 'v5-ensemble', 'validation-boundary', 'final-lineage'];
const sha = (data) => createHash('sha256').update(data).digest('hex');
const repoPath = (path) => relative(root, path).replaceAll('\\', '/');

async function main() {
  await mkdir(assets, { recursive: true });
  const viz = await instance();
  const manifest = { renderer: `Graphviz ${viz.graphvizVersion}; @viz-js/viz 3.31.0; sharp 0.35.5`, diagrams: [] };
  for (const name of names) {
    const source = resolve(sources, `${name}.dot`);
    const input = await readFile(source);
    const result = viz.render(input.toString('utf8'), { format: 'svg', engine: 'dot' });
    if (result.status !== 'success') throw new Error(`DOT rendering failed for ${name}: ${JSON.stringify(result.errors)}`);
    if (result.errors?.length) console.warn(`${name}: ${JSON.stringify(result.errors)}`);
    const svg = Buffer.from(result.output, 'utf8');
    const svgPath = resolve(assets, `${name}.svg`);
    const pngPath = resolve(assets, `${name}.png`);
    await writeFile(svgPath, svg);
    const { data: png, info } = await sharp(svg, { density: 160 }).png().toBuffer({ resolveWithObject: true });
    if (info.width < 600 || info.height < 300) throw new Error(`Unexpected dimensions for ${name}`);
    await writeFile(pngPath, png);
    manifest.diagrams.push({
      name, source: repoPath(source), source_sha256: sha(input), width: info.width, height: info.height,
      files: [{ path: repoPath(pngPath), sha256: sha(png) }, { path: repoPath(svgPath), sha256: sha(svg) }],
    });
    console.log(`${name}: ${info.width} x ${info.height} PNG and SVG`);
  }
  await writeFile(resolve(sources, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n', 'utf8');
  console.log('Four flowcharts rendered from local sources. No model fitting or Kaggle submission.');
}

main().catch((error) => { console.error(error.message); process.exitCode = 1; });
