#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { WASI } from 'node:wasi';

function parseArgs(argv) {
  const args = { bundle: path.resolve('dist/wasm/dev'), module: null };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    switch (arg) {
      case '--bundle':
        args.bundle = path.resolve(argv[++i]);
        break;
      case '--module':
        args.module = path.resolve(argv[++i]);
        break;
      default:
        throw new Error(`unknown argument: ${arg}`);
    }
  }
  return args;
}

function loadMatrix(bundle) {
  const matrixPath = path.join(bundle, 'matrix.json');
  const raw = fs.readFileSync(matrixPath, 'utf8');
  const parsed = JSON.parse(raw);
  if (!parsed.fixtures || parsed.fixtures.length === 0) {
    throw new Error(`no fixtures defined at ${matrixPath}`);
  }
  return parsed.fixtures;
}

function runFixture(compiledModule, bundle, fixture) {
  const stdoutDir = fs.mkdtempSync(path.join(os.tmpdir(), `objstore-node-${fixture.id}-stdout-`));
  const stderrDir = fs.mkdtempSync(path.join(os.tmpdir(), `objstore-node-${fixture.id}-stderr-`));
  const stdoutFile = path.join(stdoutDir, 'stdout.log');
  const stderrFile = path.join(stderrDir, 'stderr.log');
  const stdoutFd = fs.openSync(stdoutFile, 'w+');
  const stderrFd = fs.openSync(stderrFile, 'w+');
  try {
    const wasi = new WASI({
      args: [
        'objstore_wasm_matrix',
        '--bundle-root', '/bundle',
        '--fixture', fixture.path,
        '--fixture-id', fixture.id,
      ],
      env: {},
      preopens: { '/bundle': bundle },
      stdout: stdoutFd,
      stderr: stderrFd,
      returnOnExit: true,
    });

    const imports = { wasi_snapshot_preview1: wasi.wasiImport };
    const instance = new WebAssembly.Instance(compiledModule, imports);
    wasi.start(instance);

    const output = fs.readFileSync(stdoutFile, 'utf8').trim();
    if (!output) {
      const errOutput = fs.readFileSync(stderrFile, 'utf8');
      throw new Error(`fixture ${fixture.id} produced no output: ${errOutput}`);
    }
    const lastLine = output.split('\n').pop();
    return JSON.parse(lastLine);
  } finally {
    fs.closeSync(stdoutFd);
    fs.closeSync(stderrFd);
    fs.rmSync(stdoutDir, { recursive: true, force: true });
    fs.rmSync(stderrDir, { recursive: true, force: true });
  }
}

async function main() {
  const args = parseArgs(process.argv);
  const bundle = args.bundle;
  const modulePath = args.module || path.join(bundle, 'wasi', 'objstore_wasm_matrix.wasm');
  if (!fs.existsSync(bundle)) {
    throw new Error(`bundle not found: ${bundle}`);
  }
  if (!fs.existsSync(modulePath)) {
    throw new Error(`module not found: ${modulePath}`);
  }

  const moduleBytes = fs.readFileSync(modulePath);
  const compiled = await WebAssembly.compile(moduleBytes);
  const fixtures = loadMatrix(bundle);
  const failures = [];

  for (const fixture of fixtures) {
    try {
      const result = runFixture(compiled, bundle, fixture);
      console.log(`[node] ${fixture.id}: ${result.status}`);
      if (result.status !== 'ok') {
        failures.push(result);
      }
    } catch (err) {
      console.error(`[node] ${fixture.id}: failed`, err.message);
      failures.push({ fixture: fixture.id, error: err.message });
    }
  }

  if (failures.length) {
    console.error('node harness failures:', JSON.stringify(failures, null, 2));
    process.exit(1);
  }

  console.log('node harness passed all fixtures');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
