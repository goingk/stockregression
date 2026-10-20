import { access, mkdir } from 'node:fs/promises'
import { spawnSync } from 'node:child_process'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
const frontend = path.resolve(root, 'frontend')
const candidates = [
  process.env.STOCKREGRESSION_PYTHON,
  path.join(root, '.venv', 'Scripts', 'python.exe'),
  path.join(root, '.venv', 'bin', 'python'),
  'python',
].filter(Boolean)

let python
for (const candidate of candidates) {
  if (candidate === 'python') { python = candidate; break }
  try { await access(candidate); python = candidate; break } catch { /* try the next interpreter */ }
}
if (!python) throw new Error('找不到 Python 解释器；可设置 STOCKREGRESSION_PYTHON')

const exported = spawnSync(python, [path.join(root, 'scripts', 'export_openapi.py'), path.join(frontend, 'openapi.json')], {
  cwd: root,
  stdio: 'inherit',
  env: { ...process.env, PYTHONPATH: path.join(root, 'src') },
})
if (exported.error) throw exported.error
if (exported.status !== 0) process.exit(exported.status ?? 1)

await mkdir(path.join(frontend, 'src', 'generated'), { recursive: true })
const generated = spawnSync(
  process.execPath,
  [path.join(frontend, 'node_modules', 'openapi-typescript', 'bin', 'cli.js'), 'openapi.json', '-o', 'src/generated/api-schema.d.ts'],
  { cwd: frontend, stdio: 'inherit' },
)
if (generated.error) throw generated.error
if (generated.status !== 0) process.exit(generated.status ?? 1)
