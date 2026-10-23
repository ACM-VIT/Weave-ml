// Synthetic loopback-only fixture using the selected backend's actual ingestion
// auth/parser/controller, SQL persistence, and transactional outbox projection.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const root = process.env.WEAVE_BACKEND_ROOT;
assert.ok(root && path.isAbsolute(root));
const require = createRequire(path.join(root, 'backend/package.json'));
const { default: express } = await import(pathToFileURL(require.resolve('express')).href);
const load = file => import(pathToFileURL(path.join(root, 'backend', file)).href);
const { assertDisposableDatabase } = await load('tests/database/disposableDatabase.ts');
assertDisposableDatabase();
const prefix = process.env.SIGNAL_FIXTURE_PREFIX;
assert.match(prefix ?? '', /^signal-handoff-[a-f0-9]{12}$/);
assert.equal(process.env.INTERNAL_API_SECRET, 'local-signal-fixture-secret');
const { sqlClient: db } = await load('db/index.ts');
const { requireInternalSecret } = await load('middlewares/internalAuthMiddleware.ts');
const { canonicalIngestionJsonParser } = await load('middlewares/scopedJsonBodyMiddleware.ts');
const { upsertRepositories } = await load('controllers/ingestionController.ts');
const ids = new Set();
const owners = new Set();
const requests = [];
const app = express();
app.post('/api/internal/v2/ingestion/repositories/upsert', requireInternalSecret,
  canonicalIngestionJsonParser, (req, res, next) => {
    const records = req.body?.repositories;
    if (!Array.isArray(records) || records.length > 3 || requests.length >= 3
        || records.some(repo => repo.owner !== prefix || !/^\d{1,16}$/.test(repo.github_id)
          || !/^\d{1,16}$/.test(repo.owner_github_id))) return res.sendStatus(400);
    for (const record of records) { ids.add(record.github_id); owners.add(record.owner_github_id); }
    requests.push(records);
    next();
  }, upsertRepositories);
app.get('/fixture/snapshot', requireInternalSecret, async (_req, res) => {
  const githubIds = [...ids];
  const jobs = await db`SELECT job_type,payload FROM telemetry.ml_outbox
    WHERE aggregate_id IN (SELECT repo_id FROM app.repos WHERE github_id IN
      (SELECT jsonb_array_elements_text(${JSON.stringify(githubIds)}::text::jsonb)::bigint))
    ORDER BY aggregate_id,job_type`;
  const rows = await db`SELECT repo.repo_id,repo.github_id::text,repo.created_at AS product_created_at,
      content.readme,content.content_version,content.content_hash
    FROM app.repos repo JOIN app.repo_content content USING(repo_id)
    WHERE repo.github_id IN (SELECT jsonb_array_elements_text(${JSON.stringify(githubIds)}::text::jsonb)::bigint)
    ORDER BY repo.github_id`;
  res.json({ backend_revision: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(),
    wire_requests: requests, rows, jobs });
});
const server = app.listen(0, '127.0.0.1', () => {
  console.log(`SIGNAL_FIXTURE_PORT=${server.address().port}`);
});
let stopping = false;
async function stop() {
  if (stopping) return;
  stopping = true;
  server.close(); server.closeAllConnections();
  try {
    const githubIds = [...ids];
    await db`DELETE FROM telemetry.ml_outbox WHERE aggregate_id IN
      (SELECT repo_id FROM app.repos WHERE github_id IN
        (SELECT jsonb_array_elements_text(${JSON.stringify(githubIds)}::text::jsonb)::bigint))`;
    await db`DELETE FROM telemetry.readme_summary_jobs WHERE repo_id IN
      (SELECT repo_id FROM app.repos WHERE github_id IN
        (SELECT jsonb_array_elements_text(${JSON.stringify(githubIds)}::text::jsonb)::bigint))`;
    await db`DELETE FROM app.repos WHERE github_id IN
      (SELECT jsonb_array_elements_text(${JSON.stringify(githubIds)}::text::jsonb)::bigint)`;
    await db`DELETE FROM app.social_profiles WHERE provider='github' AND claimed_by_user_id IS NULL
      AND provider_subject IN (SELECT jsonb_array_elements_text(${JSON.stringify([...owners])}::text::jsonb))
      AND handle=${prefix}`;
  } finally { await db.end({ timeout: 1 }); }
}
process.on('SIGTERM', () => { stop().then(() => process.exit(0), error => { console.error(error); process.exit(1); }); });
