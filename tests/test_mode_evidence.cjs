const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

async function render() {
  const data = JSON.parse(fs.readFileSync('dashboard/static/data/mode-evidence.json', 'utf8'));
  const host = {innerHTML: ''};
  let ready;
  vm.runInNewContext(fs.readFileSync('dashboard/static/evidence.js', 'utf8'), {
    document: {getElementById: () => host, addEventListener: (event, cb) => { ready = cb; }},
    fetch: async () => ({ok: true, json: async () => data})
  });
  await ready();
  return host.innerHTML;
}

test('measured success rate is derived from Auto and discloses comparison scope', async () => {
  const host = {innerHTML: await render()};
  assert.match(host.innerHTML, /Measured success rate/);
  assert.match(host.innerHTML, /96\.8%/);
  assert.match(host.innerHTML, /30 of 31 specifications solved with Auto/);
  assert.match(host.innerHTML, /nominal circuit/);
  assert.match(host.innerHTML, /other four modes are unchanged/);
  assert.doesNotMatch(host.innerHTML, /<h2>Model accuracy/);
});

// The per-mode cards were removed: only the headline Auto rate and the methods
// table may state a solve rate, so no two rates can be read as a ranking.
test('the per-mode accuracy cards are gone and the methods table survives', async () => {
  const html = await render();
  for (const gone of ['evidence-grid', 'evidence-bar', 'evidence-stats', '<h2>Fastest</h2>', '<h2>Thinking</h2>', 'Not solved']) {
    assert.ok(!html.includes(gone), `expected the mode cards to be gone, still found: ${gone}`);
  }
  assert.ok(html.includes('<h2>Measured success rate</h2>'), 'the headline card must remain');
  assert.match(html, /<th scope="row">Fastest<\/th><td>31\/31<\/td>/);
  assert.match(html, /<th scope="row">Retarget<\/th><td>18\/31<\/td>/);
  assert.match(html, /not a like-for-like ranking/);
  assert.equal((html.match(/<h2>Measured success rate<\/h2>/g) || []).length, 1);
});

// The headline number is the one claim on the page; recompute it from the raw
// sweep rather than trusting the snapshot that renders it.
test('the headline Auto rate reproduces from the raw sweep it cites', () => {
  const snapshot = JSON.parse(fs.readFileSync('dashboard/static/data/mode-evidence.json', 'utf8'));
  const rows = JSON.parse(fs.readFileSync('dashboard/static/data/mode_bench5.json', 'utf8'));
  const auto = snapshot.modes.find((r) => r.mode === 'auto');
  const matched = new Set(snapshot.indices);
  const mine = rows.filter((r) => r.mode === 'auto' && matched.has(r.i));
  const solved = mine.filter((r) => r.status === 'solved' && r.passed === true).length;
  assert.equal(mine.length, auto.cases, 'matched case count drifted from the snapshot');
  assert.equal(solved, auto.solved, 'Auto solve count does not reproduce from mode_bench5.json');
  assert.equal(Number((100 * solved / mine.length).toFixed(1)), 96.8);
});

// Fastest's row comes from a later rerun, not from mode_bench5.json. That is the
// whole reason it outranks Auto, and the reason the cards went away -- so pin the
// disagreement rather than letting a future edit quietly "fix" it.
test('Fastest is sourced from the rerun and the page says so', () => {
  const snapshot = JSON.parse(fs.readFileSync('dashboard/static/data/mode-evidence.json', 'utf8'));
  const bench = JSON.parse(fs.readFileSync('dashboard/static/data/mode_bench5.json', 'utf8'));
  const rerun = JSON.parse(fs.readFileSync('dashboard/static/data/fastest_full32.json', 'utf8'));
  const fastest = snapshot.modes.find((r) => r.mode === 'fastest');
  const matched = new Set(snapshot.indices);
  const old = bench.filter((r) => r.mode === 'fastest' && matched.has(r.i));
  assert.notEqual(old.filter((r) => r.status === 'solved' && r.passed === true).length, fastest.solved);
  const now = rerun.filter((r) => matched.has(r.i));
  assert.equal(now.length, fastest.cases);
  assert.equal(now.filter((r) => r.status === 'solved' && r.passed === true).length, fastest.solved);
  assert.match(snapshot.timestamp_note, /Fastest row updated/);
});
