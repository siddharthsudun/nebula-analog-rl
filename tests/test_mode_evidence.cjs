const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
test('measured success rate is derived from Auto and discloses comparison scope', async () => {
  const data = JSON.parse(fs.readFileSync('dashboard/static/data/mode-evidence.json', 'utf8'));
  const host = {innerHTML: ''};
  let ready;
  vm.runInNewContext(fs.readFileSync('dashboard/static/evidence.js', 'utf8'), {
    document: {getElementById: () => host, addEventListener: (event, cb) => { ready = cb; }},
    fetch: async () => ({ok: true, json: async () => data})
  });
  await ready();
  assert.match(host.innerHTML, /Measured success rate/);
  assert.match(host.innerHTML, /96\.8%/);
  assert.match(host.innerHTML, /30 of 31 specifications solved with Auto/);
  assert.match(host.innerHTML, /nominal circuit/);
  assert.match(host.innerHTML, /other four modes are unchanged/);
  assert.doesNotMatch(host.innerHTML, /<h2>Model accuracy/);
});
