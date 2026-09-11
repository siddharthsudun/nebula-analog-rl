// Run with node --test tests/test_band_slider.cjs. No browser or server required.
// The peak band moved out of the acceptance-limits list and onto a two-knob slider; what
// matters is that the wire format did not move with it. A default run must still send no
// requirement at all, or every frozen number measured through this dashboard stops being
// comparable to one measured before the control existed.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

// readRequirements builds its object inside the vm realm; deepEqual compares prototypes
// across realms and rejects a structurally identical result, so compare the projection.
const plain = (o) => (o === null ? null : JSON.parse(JSON.stringify(o)));

const source = fs.readFileSync('dashboard/static/app.js', 'utf8');
const code = source.slice(source.indexOf('const reqInputs'), source.indexOf('function renderRequirements'))
  + source.slice(source.indexOf('function readRequirements'), source.indexOf('function selectMode'))
  // applyParse is what the language reader drives, and it is defined after selectMode, so
  // it needs its own slice. Without it the clamp-reporting contract below is untestable.
  + source.slice(source.indexOf('function applyParse'), source.indexOf('function renderChips'));

function knob(value) {
  return {value: String(value), min: '0', max: '0', step: '0.05', style: {},
          handlers: [], addEventListener(_, h) { this.handlers.push(h); },
          drag(v) { this.value = String(v); this.handlers.forEach((h) => h()); }};
}

function setup() {
  const cls = () => ({on: new Set(), toggle(n, s) { s ? this.on.add(n) : this.on.delete(n); }});
  const el = {
    'fband-lo': knob(1.25), 'fband-hi': knob(2.5), 'fband-fill': {style: {}},
    'fband-val': {textContent: '', parentElement: {classList: cls()}},
  };
  const context = vm.createContext({
    $: (s) => el[s.slice(1)] || null,
    fmt: (v, d) => Number(v).toFixed(d),
    refreshRequirementTags() {},
    defaults: {requirements: [
      {field: 'peak_freq_lo_ghz', default_disp: 1.25, scale: 1, unit: 'GHz', label: 'Peak band, low'},
      {field: 'peak_freq_hi_ghz', default_disp: 2.5, scale: 1, unit: 'GHz', label: 'Peak band, high'},
    ]},
  });
  vm.runInContext(code, context);
  context.initBand();
  return {context, el};
}

test('the rail is exactly the tunable range and both knobs start at its ends', () => {
  const {el} = setup();
  assert.equal(el['fband-lo'].min, 1.25);
  assert.equal(el['fband-hi'].max, 2.5);
  assert.equal(Number(el['fband-lo'].value), 1.25);
  assert.equal(Number(el['fband-hi'].value), 2.5);
  assert.equal(el['fband-val'].textContent, '1.25\u20132.50');
});

test('an untouched slider sends no requirement', () => {
  const {context} = setup();
  assert.equal(context.readRequirements(), null);
});

test('a narrowed band travels as ordinary requirement fields', () => {
  const {context, el} = setup();
  el['fband-lo'].drag(1.6);
  el['fband-hi'].drag(2.1);
  assert.deepEqual(plain(context.readRequirements()), {peak_freq_lo_ghz: 1.6, peak_freq_hi_ghz: 2.1});
});

test('only the moved edge is sent; the untouched one stays at the default', () => {
  const {context, el} = setup();
  el['fband-hi'].drag(2.0);
  assert.deepEqual(plain(context.readRequirements()), {peak_freq_hi_ghz: 2.0});
});

test('the knobs never cross and never close the band to zero width', () => {
  const {el} = setup();
  el['fband-lo'].drag(2.5);
  assert.equal(Number(el['fband-lo'].value), 2.45);
  assert.equal(Number(el['fband-hi'].value), 2.5);
  el['fband-hi'].drag(1.25);
  assert.equal(Number(el['fband-lo'].value), 1.25);
  assert.equal(Number(el['fband-hi'].value), 1.3);
});

test('the language reader lands on the slider, clamped into the tunable range', () => {
  const {context, el} = setup();
  // An edge INSIDE the rail lands verbatim and reports nothing to apologise for.
  assert.deepEqual(plain(context.setRequirementSI('peak_freq_lo_ghz', 1.8)),
                   {landed: 1.8, lo: 1.25, hi: 2.5, clamped: false});
  assert.equal(Number(el['fband-lo'].value), 1.8);
  // Asked for a band edge outside what the problem statement allows: clamped, not ignored.
  context.setRequirementSI('peak_freq_hi_ghz', 4.0);
  assert.equal(Number(el['fband-hi'].value), 2.5);
  assert.deepEqual(plain(context.readRequirements()), {peak_freq_lo_ghz: 1.8});
});

test('an edge the rail cannot represent SAYS so instead of clamping quietly', () => {
  // The regression this pins: "peak at 3 GHz" parses to a 2.70-3.30 GHz band, both edges
  // fall off a rail that stops at 2.50, and the knobs used to settle on 2.45-2.50 and send
  // `peak_freq_lo_ghz: 2.45` -- a requirement the user never stated, reported as theirs and
  // then reported satisfied. Clamping is the right behaviour; clamping in silence is not.
  const {context, el} = setup();
  const lo = context.setRequirementSI('peak_freq_lo_ghz', 2.7);
  const hi = context.setRequirementSI('peak_freq_hi_ghz', 3.3);
  assert.equal(lo.clamped, true, 'a 2.70 GHz low edge is off a rail that ends at 2.50');
  assert.equal(hi.clamped, true, 'and so is a 3.30 GHz high edge');
  assert.equal(lo.hi, 2.5, 'the report carries the rail, so the note can name it');
  // Still clamped -- the knobs remain inside the tunable range -- but now it is reportable.
  assert.equal(Number(el['fband-lo'].value), 2.45);
  assert.equal(Number(el['fband-hi'].value), 2.5);
});

test('a narrowed band is marked on the readout, a default one is not', () => {
  const {el} = setup();
  assert.equal(el['fband-val'].parentElement.classList.on.has('narrowed'), false);
  el['fband-lo'].drag(1.5);
  assert.equal(el['fband-val'].parentElement.classList.on.has('narrowed'), true);
});


// A plain slider, for the two rails that are not the band's.
function railed(value, min, max) {
  return {value: String(value), min: String(min), max: String(max), style: {},
          handlers: [], addEventListener(_, h) { this.handlers.push(h); }};
}

test('applyParse reports every ask its rails could not represent', () => {
  // The whole point. "20 dB of boost, peak at 3 GHz" parses cleanly, and all three
  // numbers fall off a rail: boost stops at 12 dB, the band at 2.50 GHz. Every one used
  // to be rounded into range in silence, so the run was scored against limits the user
  // never asked for -- a 12 dB target and a 2.45-2.50 GHz band -- and reported as passing
  // them. applyParse now hands the caller what it could not honour so the composer can
  // say so; what it must NOT do is drop the clamp or invent a rail it does not have.
  const {context, el} = setup();
  el.target = railed(9, 3, 12);
  el.channel = railed(12, 6, 20);
  context.applyParsedNoise = () => {};
  context.updateTarget = () => {};
  context.updateChannel = () => {};

  const out = context.applyParse({fields: [
    {field: 'target_boost_db', label: 'Target boost', unit: 'dB', asked: 20, asked_disp: 20, role: 'steers'},
    {field: 'peak_freq_lo_ghz', label: 'Peak band, low', unit: 'GHz', asked: 2.7, asked_disp: 2.7, role: 'scores'},
    {field: 'peak_freq_hi_ghz', label: 'Peak band, high', unit: 'GHz', asked: 3.3, asked_disp: 3.3, role: 'scores'},
  ]});

  assert.deepEqual(plain(out.map((c) => [c.row.field, c.landed, c.lo, c.hi])), [
    ['target_boost_db', 12, 3, 12],
    ['peak_freq_lo_ghz', 2.45, 1.25, 2.5],
    ['peak_freq_hi_ghz', 2.5, 1.25, 2.5],
  ]);
});

test('an ask that fits every rail reports nothing to apologise for', () => {
  // The other half of the contract: a note on a run that honoured the request in full
  // would train the user to ignore the notes, which is how the silent clamp got missed.
  const {context, el} = setup();
  el.target = railed(9, 3, 12);
  el.channel = railed(12, 6, 20);
  context.applyParsedNoise = () => {};
  context.updateTarget = () => {};
  context.updateChannel = () => {};

  const out = context.applyParse({fields: [
    {field: 'target_boost_db', label: 'Target boost', unit: 'dB', asked: 9, asked_disp: 9, role: 'steers'},
    {field: 'peak_freq_lo_ghz', label: 'Peak band, low', unit: 'GHz', asked: 1.8, asked_disp: 1.8, role: 'scores'},
  ]});
  assert.deepEqual(plain(out), []);
  assert.equal(Number(el['fband-lo'].value), 1.8);
});
