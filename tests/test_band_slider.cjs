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
  + source.slice(source.indexOf('function readRequirements'), source.indexOf('function selectMode'));

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
  assert.equal(context.setRequirementSI('peak_freq_lo_ghz', 1.8), true);
  assert.equal(Number(el['fband-lo'].value), 1.8);
  // Asked for a band edge outside what the problem statement allows: clamped, not ignored.
  context.setRequirementSI('peak_freq_hi_ghz', 4.0);
  assert.equal(Number(el['fband-hi'].value), 2.5);
  assert.deepEqual(plain(context.readRequirements()), {peak_freq_lo_ghz: 1.8});
});

test('a narrowed band is marked on the readout, a default one is not', () => {
  const {el} = setup();
  assert.equal(el['fband-val'].parentElement.classList.on.has('narrowed'), false);
  el['fband-lo'].drag(1.5);
  assert.equal(el['fband-val'].parentElement.classList.on.has('narrowed'), true);
});
