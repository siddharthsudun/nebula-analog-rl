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

test('a second request is not bounded by the first one answer', () => {
  // The bug this pins: the two edges were applied one at a time, and each read the OTHER
  // knob's CURRENT value as its bound. So an out-of-range ask (which parks both knobs at
  // the top of the rail) silently truncated the next request -- "peak between 1.5 and
  // 2.0 GHz" landed on 1.50-2.50 GHz, which is neither request -- and nothing reported it,
  // because each edge on its own was inside the rail.
  const {context, el} = setup();
  context.applyParsedNoise = () => {};
  const ask = (lo, hi) => context.applyParse({fields: [
    {field: 'peak_freq_hi_ghz', label: 'Peak band, high', unit: 'GHz', asked: hi, asked_disp: hi, role: 'scores'},
    {field: 'peak_freq_lo_ghz', label: 'Peak band, low', unit: 'GHz', asked: lo, asked_disp: lo, role: 'scores'},
  ]});
  ask(2.7, 3.3);                                    // parks both knobs at the rail top
  const out = ask(1.5, 2.0);                        // wholly inside the rail
  assert.equal(Number(el['fband-lo'].value), 1.5);
  assert.equal(Number(el['fband-hi'].value), 2.0, 'the high edge must be the one asked for');
  assert.deepEqual(plain(out), [], 'a request the rail can serve reports no error');
});

test('an edge the reader did not state goes back to the default, not the last request', () => {
  // The parser's own assumption text promises "left the other edge at its default". If
  // the knob keeps the PREVIOUS request's value that sentence is false, and the run
  // carries a bound neither request contained: "peak below 2.2 GHz" followed by "peak at
  // least 1.8 GHz" was landing on 1.80-2.20 GHz.
  const {context, el} = setup();
  context.applyParsedNoise = () => {};
  const edge = (field, v) => context.applyParse({fields: [
    {field, label: 'Peak band', unit: 'GHz', asked: v, asked_disp: v, role: 'scores'}]});
  edge('peak_freq_hi_ghz', 2.2);
  assert.deepEqual([Number(el['fband-lo'].value), Number(el['fband-hi'].value)], [1.25, 2.2]);
  edge('peak_freq_lo_ghz', 1.8);
  assert.deepEqual([Number(el['fband-lo'].value), Number(el['fband-hi'].value)], [1.8, 2.5],
    'the high edge must return to the competition default, not stay at 2.2 GHz');
});

test('an ask wholly outside the rail still lands as a band, not a point', () => {
  // Both edges clamp to the same rail end. Opening inward keeps peak_in_band satisfiable;
  // a zero-width band is a requirement no design can ever meet.
  const {context, el} = setup();
  context.applyParsedNoise = () => {};
  context.applyParse({fields: [
    {field: 'peak_freq_hi_ghz', label: 'Peak band, high', unit: 'GHz', asked: 3.3, asked_disp: 3.3, role: 'scores'},
    {field: 'peak_freq_lo_ghz', label: 'Peak band, low', unit: 'GHz', asked: 2.7, asked_disp: 2.7, role: 'scores'},
  ]});
  assert.ok(Number(el['fband-hi'].value) - Number(el['fband-lo'].value) > 0.049,   // one step, in float
    'a zero-width band is a peak_in_band requirement no design can ever satisfy');
});

test('only the rail counts as out of range, not a knob that was in the way', () => {
  // `clamped` compared the landed value to the ask, so an edge the OTHER knob pushed was
  // reported as a range the tool cannot represent. An error that cries wolf on a request
  // the rail can serve is how a real out-of-range message gets ignored.
  const src = source.slice(source.indexOf('const _offRail'), source.indexOf('function setBandPair'));
  assert.match(src, /si < min - 1e-9 \|\| si > max \+ 1e-9/);
  assert.ok(!source.slice(source.indexOf('function setBandEdge'), source.indexOf('const _offRail'))
    .includes('Math.abs(Number(knob.value) - si)'),
    'the single-edge setter must use the same rail-only test');
});

// -- the refusal ------------------------------------------------------------------------
// Reporting the clamp was half the fix. The other half is that the run must not HAPPEN:
// a message above a run that went ahead anyway is still a run scored against a
// requirement the user did not state, and the operator has to notice the message to know
// it. These are source assertions because runDesign is an async network call; what they
// pin is the wiring, and the wiring is the part that silently rots.
const runDesignSrc = source.slice(source.indexOf('async function runDesign'),
                                  source.indexOf('async function runDesign') + 2600);

test('a run is refused outright when an ask fell off a rail', () => {
  assert.match(runDesignSrc, /if \(useComposer && lastOutOfRange\.length && !outOfRangeAck\)/,
    'runDesign must refuse an out-of-range ask before it spends simulator budget');
  const guard = runDesignSrc.indexOf('lastOutOfRange.length');
  assert.ok(guard >= 0 && guard < runDesignSrc.indexOf('readNoiseRequest()'),
    'the refusal must come BEFORE the run is set up, not after the budget is committed');
  assert.ok(runDesignSrc.slice(guard, runDesignSrc.indexOf('readNoiseRequest()')).includes('return'),
    'the guard must RETURN -- reporting and then running is the defect it replaces');
});

test('the refusal does not fire on the slider path, which has no way to be out of range', () => {
  // The panel's own controls are the rails. Blocking a slider-driven run would be a
  // permanent refusal the user could not clear by any action in the interface.
  assert.match(runDesignSrc, /useComposer && lastOutOfRange/);
});

test('every parse clears the acknowledgement', () => {
  // Consent is to ONE substitution. If it survived the next keystroke, a user who
  // accepted 2.50 GHz once would silently accept every later clamp in the session.
  const parseSrc = source.slice(source.indexOf('async function parseNow'),
                                source.indexOf('let typeTimer'));
  const applied = parseSrc.indexOf('lastOutOfRange = applyParse(spec)');
  assert.ok(applied >= 0, 'parseNow must keep what applyParse could not honour');
  assert.ok(parseSrc.slice(applied, applied + 120).includes('outOfRangeAck = false'),
    'storing a fresh out-of-range list must clear the previous acknowledgement');
  assert.equal((parseSrc.match(/lastOutOfRange = \[\]; outOfRangeAck = false;/g) || []).length, 2,
    'an empty box and a failed parse must both clear the list: nothing was applied, so '
    + 'nothing is outstanding, and a stale list would block the next run for no reason');
});

test('the message says out of range, names the rail, and says the run is blocked', () => {
  const notes = source.slice(source.indexOf('const rangeNote ='),
                             source.indexOf('const reader = spec.llm_backend'));
  assert.match(notes, /Out of range/, 'the user asked for this to read as an error');
  assert.match(notes, /The run is blocked/);
  assert.match(notes, /tunable range the problem statement sets/,
    'say WHY the rail is where it is, or it reads as an arbitrary limitation');
  assert.match(notes, /data-ack-range/, 'an error with no way past it is a dead end');
  assert.match(source, /\$\$\("\[data-ack-range\]", strip\)/,
    'the acknowledge button must actually be wired');
  // One ask, one error: the band is two knobs and must not produce two messages.
  assert.match(notes, /BAND_FIELDS\.includes\(c\.row\.field\)/);
  assert.match(notes, /rangeNote\("Peak frequency band"/);
});
