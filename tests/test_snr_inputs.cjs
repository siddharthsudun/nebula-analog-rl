// Run with node --test tests/test_snr_inputs.cjs. No browser or server required.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function setup(mode='unknown') {
  const values = {'noise-mode':mode,'noise-form':'range','noise-low':'1','noise-high':'50',
    'noise-unit':'rms','noise-signal-reference':'tx_vpp','noise-signal':'1','noise-band-low':'10','noise-band-high':'5000','noise-value':'5'};
  const elements = Object.fromEntries(Object.entries(values).map(([k,v]) => [k,{value:v,dataset:{}}]));
  elements['noise-enabled'] = {checked:true,dataset:{}};
  elements['snr-advanced'] = {open:false};
  const context = vm.createContext({window:{},document:{querySelector:selector => elements[selector.slice(1)] || null},
    escapeHtml:s => String(s).replaceAll('<','&lt;')});
  vm.runInContext(fs.readFileSync('dashboard/static/snr-input.js','utf8'), context);
  return {context,elements};
}
test('unknown carries explicit assumptions and units', () => {
  const {context} = setup();
  const r = context.readNoiseRequest();
  assert.equal(r.mode,'unknown');
  assert.equal(r.low_vrms,.001);
  assert.equal(r.bandwidth_hz[1],5e9);
  assert.equal(r.assumed_fields.join(','),'noise,signal,bandwidth');
});

test('off bypasses even invalid fields and omits the API request', () => {
  const {context,elements} = setup('measured');
  elements['noise-enabled'].checked=false;
  elements['noise-signal'].value='invalid';
  assert.equal(context.readNoiseRequest(), undefined);
  assert.equal(JSON.stringify({noise_request:context.readNoiseRequest()}), '{}');
});

test('explicit opt out wins over parsed SNR and stays off', () => {
  const {context,elements} = setup();
  context.applyParsedNoise({source:'explicit_opt_out',enabled:false});
  context.applyParsedNoise({enabled:true,request:{mode:'unknown'}});
  assert.equal(elements['noise-enabled'].checked,false);
});

test('measured dB input does not become millivolts', () => {
  const {context,elements} = setup('measured');
  elements['noise-unit'].value='snr';
  elements['noise-value'].value='20';
  const request=context.readNoiseRequest();
  assert.equal(request.input_snr_db,20);
  assert.equal(request.value_vrms,undefined);
});
test('measured requires complete amplitude; alternate reference retained', () => {
  const {context,elements} = setup('measured');
  elements['noise-signal'].value='';
  assert.throws(() => context.readNoiseRequest(),/Complete/);
  elements['noise-signal'].value='.2';
  elements['noise-signal-reference'].value='ctle_input_vrms';
  assert.equal(context.readNoiseRequest().signal_reference,'ctle_input_vrms');
  assert.equal(context.readNoiseRequest().value_vrms,.005);
});
test('budget and invalid bandwidth are handled before submit', () => {
  const {context,elements} = setup('estimated');
  elements['noise-form'].value='budget';
  assert.equal(context.readNoiseRequest().budget_vrms,.005);
  elements['noise-band-low'].value='6000';
  assert.throws(() => context.readNoiseRequest(),/ordered band/);
});
test('conditional failures have no unconditional pass claim', () => {
  const {context} = setup();
  const html = context.noiseResultHtml({noise_evaluation:{conditional:true,status:'noise_measurement_failed',
    reason:'<failure>',provenance:{noise:'assumed'},search_note:'Noise did not steer search.'}});
  assert.match(html,/Conditional noise assessment/);
  assert.match(html,/&lt;failure>/);
  assert.doesNotMatch(html,/<failure>/);
});
