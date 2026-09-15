const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync('dashboard/static/app.js','utf8');
// Same slice the PVT dashboard test uses: CHECK_META through verdictHtml carries every
// label helper. checkLabel reads CHECK_META, so the slice has to start there.
const code=source.slice(source.indexOf('const CHECK_META ='),source.indexOf('function verdictHtml'));
const context=vm.createContext({defaults:{statuses:{}}});
vm.runInContext(code,context);
// Top-level `const` lands in the realm's lexical scope, not on the context object, so the
// arrow-function helpers are not reachable as context.<name> without this.
vm.runInContext('globalThis.checkLabel=checkLabel;globalThis.checkNames=checkNames;'
               +'globalThis.statusText=statusText;globalThis.guardReasonText=guardReasonText;',
               context);

// -- check keys ---------------------------------------------------------------------------
// `failing` is a list of the pipeline's own field names. Three places printed it straight
// at the reader: the verdict sub-line, the competition-spec chip, and the header of the
// exported netlist. CHECK_META has held the English for every one of them all along.
test('a failing check is named in English, not by its field name',()=>{
 assert.equal(context.checkNames(['boost_target','peak_in_band','eye_h']),
              'Boost on target, Peak in band, Eye width');
});
test('an unknown check key still renders as itself rather than as undefined',()=>{
 assert.equal(context.checkNames(['not_a_check']),'not_a_check');
});
test('no failing checks is an empty string, not "undefined"',()=>{
 assert.equal(context.checkNames(undefined),'');
 assert.equal(context.checkNames([]),'');
});
// A key with no CHECK_META entry is a key whose English nobody wrote. Pin the set so a
// new check in pipeline.py shows up here rather than in the UI as a bare field name.
test('every check the pipeline can fail has a human label',()=>{
 const py=fs.readFileSync('src/silq/pipeline.py','utf8');
 const block=py.slice(py.indexOf('check_of = {'),py.indexOf('user_checks ='));
 const keys=[...new Set([...block.matchAll(/:\s*"([a-z_0-9]+)"/g)].map((m)=>m[1]))];
 assert.ok(keys.length>=9,`expected the check_of map, found ${keys.length} keys`);
 const unlabelled=keys.filter((k)=>context.checkNames([k])===k);
 assert.deepEqual(unlabelled,[],'these render as their field name: '+unlabelled.join(', '));
});

// -- run statuses -------------------------------------------------------------------------
test('a run status reads as a phrase, not as a wire value',()=>{
 assert.equal(context.statusText('closed_but_failed_verification'),'closed but failed verification');
 assert.equal(context.statusText('fallback_fixed_design_not_ai'),'fell back to the fixed design');
 assert.equal(context.statusText('solved'),'verified');
});
// The map is the part that goes stale; the fallback is what keeps a fifth status readable.
test('a status the map has never heard of still loses its underscores',()=>{
 assert.equal(context.statusText('some_new_status'),'some new status');
 assert.equal(context.statusText(undefined),'');
});

// -- guard reasons ------------------------------------------------------------------------
// Every one of the twenty, read from guards.py rather than retyped: a hand-copied list
// would pass while the real vocabulary moved underneath it.
test('every guard reason in guards.py renders with its tier and without underscores',()=>{
 const codes=[...new Set([...fs.readFileSync('src/silq/guards.py','utf8')
   .matchAll(/"(T\d+\.\d+_[a-z0-9_]+)"/g)].map((m)=>m[1]))];
 assert.ok(codes.length>=20,`expected the guard vocabulary, found ${codes.length}`);
 for(const c of codes){
  const text=context.guardReasonText(c);
  assert.match(text,/^T\d+\.\d+ /,`${c} lost its tier code: ${text}`);
  assert.doesNotMatch(text,/_/,`${c} still has an underscore: ${text}`);
 }
});
test('the acronyms in a guard reason are not sentence-cased into nonsense',()=>{
 assert.equal(context.guardReasonText('T4.10_dc_gain_implausible'),'T4.10 \u2014 DC gain implausible');
 assert.equal(context.guardReasonText('T2.5_mosfet_not_in_saturation'),'T2.5 \u2014 MOSFET not in saturation');
 assert.equal(context.guardReasonText('T4.14_hd3_implausibly_good'),'T4.14 \u2014 HD3 implausibly good');
});
test('something that is not a tier code is passed through untouched',()=>{
 assert.equal(context.guardReasonText('hand written reason'),'hand written reason');
 assert.equal(context.guardReasonText(undefined),'');
});

// -- the wrappers -------------------------------------------------------------------------
// A structured-error wrapper only helps if nothing goes around it. Four callers used to
// call fetch() themselves and threw `server returned 404`: no code, no title, no next
// step -- which is what a missing results artifact looked like to the reader.
test('every request in app.js goes through the one wrapper',()=>{
 // Two callers, and the second is deliberate: the health poll needs to tell a 404
 // ("this build has no /api/health") apart from a failure, and it renders a pill rather
 // than a banner, so throwing on !res.ok would lose both. Pinned BY NAME so a third
 // caller has to justify itself here rather than appear quietly.
 const calls=[...source.matchAll(/(?:await\s+)?fetch\(/g)].map((m)=>m.index);
 assert.equal(calls.length,2,'fetch( appears '+calls.length+' times; only apiFetch and checkHealthOnce may call it');
 const apiAt=source.indexOf('async function apiFetch'), healthAt=source.indexOf('async function checkHealthOnce');
 assert.ok(calls[0]>apiAt&&calls[0]<source.indexOf('async function api(path'),'the first fetch belongs to apiFetch');
 assert.ok(calls[1]>healthAt&&calls[1]<healthAt+400,'the second fetch belongs to the health poll');
});
test('no request path throws a bare HTTP status as its whole message',()=>{
 assert.doesNotMatch(source,/server returned \$\{res\.status\}/);
});
test('the wrapper still exposes both body readers',()=>{
 assert.match(source,/async function api\(path, opts\) \{ return \(await apiFetch\(path, opts\)\)\.json\(\); \}/);
 assert.match(source,/async function apiText\(path, opts\) \{ return \(await apiFetch\(path, opts\)\)\.text\(\); \}/);
});

// -- the sentence that started all of this ------------------------------------------------
test('the PVT verdict no longer blames a budget for a corner miss',()=>{
 // Comment lines stripped first: the comment explaining the fix necessarily quotes
 // the sentence it replaced, and matching that would make the note undeletable.
 const verdict=source.slice(source.indexOf('function verdictHtml'),
                            source.indexOf('function verdictHtml')+1400)
   .split(/\r?\n/).filter((l)=>!l.trim().startsWith('//')).join(' ');
 assert.doesNotMatch(verdict,/within the repair budget/);
 assert.match(verdict,/pvtFailureLabel\(r\.pvt\)/);
});
