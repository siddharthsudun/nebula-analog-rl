const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync('dashboard/static/app.js','utf8');
// From the corner-count helpers, not from netlistWithVerdict itself: the exported
// provenance header now names the grid that actually ran, so the helpers are part of
// the unit under test.
// Starts at CHECK_META, not at the corner-count helpers: pvtFailureLabel reads the
// human label for a failing check out of it, so it is part of the unit under test.
const code=source.slice(source.indexOf('const CHECK_META ='),source.indexOf('function verdictHtml'));
const context=vm.createContext({defaults:{statuses:{solved:'solved',fallback:'fallback',closed_not_verified:'closed_but_failed_verification'}}});
vm.runInContext(code,context);
test('PVT verified exports cannot claim only TT was checked',()=>{
 const text=context.netlistWithVerdict({netlist:'* final',status:'solved',provenance:{},pvt:{accepted:true,corners_checked:45,grid_label:'full 45-corner'}});
 assert.match(text,/45 \/ 45 corners/);
 assert.doesNotMatch(text,/TYPICAL CORNER ONLY|no corner sweep/);
});
// Fastest certifies tt/ss/ff only. An exported netlist is the artifact most likely to
// outlive the session that produced it and be read as a sign-off, so its header must
// carry the grid that ran -- a three-corner pass written out as "45 / 45" would be a
// claim nobody could check afterwards.
test('a three-corner export says three corners, not forty-five',()=>{
 const text=context.netlistWithVerdict({netlist:'* final',status:'solved',provenance:{},pvt:{accepted:true,corners_checked:3,grid_label:'3-corner (tt/ss/ff)'}});
 assert.match(text,/3 \/ 3 corners \(tt\/ss\/ff\)/);
 assert.doesNotMatch(text,/45/);
});
// Results recorded before the field existed were all full sweeps.
test('a result with no recorded corner count is treated as the full grid',()=>{
 const text=context.netlistWithVerdict({netlist:'* final',status:'solved',provenance:{},pvt:{accepted:true}});
 assert.match(text,/45 \/ 45 corners/);
});
test('failed PVT is explicit in exported circuit and fixed reuse is attributed',()=>{
 const text=context.netlistWithVerdict({netlist:'* final',status:'pvt_not_verified',provenance:{fixed_anchor_reused:true},pvt:{accepted:false,status:'budget_exhausted'}});
 assert.match(text,/NOT VERIFIED/);
 assert.match(text,/fixed delivered sizing/);
 assert.doesNotMatch(text,/TYPICAL CORNER ONLY/);
});

// The failure message is the whole point of the panel. `unresolved_within_budget` is the
// single token the certifier emits for EVERY non-acceptance, and shown raw it is read as
// "the run went over budget" -- the opposite of what it means. These pin the reading.
test('a corner miss is reported as a corner miss, not as a budget',()=>{
 const text=context.pvtFailureLabel({accepted:false,status:'unresolved_within_budget',verification:{rows:[
  {corner:['tt',1.8,27],passed:true,checks:{boost_target:true}},
  {corner:['ss',1.71,125],passed:false,checks:{boost_target:false,power:true}},
  {corner:['fs',1.71,125],passed:false,checks:{boost_target:false,power:true}},
 ]}});
 assert.match(text,/1 \/ 3 corners/);
 assert.match(text,/Boost on target/);
 assert.match(text,/ss, 1\.71 V, 125/);
 assert.match(text,/fs, 1\.71 V, 125/);
 assert.doesNotMatch(text,/budget/);
});
// The one status that really IS the clock must still say so.
test('a sweep that ran out of time says so in words',()=>{
 const text=context.pvtFailureLabel({accepted:false,status:'budget_exhausted'});
 assert.match(text,/ran out of time/);
 assert.doesNotMatch(text,/corners/);
});
// A refusal the rows do not explain -- disjoint-worker-pid checks fail outside the grid --
// must fall back to the token rather than name a corner that passed.
test('a non-acceptance with no failing row falls back to the raw status',()=>{
 const text=context.pvtFailureLabel({accepted:false,status:'unresolved_within_budget',verification:{rows:[
  {corner:['tt',1.8,27],passed:true,checks:{boost_target:true}},
 ]}});
 assert.match(text,/unresolved_within_budget/);
 assert.doesNotMatch(text,/1 \/ 1 corners/);
});
// The netlist outlives the session, so it keeps the machine token AND gains the reason.
test('the exported netlist carries both the status token and the reason',()=>{
 const text=context.netlistWithVerdict({netlist:'* final',status:'pvt_not_verified',provenance:{},pvt:{
  accepted:false,status:'unresolved_within_budget',verification:{rows:[
   {corner:['ss',1.71,125],passed:false,checks:{boost_target:false}},
  ]}}});
 assert.match(text,/NOT VERIFIED \(unresolved_within_budget\)/);
 assert.match(text,/0 \/ 1 corners/);
 assert.match(text,/Boost on target/);
});
