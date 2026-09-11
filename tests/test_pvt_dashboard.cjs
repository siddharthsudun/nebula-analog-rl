const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync('dashboard/static/app.js','utf8');
// From the corner-count helpers, not from netlistWithVerdict itself: the exported
// provenance header now names the grid that actually ran, so the helpers are part of
// the unit under test.
const code=source.slice(source.indexOf('const pvtCount ='),source.indexOf('function verdictHtml'));
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
