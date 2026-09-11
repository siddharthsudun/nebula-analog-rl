const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync('dashboard/static/app.js','utf8');
const code=source.slice(source.indexOf('function netlistWithVerdict'),source.indexOf('function verdictHtml'));
const context=vm.createContext({defaults:{statuses:{solved:'solved',fallback:'fallback',closed_not_verified:'closed_but_failed_verification'}}});
vm.runInContext(code,context);
test('PVT verified exports cannot claim only TT was checked',()=>{
 const text=context.netlistWithVerdict({netlist:'* final',status:'solved',provenance:{},pvt:{accepted:true}});
 assert.match(text,/45\/45/);
 assert.doesNotMatch(text,/TYPICAL CORNER ONLY|no corner sweep/);
});
test('failed PVT is explicit in exported circuit and fixed reuse is attributed',()=>{
 const text=context.netlistWithVerdict({netlist:'* final',status:'pvt_not_verified',provenance:{fixed_anchor_reused:true},pvt:{accepted:false,status:'budget_exhausted'}});
 assert.match(text,/NOT VERIFIED/);
 assert.match(text,/fixed delivered sizing/);
 assert.doesNotMatch(text,/TYPICAL CORNER ONLY/);
});
