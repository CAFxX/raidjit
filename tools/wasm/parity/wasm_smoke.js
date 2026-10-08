// wasm_smoke.js — smoke test for the rjw_compile API in node.
// Usage: node wasm_smoke.js <module.js> '<json>'
// Prints the result JSON, writes code/const blobs to ./smoke.code.bin etc.
const fs = require('fs');

async function main() {
    const modPath = process.argv[2];
    const json = process.argv[3];
    const factory = require(modPath);
    const mod = await factory();
    const p = mod.stringToNewUTF8(json);
    const rp = mod._rjw_compile(p);
    const res = mod.UTF8ToString(rp);
    console.log(res);
    let r;
    try {
        r = JSON.parse(res);
    } catch (e) {
        console.error('RESULT NOT JSON');
        process.exit(2);
    }
    if (r.ok) {
        const lenp = mod._malloc(8);
        const cp = mod._rjw_code_bytes(lenp);
        const clen = mod.getValue(lenp, 'i32');
        fs.writeFileSync('smoke.code.bin',
                         Buffer.from(mod.HEAPU8.slice(cp, cp + clen)));
        const dp = mod._rjw_const_bytes(lenp);
        const dlen = mod.getValue(lenp, 'i32');
        fs.writeFileSync('smoke.const.bin',
                         Buffer.from(mod.HEAPU8.slice(dp, dp + dlen)));
        console.log(`wrote smoke.code.bin (${clen} B), smoke.const.bin (${dlen} B)`);
        mod._free(lenp);
    }
    mod._rjw_free_string(rp);
    mod._free(p);
    process.exit(r.ok ? 0 : 1);
}

main().catch(e => { console.error(e); process.exit(3); });
