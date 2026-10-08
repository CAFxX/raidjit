// parity_compare.js — run one option document through a wasm module.
// Usage: node parity_compare.js <module.js> <options.json>
// stdout: one JSON line:
//   {"ok":true,"result":{...},"code_sha256":"...","const_sha256":"..."}
//   {"ok":false,"err":"..."}   (compile failed cleanly)
// Blobs are hashed in-memory (no files).
const fs = require('fs');
const crypto = require('crypto');

async function main() {
    const factory = require(process.argv[2]);
    const json = fs.readFileSync(process.argv[3], 'utf8');
    const mod = await factory();
    const p = mod.stringToNewUTF8(json);
    let rp;
    try {
        rp = mod._rjw_compile(p);
    } catch (e) {
        console.log(JSON.stringify({ok: false, err: 'trap: ' + e.message}));
        process.exit(0);
    }
    const res = mod.UTF8ToString(rp);
    let r;
    try {
        r = JSON.parse(res);
    } catch (e) {
        console.log(JSON.stringify({ok: false, err: 'non-JSON result'}));
        process.exit(0);
    }
    if (!r.ok) {
        console.log(JSON.stringify({ok: false, err: r.err || '?'}));
        process.exit(0);
    }
    const lenp = mod._malloc(8);
    const cp = mod._rjw_code_bytes(lenp);
    const clen = mod.getValue(lenp, 'i32');
    const code = Buffer.from(mod.HEAPU8.slice(cp, cp + clen));
    const dp = mod._rjw_const_bytes(lenp);
    const dlen = mod.getValue(lenp, 'i32');
    const cnst = Buffer.from(mod.HEAPU8.slice(dp, dp + dlen));
    mod._free(lenp);
    mod._rjw_free_string(rp);
    mod._free(p);
    console.log(JSON.stringify({
        ok: true,
        result: r,
        code_sha256: crypto.createHash('sha256').update(code).digest('hex'),
        const_sha256: crypto.createHash('sha256').update(cnst).digest('hex'),
    }));
}

main().catch(e => {
    console.log(JSON.stringify({ok: false, err: 'harness: ' + e.message}));
});
