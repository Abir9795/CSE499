"""Node.js scripts and native assertions sharing the candidate's scope."""

from src.verifier.runners.base import LanguageRunner


class JavaScriptRunner(LanguageRunner):
    suffix = ".cjs"

    def command(self, path):
        return ["node", str(path)]

    function_driver = r'''
const fs = require("node:fs");
const vm = require("node:vm");
const {Module, createRequire} = require("node:module");
const {dirname} = require("node:path");
const {randomUUID} = require("node:crypto");
const [candidate, assertion, entryPoint, outcome] = process.argv.slice(2);
// Use Node's normal realm and a CommonJS scope. A separate VM context changes
// object prototypes and can make valid deepStrictEqual assertions fail.
globalThis.assert = require("node:assert/strict");
const candidateModule = new Module(candidate, module);
candidateModule.filename = candidate;
candidateModule.require = createRequire(candidate);
const testHook = randomUUID();
let status = "error", error = "Function test did not reach its assertions.";
candidateModule[testHook] = (entryType, check) => {
    if (entryType !== "function") {
        throw new Error("Missing callable entry point: " + entryPoint);
    }
    try {
        check();
        status = "pass";
        error = "";
    } catch (failure) {
        status = failure && failure.code === "ERR_ASSERTION" ? "fail" : "error";
        error = String(failure && failure.stack || failure);
    }
};
try {
    const source = fs.readFileSync(candidate, "utf8").replace(/^\uFEFF/, "").replace(/^#![^\r\n]*/, "");
    const checks = fs.readFileSync(assertion, "utf8");
    const program = source + "\n;module[" + JSON.stringify(testHook) + "](typeof " +
        entryPoint + ", () => {\n" + checks + "\n});\n";
    const execute = vm.runInThisContext(Module.wrap(program), {filename: candidate});
    execute.call(candidateModule.exports, candidateModule.exports, candidateModule.require,
                 candidateModule, candidate, dirname(candidate));
} catch (failure) {
    status = "error";
    error = String(failure && failure.stack || failure);
}
fs.writeFileSync(outcome, JSON.stringify({status, error}), "utf8");
'''
