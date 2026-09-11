import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";
import { createElement } from "react";
import ts from "typescript";

// Render the real component with a fixed socket snapshot; no provider/network.
const require = createRequire(import.meta.url);
const source = readFileSync(new URL("../src/features/simulator/ExperimentBrief.tsx", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
}).outputText;

function render(assistance) {
  const module = { exports: {} };
  const currentScenario = {
    difficulty: "Intermediate", objective: "Complete the game", success: "Finish safely",
    observations: ["position"], actions: ["thrust"], reward_terms: ["progress"],
    termination_conditions: ["success"], metric_label: "score",
    model_assumptions: ["Simple dynamics"], reference_controller: "PD controller",
    actor_warm_start: assistance,
  };
  const localRequire = (name) => name === "../../hooks/useTrainingSocket"
    ? { useTrainingSocket: () => ({ currentScenario, metricMode: "max", status: { seed: 42 } }) }
    : require(name);
  new Function("require", "module", "exports", compiled)(localRequire, module, module.exports);
  return renderToStaticMarkup(createElement(module.exports.default));
}

test("assisted learning is visible before collapsed details and does not claim independent starts", () => {
  const html = render({ role: "actor behavior-cloning warm start", pure_model_free_from_scratch: false,
    expert: { description: "PD demonstration labels", used_at_inference: false } });
  const visibleBrief = html.slice(0, html.indexOf("<details"));
  assert.match(visibleBrief, /first learns from demonstrations/);
  assert.match(visibleBrief, /PPO practice/);
  assert.match(visibleBrief, /expert does not control/);
  assert.doesNotMatch(html, /PPO starts independently/);
});

test("unassisted learning does not imply expert demonstrations were used", () => {
  const html = render(null);
  const visibleBrief = html.slice(0, html.indexOf("<details"));
  assert.match(visibleBrief, /without expert demonstrations/);
  assert.doesNotMatch(visibleBrief, /first learns from demonstrations/);
});
