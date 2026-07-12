// Builds src/artifacts.json from a REAL armsmith run directory so every
// number in the video is pipeline output, never hand-typed. Run id comes
// from ARMSMITH_RUN_ID or trajectories/latest. The JSON is generated and
// gitignored; trajectories/<run>/ stays the single committed source of truth.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, "..", "..", "..");
const runId =
  process.env.ARMSMITH_RUN_ID ??
  readFileSync(join(repo, "trajectories", "latest"), "utf8").trim();
const runDir = join(repo, "trajectories", runId);

const j = (name) => JSON.parse(readFileSync(join(runDir, name), "utf8"));
const recipe = j("recipe.json");
const expert = j("expert.json");
const manifest = j("manifest.json");
const steps = readFileSync(join(runDir, "trajectory.jsonl"), "utf8")
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line));

const stat = (r) => ({
  median: r.decode_tok_s.median,
  ciLow: r.decode_tok_s.ci_low,
  ciHigh: r.decode_tok_s.ci_high,
  ttftMs: r.prefill_ttft_ms.median,
  nRepeats: r.n_repeats,
});

const base = recipe.baseline_result;
const win = recipe.winning_result;
const firstKept = steps.find((s) => s.kept);

const artifacts = {
  runId: recipe.run_id,
  instance: manifest.target.instance_type,
  cores: manifest.target.n_physical_cores,
  model: recipe.model.name,
  quant: recipe.model.baseline_quant,
  baseline: stat(base),
  expert: stat(expert),
  winner: {
    ...stat(win),
    kl: win.quality.kl_vs_baseline,
    ppl: win.quality.perplexity,
    flags: recipe.winning_config.cmake_flags.join(" "),
  },
  gapClosedPct: recipe.gap_closed_pct,
  speedup: win.decode_tok_s.median / base.decode_tok_s.median,
  prior: {
    action: steps[0].action_id,
    rationale: steps[0].rationale,
    deltaPct: steps[0].delta.pct,
  },
  pivot: firstKept
    ? {
        action: firstKept.action_id,
        rationale: firstKept.rationale,
        deltaPct: firstKept.delta.pct,
      }
    : null,
  steps: steps.map((s) => ({
    action: s.action_id,
    pct: s.delta.pct,
    sig: s.delta.ci_significant,
    kept: s.kept,
  })),
};

// Optional: fresh-instance repro result, written by the repro session as
// trajectories/<run>/repro.json ({pass, decode, tolPct}). The repro scene
// only renders a PASS line when this REAL result exists - never fabricated.
try {
  artifacts.repro = JSON.parse(readFileSync(join(runDir, "repro.json"), "utf8"));
} catch {
  artifacts.repro = null;
}

const out = join(here, "..", "src", "artifacts.json");
writeFileSync(out, JSON.stringify(artifacts, null, 2) + "\n");
console.log(`wrote ${out} from run ${runId} (${steps.length} steps)`);
