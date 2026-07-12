import React from "react";
import { interpolate } from "remotion";
import {
  AnalystCard,
  Appear,
  BigCard,
  DarkFade,
  ResultCard,
  Term,
  typed,
  useT,
  WordmarkCard,
} from "./components";
import { durOf } from "./timeline";
import A from "./artifacts.json";

const f1 = (n: number): string => n.toFixed(1);
const f2 = (n: number): string => n.toFixed(2);
/** First `n` sentences of a rationale, for card-sized display. */
const firstSentences = (s: string, n: number): string =>
  s.split(". ").slice(0, n).join(". ").replace(/\.?$/, ".");

/* ---------- S0 title ---------- */
export const TitleScene: React.FC = () => (
  <DarkFade dur={durOf("title")}>
    <WordmarkCard
      tagline={
        <>
          An autonomous agent that optimizes LLM inference on AWS Graviton —
          using Arm&rsquo;s own tooling.
        </>
      }
      url="Arm AI Developer Challenge · Cloud AI"
    />
  </DarkFade>
);

/* ---------- S1 problem ---------- */
export const ProblemScene: React.FC = () => (
  <DarkFade dur={durOf("problem")}>
    <BigCard
      big={
        <>
          Most LLMs on Graviton ship at <span className="amber">{f1(A.baseline.median)}</span>{" "}
          tokens/sec.
          <br />
          The silicon can do <em>{f1(A.winner.median)}</em>.
        </>
      }
      sub="TUNING IT IS EXPERT WORK · MOST DEVELOPERS SHIP THE NAIVE BUILD"
    />
  </DarkFade>
);

/* ---------- S2 solution ---------- */
export const SolutionScene: React.FC = () => (
  <DarkFade dur={durOf("solution")}>
    <BigCard
      big={
        <>
          A <em>deterministic autotuner</em> sweeps the levers.
          <br />
          A <em>Claude analyst</em> reads Arm Performix counters and narrates
          every keep and revert.
        </>
      }
      sub="THE TUNER DECIDES · THE LLM EXPLAINS · A QUALITY GUARD VETOES"
    />
  </DarkFade>
);

/* ---------- S3 baseline ---------- */
export const BaselineScene: React.FC = () => {
  const t = useT();
  const cmd = `armsmith baseline --target ${A.instance} --model ${A.model}-${A.quant}`;
  const shown = typed(cmd, t, 0.8, 34);
  return (
    <Term title={`armsmith · control plane -> ${A.instance} (Graviton4, ${A.cores} cores)`}>
      <span className="p">$ </span>
      {shown}
      {"\n\n"}
      {t > 4.2 ? (
        <>
          <span className="ok">✓</span> honest baseline{" "}
          <span className="dim">(portable build, GGML_NATIVE=OFF)</span>
          {"   "}
          <span className="num">{f2(A.baseline.median)}</span> tok/s decode{" "}
          <span className="dim">
            [{f2(A.baseline.ciLow)}–{f2(A.baseline.ciHigh)}]
          </span>
          {"\n"}
        </>
      ) : null}
      {t > 6.6 ? (
        <>
          <span className="ok">✓</span> expert config{" "}
          <span className="dim">(pre-registered = the 100% mark)</span>
          {"     "}
          <span className="num">{f2(A.expert.median)}</span> tok/s decode{" "}
          <span className="dim">
            [{f2(A.expert.ciLow)}–{f2(A.expert.ciHigh)}]
          </span>
          {"\n\n"}
        </>
      ) : null}
      {t > 9 ? (
        <span className="dim">
          run minted: {A.runId} · N={A.baseline.nRepeats} repeats · median + 95% CI
        </span>
      ) : null}
    </Term>
  );
};

/* ---------- S4 the analyst's prior ---------- */
export const PriorScene: React.FC = () => (
  <AnalystCard
    rationale={firstSentences(A.prior.rationale, 2)}
    priority={[A.prior.action, ...A.steps.slice(1, 4).map((s) => s.action)].slice(0, 4)}
    typedUntil={6}
  />
);

/* ---------- S5 the refutation ---------- */
export const RefuteScene: React.FC = () => (
  <ResultCard
    action={`${A.prior.action} (the analyst's #1 prior)`}
    deltaPct={A.prior.deltaPct}
    detail={
      <>
        Screened on real silicon: <b>{f1(A.prior.deltaPct)}%</b> decode on this core and
        model. The datasheet said try it first. The measurement says no.
      </>
    }
    kept={false}
    badgeAt={6}
  />
);

/* ---------- S6 the pivot ---------- */
export const PivotScene: React.FC = () => (
  <ResultCard
    action={A.pivot ? A.pivot.action : "ggml_native"}
    deltaPct={A.pivot ? A.pivot.deltaPct : 0}
    detail={
      <>
        <b>{f2(A.winner.median)}</b> tok/s decode{" "}
        [{f2(A.winner.ciLow)}–{f2(A.winner.ciHigh)}], confirmed at N={A.winner.nRepeats},
        non-overlapping 95% CIs. TTFT {Math.round(A.baseline.ttftMs)} ms →{" "}
        <b>{Math.round(A.winner.ttftMs)} ms</b>.
      </>
    }
    kept={true}
    badgeAt={7}
  />
);

/* ---------- S7 the sweep ---------- */
export const SweepScene: React.FC = () => {
  const t = useT();
  return (
    <div className="sweepwrap">
      <div className="sweeptitle">the tuner exhausts the space — every step logged</div>
      <div className="strip">
        {A.steps.map((s, i) => {
          const at = 0.8 + i * 0.55;
          if (t < at) return null;
          return (
            <div className={`step ${s.kept ? "kept" : "rev"}`} key={i}>
              {s.action} {s.pct >= 0 ? "+" : ""}
              {f1(s.pct)}%{s.kept ? " ✓" : ""}
            </div>
          );
        })}
      </div>
      <Appear at={0.8 + A.steps.length * 0.55 + 0.8}>
        <div className="converge">
          converged: <b>{f2(A.winner.median)} tok/s</b> · {A.steps.length} candidates ·{" "}
          {A.steps.filter((s) => s.kept).length} kept
        </div>
      </Appear>
    </div>
  );
};

/* ---------- S8 quality guard ---------- */
export const QualityScene: React.FC = () => (
  <DarkFade dur={durOf("quality")}>
    <BigCard
      big={
        <>
          Quality guard: KL vs baseline <em>{A.winner.kl}</em>
          <br />
          perplexity <span className="amber">{f2(A.winner.ppl)}</span> — unchanged model,
          double the speed.
        </>
      }
      sub="SPEED WINS THAT DEGRADE THE MODEL ARE REJECTED, NOT REPORTED"
    />
  </DarkFade>
);

/* ---------- S9 victory bars ---------- */
export const VictoryScene: React.FC = () => {
  const t = useT();
  const grow = interpolate(t, [0.6, 2.6], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const max = Math.max(A.winner.median, A.expert.median) * 1.12;
  const bar = (v: number): number => (v / max) * 520 * grow;
  const cols: { v: number; label: string; color: string }[] = [
    { v: A.baseline.median, label: "naive baseline\n(what most ship)", color: "#68737d" },
    { v: A.expert.median, label: "expert hand-tuned\n(pinned BEFORE the run)", color: "#3fb6e0" },
    { v: A.winner.median, label: "armsmith\n(autonomous)", color: "#2eb67d" },
  ];
  return (
    <div className="barswrap">
      <div className="bars">
        {cols.map((c) => (
          <div className="barcol" key={c.label}>
            <div className="bar1" style={{ height: bar(c.v), background: c.color }}>
              <div className="val" style={{ opacity: grow }}>
                {f2(c.v)} tok/s
              </div>
            </div>
            <div className="barlabel" style={{ whiteSpace: "pre-line" }}>
              {c.label}
            </div>
          </div>
        ))}
      </div>
      <Appear at={3.4}>
        <div className="gapline">
          <b>{Math.round(A.gapClosedPct)}%</b> of the naive→expert gap closed ·{" "}
          {A.speedup.toFixed(1)}x decode
        </div>
      </Appear>
    </div>
  );
};

/* ---------- S10 repro ---------- */
export const ReproScene: React.FC = () => {
  const t = useT();
  const cmd = `armsmith repro ${A.runId} --target <fresh instance>`;
  const shown = typed(cmd, t, 0.8, 34);
  const repro = (A as { repro?: { pass: boolean; decode: number; tolPct: number } }).repro;
  return (
    <Term title="armsmith · replay the saved recipe — no LLM in the loop">
      <span className="p">$ </span>
      {shown}
      {"\n\n"}
      {t > 4 ? (
        <>
          <span className="dim">
            recipe.json → build flags, quant, threads, KV-cache · deterministic replay
          </span>
          {"\n"}
        </>
      ) : null}
      {t > 6.5 && repro ? (
        <>
          {"\n"}
          <span className="ok">PASS</span>: decode{" "}
          <span className="num">{f2(repro.decode)}</span> tok/s{" "}
          <span className="dim">
            (recipe {f2(A.winner.median)}, tol ±{repro.tolPct}%)
          </span>
        </>
      ) : null}
    </Term>
  );
};

/* ---------- S11 close ---------- */
export const CloseScene: React.FC = () => (
  <DarkFade dur={durOf("close")}>
    <WordmarkCard
      tagline="The tuner decides. The analyst explains. The recipe replays."
      strip={[
        { value: `${A.speedup.toFixed(1)}x`, label: "decode speedup" },
        { value: `${Math.round(A.gapClosedPct)}%`, label: "gap closed" },
        { value: String(A.winner.kl), label: "KL vs baseline" },
      ]}
      url="github.com/kitfunso/armsmith · pip install armsmith · Apache-2.0"
    />
  </DarkFade>
);
