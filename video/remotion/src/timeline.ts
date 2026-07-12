// The beat sheet as data. Durations in seconds; every scene, caption, and chip
// hangs off this one table so re-timing the video is a one-file edit.
export const FPS = 30;
export const sec = (s: number): number => Math.round(s * FPS);

export const SCENES = [
  { id: "title", dur: 8 },
  { id: "problem", dur: 12 },
  { id: "solution", dur: 14 },
  { id: "baseline", dur: 14 },
  { id: "prior", dur: 13 },
  { id: "refute", dur: 12 },
  { id: "pivot", dur: 14 },
  { id: "sweep", dur: 16 },
  { id: "quality", dur: 12 },
  { id: "victory", dur: 15 },
  { id: "repro", dur: 14 },
  { id: "close", dur: 10 },
] as const;

export type SceneId = (typeof SCENES)[number]["id"];

const starts: Record<string, number> = {};
{
  let at = 0;
  for (const s of SCENES) {
    starts[s.id] = at;
    at += s.dur;
  }
}

/** Scene start in seconds from the top of the video. */
export const startOf = (id: SceneId): number => starts[id];
/** Scene duration in seconds. */
export const durOf = (id: SceneId): number =>
  SCENES.find((s) => s.id === id)!.dur;

export const TOTAL_SECONDS = SCENES.reduce((a, s) => a + s.dur, 0);
export const TOTAL_FRAMES = sec(TOTAL_SECONDS);

// Bottom captions, absolute seconds. No free-floating numbers: anchor to scenes.
const cap = (
  scene: SceneId,
  from: number,
  to: number,
  text: string
): { from: number; to: number; text: string } => ({
  from: startOf(scene) + from,
  to: startOf(scene) + to,
  text,
});

export const CAPTIONS = [
  cap("baseline", 1.5, 13, "An honest baseline: the portable build a developer actually ships. The expert config is pinned BEFORE discovery."),
  cap("prior", 1, 12, "The analyst reads the evidence and proposes a search order. It is advisory: the deterministic tuner decides."),
  cap("refute", 1, 11, "Measured. KleidiAI is slower on this core and model. The prior loses to the measurement: REVERTED."),
  cap("pivot", 1, 13, "The native build doubles decode speed. Confirmed with repeats, non-overlapping confidence intervals: KEPT."),
  cap("sweep", 1, 15, "The tuner exhausts the lever space. Every keep and revert is evidence, logged with its rationale."),
  cap("quality", 1, 11, "Every candidate is scored for quality. A config can never win by quantizing the model into mush."),
  cap("victory", 1, 14, "The autotuner beat the pre-registered expert config. The success bar could not be gamed after the fact."),
  cap("repro", 1, 13, "The recipe replays on a fresh instance with no LLM in the loop. If it is not replayable, it is not a result."),
];

export const CHIPS = [
  { from: startOf("baseline"), to: startOf("sweep"), text: "Deterministic tuner + LLM analyst" },
  { from: startOf("sweep"), to: startOf("victory"), text: "Quality-guarded search" },
  { from: startOf("victory"), to: startOf("close"), text: "Replayable recipe" },
];
