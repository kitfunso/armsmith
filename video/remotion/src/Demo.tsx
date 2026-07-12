import React from "react";
import { AbsoluteFill, interpolate, Sequence, useCurrentFrame } from "remotion";
import "./style.css";
import { CAPTIONS, CHIPS, FPS, SCENES, sec, startOf } from "./timeline";
import {
  BaselineScene,
  CloseScene,
  PivotScene,
  PriorScene,
  ProblemScene,
  QualityScene,
  RefuteScene,
  ReproScene,
  SolutionScene,
  SweepScene,
  TitleScene,
  VictoryScene,
} from "./scenes";

const SCENE_COMPONENTS: Record<string, React.FC> = {
  title: TitleScene,
  problem: ProblemScene,
  solution: SolutionScene,
  baseline: BaselineScene,
  prior: PriorScene,
  refute: RefuteScene,
  pivot: PivotScene,
  sweep: SweepScene,
  quality: QualityScene,
  victory: VictoryScene,
  repro: ReproScene,
  close: CloseScene,
};

const bandOpacity = (t: number, from: number, to: number): number =>
  Math.min(
    interpolate(t, [from, from + 0.35], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
    interpolate(t, [to - 0.35, to], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })
  );

const CaptionTrack: React.FC = () => {
  const t = useCurrentFrame() / FPS;
  const active = CAPTIONS.find((c) => t >= c.from && t <= c.to);
  if (!active) return null;
  return (
    <div className="cap" style={{ opacity: bandOpacity(t, active.from, active.to) }}>
      {active.text}
    </div>
  );
};

const ChipTrack: React.FC = () => {
  const t = useCurrentFrame() / FPS;
  const active = CHIPS.find((c) => t >= c.from && t <= c.to);
  if (!active) return null;
  return (
    <div className="chip" style={{ opacity: bandOpacity(t, active.from, active.to) }}>
      {active.text}
    </div>
  );
};

export const Demo: React.FC = () => (
  <AbsoluteFill
    style={{
      background: "#000",
      fontFamily: '"Segoe UI", Lato, Helvetica, Arial, sans-serif',
    }}
  >
    {SCENES.map((s) => {
      const Scene = SCENE_COMPONENTS[s.id];
      return (
        <Sequence key={s.id} from={sec(startOf(s.id))} durationInFrames={sec(s.dur)}>
          <Scene />
        </Sequence>
      );
    })}
    <CaptionTrack />
    <ChipTrack />
  </AbsoluteFill>
);
