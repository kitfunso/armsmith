import React from "react";
import { interpolate, useCurrentFrame } from "remotion";
import { FPS } from "./timeline";

export const useT = (): number => useCurrentFrame() / FPS;

/** Frame-driven typewriter: characters revealed at `cps` chars/second. */
export const typed = (full: string, t: number, startAt: number, cps: number): string => {
  if (t <= startAt) return "";
  return full.slice(0, Math.max(0, Math.floor((t - startAt) * cps)));
};

/** Fade wrapper for anything appearing mid-scene. Seconds are scene-relative. */
export const Appear: React.FC<{
  at: number;
  children: React.ReactNode;
  fade?: number;
  fill?: boolean;
}> = ({ at, children, fade = 0.3, fill }) => {
  const t = useT();
  const opacity = interpolate(t, [at, at + fade], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  if (opacity === 0) return null;
  const style: React.CSSProperties = fill
    ? { opacity, position: "absolute", inset: 0 }
    : { opacity };
  return <div style={style}>{children}</div>;
};

/** Fade a full-frame card in and out over the scene duration. */
export const DarkFade: React.FC<{ dur: number; children: React.ReactNode }> = ({
  dur,
  children,
}) => {
  const t = useT();
  const opacity =
    interpolate(t, [0, 0.45], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }) *
    interpolate(t, [dur - 0.45, dur], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return <div style={{ position: "absolute", inset: 0, opacity }}>{children}</div>;
};

/* ---------- dark cards ---------- */

export const BigCard: React.FC<{ big: React.ReactNode; sub?: string }> = ({ big, sub }) => (
  <div className="darkcard">
    <div className="bigline">{big}</div>
    {sub ? <div className="subline">{sub}</div> : null}
  </div>
);

export const WordmarkCard: React.FC<{
  tagline: React.ReactNode;
  url?: string;
  strip?: { value: string; label: string }[];
}> = ({ tagline, url, strip }) => (
  <div className="darkcard">
    <div className="wordmark">
      <span className="anvil">⚒</span> armsmith
    </div>
    <div className="tagline">{tagline}</div>
    {strip ? (
      <div className="numstrip">
        {strip.map((s) => (
          <div className="n" key={s.label}>
            <b>{s.value}</b>
            <span>{s.label}</span>
          </div>
        ))}
      </div>
    ) : null}
    {url ? <div className="urlline">{url}</div> : null}
  </div>
);

/* ---------- terminal ---------- */

export const Term: React.FC<{ title: string; children: React.ReactNode }> = ({
  title,
  children,
}) => (
  <div className="termwrap">
    <div className="term">
      <div className="bar">
        <i className="r" />
        <i className="y" />
        <i className="g" />
        <span>{title}</span>
      </div>
      <pre>{children}</pre>
    </div>
  </div>
);

/* ---------- analyst card ---------- */

export const AnalystCard: React.FC<{
  rationale: string;
  priority: string[];
  typedUntil?: number;
}> = ({ rationale, priority, typedUntil }) => {
  const t = useT();
  const text =
    typedUntil === undefined ? rationale : typed(rationale, t, 1.2, rationale.length / typedUntil);
  return (
    <div className="analystwrap">
      <div className="analyst">
        <div className="bar">
          ✳ Claude — Arm Performix analyst
          <span className="tag">ADVISORY</span>
        </div>
        <div className="body">
          <div className="rationale">&ldquo;{text}&rdquo;</div>
          <Appear at={typedUntil === undefined ? 0.5 : typedUntil + 1.4}>
            <div className="prio">
              {priority.map((p, i) => (
                <div className="pi" key={p}>
                  <b>{i + 1}</b> {p}
                </div>
              ))}
            </div>
            <div className="advisory">The tuner decides. The analyst only reorders the search.</div>
          </Appear>
        </div>
      </div>
    </div>
  );
};

/* ---------- result card ---------- */

export const ResultCard: React.FC<{
  action: string;
  deltaPct: number;
  detail: React.ReactNode;
  kept: boolean;
  badgeAt: number;
}> = ({ action, deltaPct, detail, kept, badgeAt }) => {
  const up = deltaPct >= 0;
  return (
    <div className="resultwrap">
      <div className="result">
        <div className="act">try: {action}</div>
        <div className={`delta ${up ? "up" : "down"}`}>
          {up ? "+" : ""}
          {deltaPct.toFixed(1)}% decode
        </div>
        <div className="detail">{detail}</div>
        <Appear at={badgeAt}>
          <span className={`badge ${kept ? "kept" : "reverted"}`}>
            {kept ? "KEPT" : "REVERTED"}
          </span>
        </Appear>
      </div>
    </div>
  );
};
