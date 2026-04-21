"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

type Cluster = "everyday" | "symbolic" | "high-emotion" | "noise";
type ViewMode = "clusters" | "salience";

type DreamPoint = {
  id: number;
  x: number;
  y: number;
  cluster: Cluster;
};

type StoryTheme = "emotions" | "mental-health" | "therapy";

const stats = [
  { label: "Dream reports", value: "~44,000" },
  { label: "Participants", value: "~16,000" },
  { label: "Average report length", value: "~100 words" },
  { label: "Primary archetype clusters", value: "3" },
];

const pipelineSteps = [
  {
    title: "Data Preparation",
    detail:
      "Dream reports were standardized, short records were removed, and Sleep & Dream data was merged with DreamBank for broader coverage.",
  },
  {
    title: "Embedding",
    detail:
      "Each narrative was encoded into a semantic vector using transformer-based embeddings so meaning, not just keywords, drives grouping.",
  },
  {
    title: "Dimensionality Reduction",
    detail:
      "Reducing dimensionality before clustering exposed fine-grained archetypes that raw embeddings alone did not surface.",
  },
  {
    title: "Density Clustering",
    detail:
      "Density-based clustering uncovered latent archetypes organically without predefined dream labels or hand-built taxonomies.",
  },
  {
    title: "Dream Universe",
    detail:
      "The resulting universe positions dreams by semantic similarity: center as everyday dream space, periphery as emotionally salient narratives.",
  },
];

const impactContent: Record<
  StoryTheme,
  {
    title: string;
    body: string;
  }
> = {
  emotions: {
    title: "Modeling Human Emotions",
    body: "Recurring themes across a large dream corpus provide a data-driven view into collective emotional patterns and subconscious processing.",
  },
  "mental-health": {
    title: "Mental Health Indicators",
    body: "Clustered dream content can surface latent markers of stress and emotional dysregulation without requiring diagnostic labels.",
  },
  therapy: {
    title: "Psychotherapeutic Support",
    body: "Archetype-level context can help clinicians interpret patient dream narratives against empirically grounded thematic structures.",
  },
};

function seededRandom(seed: number) {
  let value = seed >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) % 4294967296;
    return value / 4294967296;
  };
}

function buildDreamUniverse(): DreamPoint[] {
  const rand = seededRandom(4602);
  let id = 0;
  const points: DreamPoint[] = [];

  for (let i = 0; i < 46; i += 1) {
    const angle = rand() * Math.PI * 2;
    const radius = 4 + rand() * 16;
    points.push({
      id: id++,
      x: 50 + Math.cos(angle) * radius,
      y: 50 + Math.sin(angle) * radius,
      cluster: "everyday",
    });
  }

  for (let i = 0; i < 22; i += 1) {
    const angle = rand() * Math.PI * 2;
    const radius = 4 + rand() * 13;
    points.push({
      id: id++,
      x: 24 + Math.cos(angle) * radius,
      y: 30 + Math.sin(angle) * radius,
      cluster: "symbolic",
    });
  }

  for (let i = 0; i < 20; i += 1) {
    const angle = rand() * Math.PI * 2;
    const radius = 5 + rand() * 15;
    points.push({
      id: id++,
      x: 77 + Math.cos(angle) * radius,
      y: 33 + Math.sin(angle) * radius,
      cluster: "high-emotion",
    });
  }

  for (let i = 0; i < 18; i += 1) {
    points.push({
      id: id++,
      x: 8 + rand() * 84,
      y: 12 + rand() * 76,
      cluster: "noise",
    });
  }

  return points;
}

function clusterLabel(cluster: Cluster) {
  switch (cluster) {
    case "everyday":
      return "Everyday Dream Space";
    case "symbolic":
      return "Thematic / Symbolic";
    case "high-emotion":
      return "Emotionally Salient";
    case "noise":
      return "Noise";
    default:
      return cluster;
  }
}

export default function Home() {
  const [viewMode, setViewMode] = useState<ViewMode>("clusters");
  const [focusCluster, setFocusCluster] = useState<Cluster | "all">("all");
  const [activeStep, setActiveStep] = useState(pipelineSteps[0].title);
  const [activeTheme, setActiveTheme] = useState<StoryTheme>("emotions");

  const points = useMemo(() => buildDreamUniverse(), []);

  const stepDetail =
    pipelineSteps.find((step) => step.title === activeStep) ?? pipelineSteps[0];

  return (
    <main className="home-shell">
      <div className="home-aurora" aria-hidden="true">
        <span className="home-orb home-orb-one" />
        <span className="home-orb home-orb-two" />
        <span className="home-orb home-orb-three" />
      </div>
      <div className="home-grid" aria-hidden="true" />

      <div className="home-container">
        <header className="home-topbar">
          <p className="home-badge">DTSC 4602 Data Science Project</p>
          <Link href="/analyze" className="home-analyze-link">
            Go to Dream Input
          </Link>
        </header>

        <section className="hero-card">
          <p className="hero-kicker">DreamCatcher</p>
          <h1>Uncovering Latent Dream Archetypes with Unsupervised NLP</h1>
          <p>
            We built DreamCatcher to study how large-scale dream narratives can be
            grouped into meaningful archetypes using transformer embeddings,
            dimensionality reduction, and density-based clustering.
          </p>
          <p>
            Team: Soumil Kothari, Michael Stelmack, Garrett Swaney, Ahmad Jebril,
            Nick Greco, Aryaman Kachroo, Aidan Thomas. Adviser: Mirsad Hadzikadic.
          </p>
          <div className="hero-actions">
            <Link href="/analyze" className="hero-cta-primary">
              Start Dream Analysis
            </Link>
            <a href="#why-it-matters" className="hero-cta-secondary">
              Why It Matters
            </a>
          </div>
        </section>

        <section className="stats-grid" aria-label="Key project metrics">
          {stats.map((item) => (
            <article className="stat-card" key={item.label}>
              <p className="stat-value">{item.value}</p>
              <p className="stat-label">{item.label}</p>
            </article>
          ))}
        </section>

        <section className="story-grid" id="why-it-matters">
          <article className="story-card">
            <h2>Why We Did This</h2>
            <p>
              Traditional dream coding is powerful but labor-intensive and rigid.
              Existing frameworks often miss semantic nuance at scale. We wanted a
              data-first way to discover archetypes directly from dream meaning.
            </p>
          </article>
          <article className="story-card">
            <h2>What We Found</h2>
            <p>
              Initial clustering on raw embeddings showed only dominant structure.
              Applying dimensionality reduction first revealed finer archetypes and
              converged to three primary clusters with peripheral, high-salience
              regions.
            </p>
          </article>
          <article className="story-card">
            <h2>Why It Matters</h2>
            <p>
              Dream archetypes can help model emotional patterns, support mental
              health signal discovery, and provide clinicians richer context for
              patient narratives.
            </p>
          </article>
        </section>

        <section className="viz-grid" aria-label="Interactive project visuals">
          <article className="viz-card universe-card">
            <div className="viz-head">
              <h2>Interactive Dream Universe</h2>
              <p>
                Inspired by our 2D/3D universe figures: each point is one dream;
                distance encodes semantic similarity.
              </p>
            </div>

            <div className="viz-controls">
              <div className="control-group">
                <span>View:</span>
                <button
                  className={viewMode === "clusters" ? "viz-pill active" : "viz-pill"}
                  onClick={() => setViewMode("clusters")}
                >
                  Clusters
                </button>
                <button
                  className={viewMode === "salience" ? "viz-pill active" : "viz-pill"}
                  onClick={() => setViewMode("salience")}
                >
                  Emotional Salience
                </button>
              </div>

              <div className="control-group">
                <span>Focus:</span>
                {([
                  "all",
                  "everyday",
                  "symbolic",
                  "high-emotion",
                  "noise",
                ] as const).map((cluster) => (
                  <button
                    key={cluster}
                    className={focusCluster === cluster ? "viz-pill active" : "viz-pill"}
                    onClick={() => setFocusCluster(cluster)}
                  >
                    {cluster === "all" ? "All" : clusterLabel(cluster)}
                  </button>
                ))}
              </div>
            </div>

            <div className="universe-frame">
              <svg viewBox="0 0 100 100" className="universe-svg" role="img">
                <title>Dream universe projection</title>
                <circle cx="50" cy="50" r="21" className="universe-center-ring" />
                {points.map((point) => {
                  const distance = Math.hypot(point.x - 50, point.y - 50);
                  const isHidden =
                    focusCluster !== "all" && focusCluster !== point.cluster;

                  let fill = "#c8b4ff";
                  if (viewMode === "clusters") {
                    if (point.cluster === "symbolic") {
                      fill = "#9f7dff";
                    }
                    if (point.cluster === "high-emotion") {
                      fill = "#f18cff";
                    }
                    if (point.cluster === "noise") {
                      fill = "#41295f";
                    }
                  } else {
                    const intensity = Math.min(1, distance / 45);
                    fill = `hsl(${286 + intensity * 45} 95% ${77 - intensity * 30}%)`;
                  }

                  const radius = point.cluster === "noise" ? 0.8 : 1.15;

                  return (
                    <circle
                      key={point.id}
                      cx={point.x}
                      cy={point.y}
                      r={radius}
                      fill={fill}
                      opacity={isHidden ? 0.1 : point.cluster === "noise" ? 0.42 : 0.93}
                    />
                  );
                })}
              </svg>
            </div>

            <p className="viz-footnote">
              Dark purple points represent noise. The center captures everyday
              narratives, while the periphery tends toward more emotionally salient
              themes.
            </p>
          </article>

          <article className="viz-card pipeline-card">
            <h2>Method Explorer</h2>
            <p>
              Click each stage to see how the DreamCatcher pipeline moves from raw
              narratives to emergent archetypes.
            </p>
            <div className="pipeline-buttons">
              {pipelineSteps.map((step) => (
                <button
                  key={step.title}
                  className={activeStep === step.title ? "pipeline-step active" : "pipeline-step"}
                  onClick={() => setActiveStep(step.title)}
                >
                  {step.title}
                </button>
              ))}
            </div>
            <div className="pipeline-detail">
              <h3>{stepDetail.title}</h3>
              <p>{stepDetail.detail}</p>
            </div>

            <div className="impact-panel">
              <h3>Impact Lens</h3>
              <div className="impact-tabs">
                <button
                  className={activeTheme === "emotions" ? "impact-tab active" : "impact-tab"}
                  onClick={() => setActiveTheme("emotions")}
                >
                  Human Emotions
                </button>
                <button
                  className={
                    activeTheme === "mental-health" ? "impact-tab active" : "impact-tab"
                  }
                  onClick={() => setActiveTheme("mental-health")}
                >
                  Mental Health
                </button>
                <button
                  className={activeTheme === "therapy" ? "impact-tab active" : "impact-tab"}
                  onClick={() => setActiveTheme("therapy")}
                >
                  Psychotherapy
                </button>
              </div>
              <p className="impact-title">{impactContent[activeTheme].title}</p>
              <p>{impactContent[activeTheme].body}</p>
            </div>
          </article>
        </section>

        <section className="home-bottom-cta">
          <h2>Ready to Analyze Your Own Dream?</h2>
          <p>
            Go to the Dream Input page and test the analyzer interface built for our
            DTSC 4602 project demo.
          </p>
          <Link href="/analyze" className="hero-cta-primary">
            Open Dream Input Page
          </Link>
        </section>
      </div>
    </main>
  );
}