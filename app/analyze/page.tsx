"use client";

import Link from "next/link";
import { useState } from "react";

export default function AnalyzePage() {
  const [dream, setDream] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<null | {
    emotion: string;
    confidence: number;
    cluster: number;
  }>(null);

  async function handleAnalyze() {
    if (!dream.trim()) {
      setError("Please enter a dream first.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError("");

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ dreamText: dream }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Request failed");
      }

      setResult(data);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Something went wrong.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="dream-shell">
      <div className="dream-aurora" aria-hidden="true">
        <span className="orb orb-one" />
        <span className="orb orb-two" />
        <span className="orb orb-three" />
      </div>
      <div className="dream-grid" aria-hidden="true" />

      <section className="dream-card">
        <header className="dream-card-top">
          <p className="dream-kicker">Dream Signal Reader</p>
          <Link href="/" className="dream-back-link">
            Back to Homepage
          </Link>
        </header>

        <h1>Dream Emotion Analyzer</h1>
        <p className="dream-subtitle">
          Share a dream and get a fast emotional read from the analyzer.
        </p>

        <textarea
          value={dream}
          onChange={(e) => setDream(e.target.value)}
          placeholder="Type your dream here..."
          rows={8}
          className="dream-input"
        />

        <button onClick={handleAnalyze} className="dream-button" disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Dream"}
        </button>

        {error && <p className="dream-error">{error}</p>}

        {!error && loading && <p className="dream-loading">Analyzing dream...</p>}

        {result && (
          <div className="dream-result">
            <p className="dream-result-label">You</p>
            <p className="dream-result-copy">{dream}</p>
            <p className="dream-result-label">Bot</p>
            <p className="dream-result-copy">
              This dream feels like <b>{result.emotion}</b> (confidence {" "}
              {result.confidence}%, cluster {result.cluster}).
            </p>
          </div>
        )}
      </section>
    </main>
  );
}