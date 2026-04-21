"use client"; // Tell node.js to run compnent on browser

import { useState } from "react"; // React hook for managing state

export default function Home() {
  // State to store user's input for dream
  const [dream, setDream] = useState("");

  // Track if app is waiting for a response
  const [loading, setLoading] = useState(false);

  // State to store any error messages
  const [error, setError] = useState("");

  // State to store the API result (emotion, confidence, cluster)
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
    } catch (err: any) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: "700px", margin: "40px auto", padding: "20px" }}>
      <h1>Dream Emotion Analyzer</h1>

      <textarea
        value={dream}
        onChange={(e) => setDream(e.target.value)}
        placeholder="Type your dream here..."
        rows={8}
        style={{ width: "100%", marginTop: "20px", padding: "12px" }}
      />

      <button
        onClick={handleAnalyze}
        style={{ marginTop: "12px", padding: "10px 16px", cursor: "pointer" }}
      >
        Analyze Dream
      </button>

      {error && <p style={{ marginTop: "16px", color: "red" }}>{error}</p>}

      {loading && <p style={{ marginTop: "16px" }}>Analyzing dream...</p>}

      {result && (
        <div
          style={{
            marginTop: "24px",
            padding: "16px",
            border: "1px solid #ccc",
            borderRadius: "8px",
          }}
        >
          <p>
            <strong>You:</strong> {dream}
          </p>
          <p>
            <strong>Bot:</strong> This dream feels like <b>{result.emotion}</b>{" "}
            (confidence {result.confidence}%, cluster {result.cluster}).
          </p>
        </div>
      )}
    </main>
  );
}