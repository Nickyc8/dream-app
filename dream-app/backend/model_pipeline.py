"""
DreamCatcher model pipeline extracted from the notebook.

What this file does:
- loads dreamsearch.csv
- cleans dream text
- creates sentence embeddings
- fits UMAP reducers (10D for clustering, 2D and 3D for visualization)
- fits HDBSCAN with prediction_data=True so new dreams can be assigned later
- stores archetype labels
- predicts a cluster/archetype for a new dream
- returns 3D coordinates for plotting the new dream on the existing dream universe

Recommended project layout:
backend/
  model_pipeline.py   <- this file
  artifacts/          <- saved joblib artifacts
  Data/dreamsearch.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import re

import joblib
import numpy as np
import pandas as pd
import umap
import hdbscan

from hdbscan.prediction import approximate_predict
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# -----------------------------
# Configuration
# -----------------------------

MODEL_NAME = "all-MiniLM-L6-v2"
MIN_WORDS = 20

# Archetype names taken from the notebook.
ARCHETYPE_NAMES = {
    46: "Grief & Loss",
    42: "Overwhelm / Control Loss",
    78: "Identity & Exposure",
}

COLOR_MAP = {
    "Other / Unlabeled": "lightgray",
    "Overwhelm / Control Loss": "red",
    "Identity & Exposure": "green",
    "Grief & Loss": "purple",
}

EMOTION_LEXICON = {
    "Fear": {
        "afraid",
        "alarm",
        "anxious",
        "chase",
        "chased",
        "danger",
        "dangerous",
        "dark",
        "die",
        "died",
        "escape",
        "fear",
        "fearful",
        "frightened",
        "hide",
        "hiding",
        "lost",
        "monster",
        "panic",
        "run",
        "running",
        "scared",
        "scream",
        "screaming",
        "threat",
        "trapped",
        "worry",
        "worried",
    },
    "Sadness": {
        "alone",
        "cry",
        "crying",
        "dead",
        "death",
        "despair",
        "grief",
        "guilt",
        "hurt",
        "lonely",
        "loss",
        "lost",
        "miss",
        "missing",
        "mourning",
        "sad",
        "sadness",
        "shame",
        "tears",
        "unhappy",
    },
    "Anger": {
        "angry",
        "argue",
        "argued",
        "attack",
        "attacked",
        "fight",
        "fighting",
        "furious",
        "hate",
        "hit",
        "mad",
        "rage",
        "shout",
        "shouting",
        "yell",
        "yelling",
    },
    "Joy": {
        "beautiful",
        "calm",
        "comfort",
        "comfortable",
        "delight",
        "excited",
        "free",
        "friend",
        "fun",
        "glad",
        "happy",
        "hope",
        "joy",
        "laugh",
        "love",
        "peace",
        "peaceful",
        "pleasant",
        "relief",
        "safe",
        "wonderful",
    },
    "Confusion": {
        "confused",
        "forgot",
        "forgotten",
        "strange",
        "unknown",
        "weird",
        "where",
        "why",
    },
    "Disgust": {
        "blood",
        "dirty",
        "disease",
        "gross",
        "ill",
        "sick",
        "smell",
        "vomit",
        "waste",
    },
}

THEME_LEXICON = {
    "Being chased or threatened": {
        "attack",
        "attacked",
        "chase",
        "chased",
        "danger",
        "escape",
        "hide",
        "hiding",
        "run",
        "running",
        "threat",
        "trapped",
    },
    "Falling, flying, or movement": {
        "climb",
        "climbing",
        "fall",
        "falling",
        "flew",
        "flight",
        "float",
        "floating",
        "fly",
        "flying",
        "jump",
    },
    "Family and relationships": {
        "baby",
        "brother",
        "child",
        "children",
        "dad",
        "father",
        "friend",
        "friends",
        "girl",
        "husband",
        "mom",
        "mother",
        "partner",
        "sister",
        "wife",
    },
    "Home, rooms, and familiar places": {
        "apartment",
        "bed",
        "bedroom",
        "door",
        "home",
        "house",
        "kitchen",
        "room",
        "school",
        "stairs",
        "window",
        "work",
    },
    "Loss, grief, or separation": {
        "alone",
        "cry",
        "crying",
        "dead",
        "death",
        "died",
        "grief",
        "left",
        "loss",
        "lost",
        "missing",
        "sad",
        "tears",
    },
    "Identity, exposure, or judgment": {
        "ashamed",
        "clothes",
        "embarrassed",
        "exam",
        "exposed",
        "forgot",
        "naked",
        "presentation",
        "shame",
        "watched",
    },
    "Control, pressure, or responsibility": {
        "busy",
        "control",
        "late",
        "pressure",
        "responsible",
        "stuck",
        "task",
        "test",
        "work",
    },
    "Water, nature, or animals": {
        "animal",
        "beach",
        "bird",
        "cat",
        "dog",
        "forest",
        "lake",
        "mountain",
        "ocean",
        "river",
        "snake",
        "tree",
        "water",
        "woods",
    },
    "Travel, vehicles, or transition": {
        "airport",
        "bus",
        "car",
        "drive",
        "driving",
        "road",
        "train",
        "travel",
        "trip",
        "vehicle",
    },
}

GENERIC_CLUSTER_KEYWORDS = {
    "asked",
    "back",
    "dream",
    "felt",
    "get",
    "going",
    "like",
    "many",
    "one",
    "ordinary",
    "people",
    "remember",
    "said",
    "saw",
    "see",
    "something",
}


def get_stopwords() -> set[str]:
    """Load English stopwords from NLTK."""
    return set(stopwords.words("english"))


STOPWORDS = None  # lazy-loaded


def clean_text(text: str) -> str:
    """Match the notebook's cleaning logic as closely as possible."""
    global STOPWORDS
    if STOPWORDS is None:
        STOPWORDS = get_stopwords()

    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)


def shorten_text(text: str, max_len: int = 120) -> str:
    text = str(text).replace("\n", " ")
    return text[:max_len] + "..." if len(text) > max_len else text


def score_lexicon(cleaned_text: str, lexicon: dict[str, set[str]], limit: int | None = None) -> list[dict[str, Any]]:
    tokens = cleaned_text.split()
    token_counts = {token: tokens.count(token) for token in set(tokens)}
    scored: list[dict[str, Any]] = []

    for label, words in lexicon.items():
        matches = sorted(word for word in words if word in token_counts)
        score = sum(token_counts[word] for word in matches)
        if score > 0:
            scored.append({"label": label, "score": score, "matches": matches[:5]})

    scored.sort(key=lambda item: (-item["score"], item["label"]))
    return scored[:limit] if limit else scored


def infer_emotion(cleaned_text: str) -> dict[str, Any]:
    scores = score_lexicon(cleaned_text, EMOTION_LEXICON)
    if not scores:
        return {
            "label": "Mixed / reflective",
            "confidence": 0.35,
            "signals": [],
        }

    top = scores[0]
    total = sum(item["score"] for item in scores)
    confidence = top["score"] / total if total else 0.35

    return {
        "label": top["label"],
        "confidence": round(float(confidence), 2),
        "signals": top["matches"],
    }


def infer_themes(cleaned_text: str, fallback_keywords: list[str]) -> list[str]:
    theme_scores = score_lexicon(cleaned_text, THEME_LEXICON, limit=4)
    themes = [item["label"] for item in theme_scores]

    for keyword in fallback_keywords:
        if keyword.lower() in GENERIC_CLUSTER_KEYWORDS:
            continue
        keyword_label = keyword.replace("_", " ").strip().title()
        if keyword_label and keyword_label not in themes:
            themes.append(keyword_label)
        if len(themes) >= 4:
            break

    if not themes:
        themes.append("Mixed dream imagery")

    return themes


@dataclass
class DreamArtifacts:
    """In-memory objects needed for inference and plotting."""
    embed_model: SentenceTransformer
    reducer_10d: Any
    reducer_2d: Any
    reducer_3d: Any
    clusterer: hdbscan.HDBSCAN
    plot_df_3d: pd.DataFrame
    cluster_keywords: dict[int, list[str]]
    archetype_names: dict[int, str]


class DreamPipeline:
    def __init__(
        self,
        data_path: str | Path | None = None,
        artifacts_dir: str | Path | None = None,
        model_name: str = MODEL_NAME,
        min_words: int = MIN_WORDS,
    ) -> None:
        self.data_path = Path(data_path) if data_path else None
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self.model_name = model_name
        self.min_words = min_words
        self.artifacts: DreamArtifacts | None = None

    # -----------------------------
    # Training / artifact creation
    # -----------------------------
    def build_from_csv(self) -> DreamArtifacts:
        """
        Rebuild the pipeline from the CSV.
        Use this once, then save artifacts. For demos, load the saved artifacts
        instead of rebuilding on every start.
        """
        if not self.data_path:
            raise ValueError("data_path is required to build artifacts from CSV")

        df = pd.read_csv(self.data_path)

        model_cols = [
            "Dream Report ID",
            "Dream Text",
            "Word Count",
            "Participant ID",
            "Dream Date",
            "Age",
            "Gender",
        ]
        df_model = df[model_cols].copy()
        df_model = df_model.rename(
            columns={
                "Dream Report ID": "dream_id",
                "Dream Text": "text",
                "Word Count": "word_count",
                "Participant ID": "participant_id",
                "Dream Date": "date",
            }
        )

        df_model = df_model.dropna(subset=["text"]).copy()
        df_model = df_model[df_model["word_count"] >= self.min_words].copy()
        df_model["clean_text"] = df_model["text"].astype(str).apply(clean_text)
        df_model["clean_len"] = df_model["clean_text"].apply(lambda x: len(x.split()))
        df_model["hover_text"] = df_model["text"].apply(shorten_text)

        embed_model = SentenceTransformer(self.model_name)
        texts = df_model["clean_text"].tolist()

        embeddings = embed_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = normalize(embeddings)

        # 10D reducer used in the notebook before HDBSCAN clustering
        reducer_10d = umap.UMAP(
            n_neighbors=20,
            n_components=10,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        embeddings_10d = reducer_10d.fit_transform(embeddings)

        # HDBSCAN for cluster assignment on reduced space.
        # prediction_data=True is important so approximate_predict works for new dreams.
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=25,
            min_samples=5,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        labels = clusterer.fit_predict(embeddings_10d)
        df_model["cluster_embed"] = labels

        # 2D visualization from notebook
        reducer_2d = umap.UMAP(
            n_neighbors=20,
            n_components=2,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        umap_2d = reducer_2d.fit_transform(embeddings)
        df_model["x"] = umap_2d[:, 0]
        df_model["y"] = umap_2d[:, 1]

        # 3D visualization from notebook
        reducer_3d = umap.UMAP(
            n_neighbors=20,
            n_components=3,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        umap_3d = reducer_3d.fit_transform(embeddings)
        df_model["x3"] = umap_3d[:, 0]
        df_model["y3"] = umap_3d[:, 1]
        df_model["z3"] = umap_3d[:, 2]

        # Archetype names from notebook
        df_model["archetype_name"] = df_model["cluster_embed"].map(ARCHETYPE_NAMES)
        df_model["archetype_name"] = df_model["archetype_name"].fillna("Other / Unlabeled")

        # Precompute keywords for top named clusters
        keyword_lookup = self._build_cluster_keyword_lookup(df_model)

        plot_df_3d = df_model[
            ["x3", "y3", "z3", "cluster_embed", "archetype_name", "hover_text"]
        ].copy()

        artifacts = DreamArtifacts(
            embed_model=embed_model,
            reducer_10d=reducer_10d,
            reducer_2d=reducer_2d,
            reducer_3d=reducer_3d,
            clusterer=clusterer,
            plot_df_3d=plot_df_3d,
            cluster_keywords=keyword_lookup,
            archetype_names=ARCHETYPE_NAMES.copy(),
        )
        self.artifacts = artifacts
        return artifacts

    def _build_cluster_keyword_lookup(self, df_model: pd.DataFrame, n: int = 12) -> dict[int, list[str]]:
        keyword_lookup: dict[int, list[str]] = {}
        for cluster_id in sorted(df_model["cluster_embed"].unique()):
            if cluster_id == -1:
                continue
            texts = df_model.loc[df_model["cluster_embed"] == cluster_id, "clean_text"]
            if texts.empty:
                continue
            vec = TfidfVectorizer(max_features=5000)
            X = vec.fit_transform(texts)
            means = np.asarray(X.mean(axis=0)).ravel()
            vocab = np.array(vec.get_feature_names_out())
            keyword_lookup[int(cluster_id)] = vocab[means.argsort()[::-1][:n]].tolist()
        return keyword_lookup

    def save_artifacts(self, artifacts_dir: str | Path | None = None) -> None:
        if self.artifacts is None:
            raise ValueError("No artifacts available. Run build_from_csv() first.")

        target = Path(artifacts_dir) if artifacts_dir else self.artifacts_dir
        if target is None:
            raise ValueError("artifacts_dir is required")
        target.mkdir(parents=True, exist_ok=True)

        # SentenceTransformer can be saved separately using its own method.
        model_dir = target / "sentence_transformer"
        self.artifacts.embed_model.save(str(model_dir))

        joblib.dump(self.artifacts.reducer_10d, target / "reducer_10d.joblib")
        joblib.dump(self.artifacts.reducer_2d, target / "reducer_2d.joblib")
        joblib.dump(self.artifacts.reducer_3d, target / "reducer_3d.joblib")
        joblib.dump(self.artifacts.clusterer, target / "clusterer.joblib")
        self.artifacts.plot_df_3d.to_parquet(target / "plot_df_3d.parquet", index=False)

        metadata = {
            "archetype_names": self.artifacts.archetype_names,
            "cluster_keywords": self.artifacts.cluster_keywords,
            "color_map": COLOR_MAP,
        }
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def load_artifacts(self, artifacts_dir: str | Path | None = None) -> DreamArtifacts:
        target = Path(artifacts_dir) if artifacts_dir else self.artifacts_dir
        if target is None:
            raise ValueError("artifacts_dir is required")

        embed_model = SentenceTransformer(str(target / "sentence_transformer"))
        reducer_10d = joblib.load(target / "reducer_10d.joblib")
        reducer_2d = joblib.load(target / "reducer_2d.joblib")
        reducer_3d = joblib.load(target / "reducer_3d.joblib")
        clusterer = joblib.load(target / "clusterer.joblib")
        plot_df_3d = pd.read_parquet(target / "plot_df_3d.parquet")
        metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))

        artifacts = DreamArtifacts(
            embed_model=embed_model,
            reducer_10d=reducer_10d,
            reducer_2d=reducer_2d,
            reducer_3d=reducer_3d,
            clusterer=clusterer,
            plot_df_3d=plot_df_3d,
            cluster_keywords={int(k): v for k, v in metadata["cluster_keywords"].items()},
            archetype_names={int(k): v for k, v in metadata["archetype_names"].items()},
        )
        self.artifacts = artifacts
        return artifacts

    # -----------------------------
    # Inference
    # -----------------------------
    def predict(self, dream_text: str) -> dict[str, Any]:
        """Predict cluster/archetype and return coordinates for plotting."""
        if self.artifacts is None:
            raise ValueError("Artifacts are not loaded. Call load_artifacts() first.")

        cleaned = clean_text(dream_text)
        if not cleaned.strip():
            raise ValueError("Dream text is empty after cleaning.")

        embedding = self.artifacts.embed_model.encode(
            [cleaned],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embedding = normalize(embedding)

        embed_10d = self.artifacts.reducer_10d.transform(embedding)
        labels, strengths = approximate_predict(self.artifacts.clusterer, embed_10d)
        cluster_id = int(labels[0])
        confidence = float(strengths[0])

        xy = self.artifacts.reducer_2d.transform(embedding)
        xyz = self.artifacts.reducer_3d.transform(embedding)

        archetype_name = self.artifacts.archetype_names.get(cluster_id, "Other / Unlabeled")
        keywords = self.artifacts.cluster_keywords.get(cluster_id, [])
        emotion = infer_emotion(cleaned)
        themes = infer_themes(cleaned, keywords)

        return {
            "input_text": dream_text,
            "clean_text": cleaned,
            "cluster": cluster_id,
            "confidence": confidence,
            "archetype_name": archetype_name,
            "keywords": keywords,
            "emotion": emotion,
            "themes": themes,
            "plot_point_2d": {
                "x": float(xy[0, 0]),
                "y": float(xy[0, 1]),
            },
            "plot_point_3d": {
                "x": float(xyz[0, 0]),
                "y": float(xyz[0, 1]),
                "z": float(xyz[0, 2]),
            },
        }

    # -----------------------------
    # Plot helpers
    # -----------------------------
    def get_plot_data_3d(self, sample_size: int | None = 5000) -> list[dict[str, Any]]:
        """
        Returns the base 3D dream universe points for plotting on the frontend.
        For performance, default to a sample instead of every point.
        """
        if self.artifacts is None:
            raise ValueError("Artifacts are not loaded. Call load_artifacts() first.")

        df = self.artifacts.plot_df_3d
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        return df.to_dict(orient="records")


if __name__ == "__main__":
    # Example usage:
    #
    # 1) Build artifacts once:
    # pipeline = DreamPipeline(data_path="Data/dreamsearch.csv", artifacts_dir="artifacts")
    # pipeline.build_from_csv()
    # pipeline.save_artifacts()
    #
    # 2) Load artifacts later for fast startup:
    pipeline = DreamPipeline(artifacts_dir="artifacts")
    pipeline.load_artifacts()

    result = pipeline.predict("I was running through a school hallway and could not find my class.")
    print(json.dumps(result, indent=2))
