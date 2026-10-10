"""Food for thought: score replies from OUTSIDE the dataset with the main model, and explain the result.

The model is TF-IDF + LR on masked outputs (output only), exactly as reported: it is refit with the C
chosen on validation (results/baselines_runs.csv) and checked to reproduce the reported test accuracy.
The fitted model is cached in models/ (git-ignored) so later runs only load it.

Replies are listed in data/external_replies/replies.csv (file, question, true_source, topic_terms).

Usage (from the project root, after scripts/baselines.py):
    python scripts/score_reply.py

Outputs:
    results/food_for_thought.csv            probability per family for each reply
    results/food_for_thought_style.csv      each reply's style features next to the family profiles
    results/food_for_thought_features.csv   features that pushed the prediction toward the predicted vs true family
    results/food_for_thought_checks.csv     training-data checks (topic coverage, similar-looking true-family outputs)
"""

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]  # project folder (parent of scripts/)
sys.path.insert(0, str(ROOT))  # so "from src... / from scripts... import" works when run as a script
from scripts.baselines import tfidf_vectorizers  # noqa: E402
from src.data import FAMILIES, load_data, load_splits, mask_names  # noqa: E402
from src.transforms import style_features  # noqa: E402

REPLIES = ROOT / "data" / "external_replies"
MODEL_FILE = ROOT / "models" / "tfidf_lr_output_only.pkl"
TOP_K = 10
# Outputs with any markdown header, bold span or emoji (used to find plain-looking training outputs).
FORMATTED = re.compile("(^|\n)\\s*#|\\*\\*|[\U0001F300-\U0001FAFF☀-➿]")
LONG_WORDS = 500


def get_model(data):
    """Load the cached model, or refit it exactly as in scripts/baselines.py (deterministic)."""
    # pickle is safe here only because this script itself wrote the file, locally (models/ is git-ignored).
    # Never load a model file obtained from someone else this way.
    if MODEL_FILE.exists():
        return pickle.loads(MODEL_FILE.read_bytes())
    runs = pd.read_csv(ROOT / "results" / "baselines_runs.csv")
    best_c = runs.query("baseline == 'tfidf_lr' and condition == 'output_only'")["best_C"].iloc[0]
    texts = data["train"]["LLM_output_masked"]
    word, char = (vectorizer.fit(texts) for vectorizer in tfidf_vectorizers())
    features = hstack([word.transform(texts), char.transform(texts)]).tocsr()
    lr = LogisticRegression(C=best_c, max_iter=5000, random_state=0).fit(features, data["train"]["LLM_name"])
    MODEL_FILE.parent.mkdir(exist_ok=True)
    MODEL_FILE.write_bytes(pickle.dumps((word, char, lr)))
    return word, char, lr


def main():
    df = load_data()
    data = load_splits(df)
    word, char, lr = get_model(data)

    def featurize(texts):
        return hstack([word.transform(texts), char.transform(texts)]).tocsr()

    # The model must be the reported one.
    test = data["test"]
    acc = (lr.predict(featurize(test["LLM_output_masked"])) == test["LLM_name"]).mean()
    reported = pd.read_csv(ROOT / "results" / "baselines.csv").query(
        "baseline == 'tfidf_lr' and condition == 'output_only'")["test_acc_mean"].iloc[0]
    assert round(acc, 4) == reported, f"model test accuracy {acc:.4f} != reported {reported}"
    print(f"Model check: test accuracy {acc:.4f} = reported {reported}")

    manifest = pd.read_csv(REPLIES / "replies.csv")
    classes = list(lr.classes_)
    n_word = len(word.vocabulary_)
    names = np.concatenate([word.get_feature_names_out(), ["char:" + repr(c) for c in char.get_feature_names_out()]])
    profiles = pd.read_csv(ROOT / "results" / "rq4_features.csv").set_index("LLM_name")
    prob_rows, style_rows, feature_rows, check_rows = [], [], [], []

    for _, item in manifest.iterrows():
        reply = mask_names((REPLIES / item["file"]).read_text(encoding="utf-8").strip())
        x = featurize([reply])
        probs = dict(zip(classes, lr.predict_proba(x)[0]))
        ranked = sorted(probs, key=probs.get, reverse=True)
        truth = item["true_source"]
        prob_rows.append({"file": item["file"], "true_source": truth, "predicted": ranked[0],
                          "correct": ranked[0] == truth if truth in FAMILIES else None,
                          "true_rank": ranked.index(truth) + 1 if truth in FAMILIES else None,
                          **{f"p_{f}": probs[f] for f in FAMILIES}})

        # Style of this reply, in the same units as results/rq4_features.csv.
        s = style_features(reply)
        words = s["words"]
        style_rows.append({"who": item["file"], "words": words, "ttr_200": s["ttr_200"],
                           "mean_sentence_len": s["mean_sentence_len"],
                           "headers_per_1k": s["headers"] / words * 1000, "bullets_per_1k": s["bullets"] / words * 1000,
                           "bold_per_1k": s["bold"] / words * 1000, "emoji_per_1k": s["emoji"] / words * 1000})

        # Why the predicted family beat the true one: contribution = feature value x (w_pred - w_true).
        if truth in FAMILIES and ranked[0] != truth:
            gap = lr.coef_[classes.index(ranked[0])] - lr.coef_[classes.index(truth)]
            contrib = x.multiply(gap).toarray()[0]
            check_rows += [
                {"file": item["file"], "check": f"score gap {ranked[0]} - {truth}, word features", "value": contrib[:n_word].sum()},
                {"file": item["file"], "check": f"score gap {ranked[0]} - {truth}, char features", "value": contrib[n_word:].sum()},
            ]
            order = np.argsort(contrib)
            for direction, idx in [(ranked[0], order[::-1][:TOP_K]), (truth, order[:TOP_K])]:
                feature_rows += [{"file": item["file"], "pushes_toward": direction, "rank": r + 1,
                                  "feature": names[i], "contribution": contrib[i]} for r, i in enumerate(idx)]

        # Training-data checks: topic coverage, and true-family outputs that look like this reply.
        train = data["train"]
        for term in str(item["topic_terms"]).split(";"):
            hit = train["LLM_output_masked"].str.contains(term, case=False, regex=False)
            check_rows.append({"file": item["file"], "check": f"train outputs containing '{term}'", "value": int(hit.sum())})
            for family, n in train[hit]["LLM_name"].value_counts().items():
                check_rows.append({"file": item["file"], "check": f"train outputs containing '{term}' from {family}", "value": int(n)})
        if truth in FAMILIES:
            fam = df[df["LLM_name"] == truth]["LLM_output"]
            plain_long = (~fam.map(lambda t: bool(FORMATTED.search(t)))) & (fam.str.split().str.len() >= LONG_WORDS)
            check_rows += [
                {"file": item["file"], "check": f"{truth} outputs (all {len(fam)}) with no headers/bold/emoji and >= {LONG_WORDS} words",
                 "value": int(plain_long.sum())},
                {"file": item["file"], "check": f"{truth} outputs in the dataset", "value": len(fam)},
            ]

    out = ROOT / "results"
    pd.DataFrame(prob_rows).round(4).to_csv(out / "food_for_thought.csv", index=False)
    style = pd.concat([profiles.rename(columns={"median_words": "words"})[list(style_rows[0])[1:]]
                       .rename_axis("who").reset_index(), pd.DataFrame(style_rows)], ignore_index=True)
    style.round(3).to_csv(out / "food_for_thought_style.csv", index=False)
    pd.DataFrame(feature_rows).round(4).to_csv(out / "food_for_thought_features.csv", index=False)
    pd.DataFrame(check_rows).round(4).to_csv(out / "food_for_thought_checks.csv", index=False)

    print("\nProbabilities (results/food_for_thought.csv):")
    print(pd.DataFrame(prob_rows).round(3).to_string(index=False))
    print("\nStyle vs family profiles (results/food_for_thought_style.csv; family rows use the median word count):")
    print(style.round(2).to_string(index=False))
    print("\nChecks (results/food_for_thought_checks.csv):")
    print(pd.DataFrame(check_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
