# imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import tqdm
import pandas as pd
from sklearn.manifold import TSNE

PROBLEMS_FOLDER = "./problem_output"
EMBEDDINGS_FOLDER = "./problem_embeddings"
OUTPUT_FOLDER = "./embedding_graphs"

def load_pickle(path):
    """Safely load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

def euclidean_similarity(vec1, vec2):
    """Compute inverted Euclidean similarity (larger = more similar)."""
    # distance = np.linalg.norm(vec1 - vec2)
    distance = vec1.transpose().dot(vec2)
    return distance

def cosine_similarity(vec1, vec2, Rhx0):
    """Compute cosine similarity between the difference between two vectors and a given angle, Rhx0):"""
    diff = vec1 - vec2
    similarity = 1 - cosine(Rhx0, diff)
    return similarity

def plot_RH_map(Rh_scores, sample2score, best_score_text, output_path):
    """
    Creates a 2D TSNE plot of Rh scores, color-coded by whether the sample2score
    is above or below 0.5. Rhx0 is shown as a red start.
    """

    # Extract embedding matrix and labels
    texts = []
    vectors = []
    colors = []
    labels = []

    for sample, emb in Rh_scores.items():
        if not isinstance(emb, np.ndarray):
            continue
        texts.append(sample)
        vectors.append(emb)

        # Assign color/label
        if sample == best_score_text:
            labels.append("best score")
            colors.append("red")
        elif sample in sample2score:
            score = sample2score[sample]
            if score >= 0.5:
                labels.append("score >= 0.5")
                colors.append("orange")
            else:
                labels.append("score < 0.5")
                colors.append("blue")
        else:
            labels.append("unscored")
            colors.append("gray")

    if not vectors:
        print(f"No valid embeddings to plot at {output_path}")
        return

    X = np.vstack(vectors)

    # Dimensionality reduction to 2D
    reducer = TSNE(n_components=2, random_state=42, init="random", perplexity=20)
    X_embedded = reducer.fit_transform(X)

    # Plot
    plt.figure(figsize=(8, 6))
    for i, (x, y) in enumerate(X_embedded):
        plt.scatter(x, y, color=colors[i], alpha=0.7, label=labels[i] if labels[i] == "hypothesis" else "")
        if labels[i] == "hypothesis":
            plt.text(x + 0.01, y + 0.01, "HYPOTHESIS", color="red", fontsize=9, weight="bold")

    plt.title("Rh Map (TSNE)")
    plt.axis("off")

    # Create a legend (unique labels only)
    unique_labels = list(dict.fromkeys(labels))
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          label=l, markerfacecolor=c, markersize=8)
               for l, c in zip(unique_labels, dict(zip(unique_labels, colors)).values())]
    plt.legend(handles=handles, loc="best")
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    avg_similarities = {}

    embedding_files = [
        f for f in os.listdir(EMBEDDINGS_FOLDER) if f.endswith(".pkl")
    ]

    for emb_filename in tqdm.tqdm(embedding_files):
        base_name = os.path.splitext(emb_filename)[0]
        emb_path = os.path.join(EMBEDDINGS_FOLDER, emb_filename)
        problem_path = os.path.join(PROBLEMS_FOLDER, emb_filename)

        # Ensure corresponding problem file exists
        if not os.path.exists(problem_path):
            print(f"Skipping {emb_filename}: no matching problem file found.")
            continue

        # Load data
        try:
            problem_data = load_pickle(problem_path)
            embedding_data = load_pickle(emb_path)
        except Exception as e:
            print(f"Error loading {emb_filename}: {e}")
            continue

        # Create folder for this problem's graphs
        out_dir = os.path.join(OUTPUT_FOLDER, base_name)
        os.makedirs(out_dir, exist_ok=True)

        high_similarities = []
        low_similarities = []

        counter = 0
        for hypothesis in problem_data:
            hyp = problem_data[hypothesis]
            # Extract needed fields
            hypothesis_text = hyp["hypothesis"]
            sample2score = hyp["sample2score"]

            if hypothesis_text not in embedding_data:
                print(f"Hypothesis text not found in embeddings for {base_name}.")
                continue

            hypothesis_emb = np.array(embedding_data[hypothesis_text])

            scores = []
            similarities = []
            
#            good_scores = [(i, s) for (i, s) in sample2score.items() if s > 0.5]
 #           if len(good_scores):
  #              base_score = min(good_scores, key=lambda item: item[1])
   #         else:
            base_score = max(sample2score.items(), key=lambda item: item[1])
            base_score_emb = np.array(embedding_data[base_score[0]])
            Rhx0 = hypothesis_emb - base_score_emb
            rh_dict = {}

            # Compute similarity for each sample
            for sample, score in sample2score.items():
                if sample not in embedding_data:
                    print(f"Missing sample '''{sample}'''")
                    continue  # skip missing samples
                if score == base_score[1] and sample == base_score[0]:
                    continue # skip the sample chosen as the best

                sample_emb = np.array(embedding_data[sample])
#                sim = euclidean_similarity(hypothesis_emb, sample_emb)
                sim = cosine_similarity(hypothesis_emb, sample_emb, Rhx0)
                similarities.append(sim)
                if score >= 0.5:
                    high_similarities.append(sim)
                else:
                    low_similarities.append(sim)
                scores.append(score)
                rh_dict[sample] = sample_emb

            if not similarities:
                print(f"No valid samples for {base_name}.")
                continue

            plot_RH_map(rh_dict, sample2score, base_score[0], os.path.join(out_dir, f"TSNE_[{counter}].png"))

            # Scatter plot: sample2score vs embedding similarity
            plt.figure(figsize=(8, 6))
            plt.scatter(scores, similarities, alpha=0.6)
            plt.title(f"Score vs Embedding Similarity : {base_name}")
            plt.xlabel("Sample2Score (higher = more similar)")
            plt.ylabel("Cosine Distance")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"scatter_[{counter}].png"))
            plt.close()

            counter += 1

        # Store average similarities
        avg_high_sim = float(np.mean(high_similarities))
        avg_low_sim = float(np.mean(low_similarities))
        avg_similarities[f"{base_name}_hi"] = avg_high_sim
        avg_similarities[f"{base_name}_lo"] = avg_low_sim


    # Generate final bar chart of average similarities
    if avg_similarities:
        plt.figure(figsize=(10, 6))
        names = list(avg_similarities.keys())
        values = list(avg_similarities.values())
        plt.bar(names, values)
        plt.title("Average Embedding Similarity per Problem File (split by score val)")
        plt.ylabel("Average Similarity")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, "average_similarities.png"))
        plt.close()
        print(f"Saved summary plot: average_similarities.png")
    else:
        print("No valid results to plot.")
