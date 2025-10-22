# imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tqdm

PROBLEMS_FOLDER = "./problem_output"
EMBEDDINGS_FOLDER = "./problem_embeddings"
OUTPUT_FOLDER = "./embedding_graphs"

def load_pickle(path):
    """Safely load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

def euclidean_similarity(vec1, vec2):
    """Compute inverted Euclidean similarity (larger = more similar)."""
    distance = np.linalg.norm(vec1 - vec2)
    return 1 / (1 + distance)

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

        all_similarities = []

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

            # Compute similarity for each sample
            for sample, score in sample2score.items():
                if sample not in embedding_data:
                    print(f"Missing sample '''{sample}'''")
                    continue  # skip missing samples

                sample_emb = np.array(embedding_data[sample])
                sim = euclidean_similarity(hypothesis_emb, sample_emb)
                similarities.append(sim)
                all_similarities.append(sim)
                scores.append(score)

            if not similarities:
                print(f"No valid samples for {base_name}.")
                continue

            # Scatter plot: sample2score vs embedding similarity
            plt.figure(figsize=(8, 6))
            plt.scatter(scores, similarities, alpha=0.6)
            plt.title(f"Score vs Embedding Similarity — {base_name}")
            plt.xlabel("Sample2Score (higher = more similar)")
            plt.ylabel("Embedding Similarity (1 / (1 + distance))")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"scatter_[{counter}].png"))
            plt.close()

            counter += 1

        # Store average similarity
        avg_sim = float(np.mean(all_similarities))
        avg_similarities[base_name] = avg_sim

    # Generate final bar chart of average similarities
    if avg_similarities:
        plt.figure(figsize=(10, 6))
        names = list(avg_similarities.keys())
        values = list(avg_similarities.values())
        plt.bar(names, values)
        plt.title("Average Embedding Similarity per Problem File")
        plt.ylabel("Average Similarity (1 / (1 + distance))")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, "average_similarities.png"))
        plt.close()
        print(f"Saved summary plot: average_similarities.png")
    else:
        print("No valid results to plot.")
