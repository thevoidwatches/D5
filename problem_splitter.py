import os
import pickle

INPUT_FILE="OpenD5.pkl"
OUTPUT_FOLDER="problem_set"

# Ensure the output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the original pickle file
with open(INPUT_FILE, "rb") as f:
    data = pickle.load(f)

# Ensure the data is iterable
if not isinstance(data, (list, tuple)):
    raise TypeError(f"Expected list or tuple, but got {type(data)}")

print(f"Loaded {len(data)} problem sets from {INPUT_FILE}")

# Save each item as its own pickle file
for i, item in enumerate(data):
    output_path = os.path.join(OUTPUT_FOLDER, f"task_{i}.pkl")
    with open(output_path, "wb") as out_f:
        pickle.dump(item, out_f)
    # print(f"Saved: {output_path}")

print(f"Successfully split {len(data)} tasks into ./{OUTPUT_FOLDER}.")
