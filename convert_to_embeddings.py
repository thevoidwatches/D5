import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import tqdm

INPUT_FOLDER = "./problem_output"
OUTPUT_FOLDER = "./problem_embeddings"

MODEL_NAME = "all-mpnet-base-v2"

# function to take one of the loaded files and get the categories from it, returning them as a list of dictionaries
def extract_texts(file):
    texts = []
    print("Extracting texts...")
    for cat in file:
        hyp=file[cat]
        for sample in hyp['sample2score'].keys():
            if not sample in texts:
                texts.append(sample)
        texts.append(hyp['hypothesis'])
    return texts

# function to generate the embedding for a piece of text and return it
def embed_text(model, text):
    embedding = model.encode(text)
    return embedding

if __name__ == '__main__':
    model = SentenceTransformer(MODEL_NAME)
    # for each file:
    file_list = os.listdir(INPUT_FOLDER)
    print(f"Found {len(file_list)} files in folder {INPUT_FOLDER}")
    for file_name in file_list:
        # load it
        print(f"Running on file {file_name}...")
        with open(f"{INPUT_FOLDER}/{file_name}", "rb") as infile:
            data = pickle.load(infile)
        # pull the relevant texts from it with extract_texts
            texts = extract_texts(data)

            file_embeddings = {}
        # for each text
            print("Generating embeddings...")
            for text in tqdm.tqdm(texts):
                if not text in file_embeddings.keys():
                    embedding = embed_text(model, text)
                    file_embeddings[text] = embedding

        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        with open(f"{OUTPUT_FOLDER}/{file_name}", "wb") as outfile:
            pickle.dump(file_embeddings, outfile)
        print(f"Finished with file {file_name}")
            # generate an embedding for the relevant text
            # add the embedding to a dictionary with the text as its key
        # save a new pickle file with the embeddings, the corpuses, and the hypotheses?
