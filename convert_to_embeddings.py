import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FOLDER = "./problem_output"
OUTPUT_FOLDER = "./problem_embeddings"

MODEL_NAME = "all-mpnet-base-v2"

# function to take one of the loaded files and get the categories from it, returning them as a list of dictionaries
def extract_texts(file):
    pass

# function to generate the embedding for a piece of text and return it
def embed_text(model, text):
    embedding = model.encode(text, show_progress_bar=True)
    return embedding

if __name__ == '__main__':
    model = SentenceTransformer(MODEL_NAME)
    # for each file:
    for file_name in os.listdir(INPUT_FOLDER):
        # load it
        with open(f"{INPUT_FOLDER}/{file_name}", "rb") as infile:
            data = pickle.load(infile)
        # pull the relevant texts from it with extract_texts
            texts = extract_texts(data)

            file_embeddings = {}
        # for each text
            for text in texts:
                embedding = embed_text(text)
                file_embeddings[text] = embedding

        with open(f"{OUTPUT_FOLDER}/{file_name}", "wb") as outfile:
            pickle.dumb(file_embeddings, outfile)
            # generate an embedding for the relevant text
            # add the embedding to a dictionary with the text as its key
        # save a new pickle file with the embeddings, the corpuses, and the hypotheses?
    pass