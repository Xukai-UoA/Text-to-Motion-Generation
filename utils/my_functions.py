"""
PyTorch version of my_functions.py
Helper functions for loading Word2Vec model and metadata
"""

import os
from os import listdir
from os.path import isfile, join
import gzip
import gensim
import numpy as np
import requests
from tqdm import tqdm


def download_file(url, dest_path, chunk_size=8192):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def load_w2v(w2v_path):
    """
    Load Word2Vec model (Google News vectors)

    Args:
        w2v_path: Path to save/load the word2vec model

    Returns:
        w2v_model: Gensim Word2Vec model
    """
    if not isfile(w2v_path):
        print("Word2Vec model not found.")
        print("Please download GoogleNews-vectors-negative300.bin manually from:")
        print("https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view")
        print(f"And place it at: {w2v_path}")

        # Alternative: try to download from alternative source
        print("\nAttempting to download from alternative source...")
        gz_path = w2v_path + '.gz'

        try:
            # Note: This URL might not work; user should download manually
            # from Google Drive link above
            download_file(
                'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz',
                gz_path
            )

            print(f"Extracting {gz_path}...")
            with gzip.open(gz_path, 'rb') as inF:
                with open(w2v_path, 'wb') as outF:
                    outF.write(inF.read())

            os.remove(gz_path)
            print(f"Word2Vec model saved to {w2v_path}")

        except Exception as e:
            print(f"Failed to download: {e}")
            print("Please download the file manually from the Google Drive link above.")
            raise FileNotFoundError(f"Word2Vec model not found at {w2v_path}")

    print(f"Loading Word2Vec model from {w2v_path}...")
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    print(f"Word2Vec model loaded successfully! Vocabulary size: {len(w2v_model)}")

    return w2v_model


def load_metadata(metadata_path):
    """
    Load preprocessed metadata

    Args:
        metadata_path: Path to the metadata .npz file

    Returns:
        train_action: Action sequences [num_data, dim_action, action_steps]
        train_script: Script embeddings [num_data, dim_sentence, sentence_steps]
        train_length: Sentence lengths [num_data]
        sentence_steps: Maximum sentence length
    """
    if not isfile(metadata_path):
        print(f"Metadata not found at {metadata_path}")
        print("Please run process_data.py first to generate the metadata.")
        print("Or download preprocessed metadata from:")
        print("https://drive.google.com/file/d/1k3FJOYslo7PU3U4TyM3VFuiIgpcxMEjZ/view")
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    print(f"Loading metadata from {metadata_path}...")
    npzfile = np.load(metadata_path, allow_pickle=True)

    train_action = npzfile['arr_0']
    train_script = npzfile['arr_1']
    train_length = npzfile['arr_2']
    sentence_steps = int(npzfile['arr_3'])

    print(f"Metadata loaded successfully!")
    print(f"  - Actions shape: {train_action.shape}")
    print(f"  - Scripts shape: {train_script.shape}")
    print(f"  - Lengths shape: {train_length.shape}")
    print(f"  - Max sentence length: {sentence_steps}")

    return train_action, train_script, train_length, sentence_steps


def get_init_pose(train_action):
    """
    Calculate mean initial pose from training data

    Args:
        train_action: Action sequences [num_data, dim_action, action_steps]

    Returns:
        init_pose: Mean first pose [dim_action, 1]
    """
    # Get all first frames
    first_frames = train_action[:, :, 0]  # [num_data, dim_action]

    # Calculate mean
    init_pose = np.mean(first_frames, axis=0, keepdims=True).T  # [dim_action, 1]

    return init_pose


def save_metadata(save_path, train_action, train_script, train_length, sentence_steps):
    """
    Save processed metadata to .npz file

    Args:
        save_path: Path to save the metadata
        train_action: Action sequences
        train_script: Script embeddings
        train_length: Sentence lengths
        sentence_steps: Maximum sentence length
    """
    np.savez(save_path, train_action, train_script, train_length, sentence_steps)
    print(f"Metadata saved to {save_path}")


if __name__ == "__main__":
    # Test loading functions
    print("Testing Word2Vec loading...")
    w2v_path = '../data/GoogleNews-vectors-negative300.bin'

    try:
        w2v_model = load_w2v(w2v_path)

        # Test word embedding
        test_word = 'woman'
        if test_word in w2v_model:
            embedding = w2v_model[test_word]
            print(f"\nEmbedding for '{test_word}': shape {embedding.shape}")
            print(f"First 5 values: {embedding[:5]}")
    except FileNotFoundError as e:
        print(f"\n{e}")

    print("\n" + "="*60)
    print("Testing metadata loading...")
    metadata_path = '../data/metadata.npz'

    try:
        train_action, train_script, train_length, sentence_steps = load_metadata(metadata_path)

        # Calculate initial pose
        init_pose = get_init_pose(train_action)
        print(f"\nInitial pose shape: {init_pose.shape}")
        print(f"Initial pose values:\n{init_pose.flatten()}")

    except FileNotFoundError as e:
        print(f"\n{e}")