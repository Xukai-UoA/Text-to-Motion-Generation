"""
PyTorch version of process_data.py
Process MSR-VTT data: extract poses and convert text to word embeddings
"""

import numpy as np
import os
import scipy.io as scio
from my_functions import load_w2v, save_metadata
from tqdm import tqdm


def preprocess(data_dir='../data', output_path='../data/metadata.npz'):
    """
    Preprocess the MSR-VTT dataset:
    1. Load Word2Vec model
    2. Process text descriptions into embeddings
    3. Load pose data from .mat files
    4. Save processed data

    Args:
        data_dir: Directory containing the data
        output_path: Path to save the processed metadata
    """

    # Load Word2Vec model
    w2v_path = os.path.join(data_dir, 'GoogleNews-vectors-negative300.bin')
    print("Loading Word2Vec model...")
    w2v_model = load_w2v(w2v_path)

    embed_size = w2v_model.vector_size  # Updated for newer gensim versions
    print(f"Word embedding size: {embed_size}")

    # Load all text descriptions to find max length
    script_file = os.path.join(data_dir, 'total_script.txt')
    print(f"\nReading scripts from {script_file}...")

    if not os.path.exists(script_file):
        print(f"Error: {script_file} not found!")
        print("Please create a file with all text descriptions, one per line.")
        return

    with open(script_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    max_length = 0
    for line in lines:
        line = line.lower().strip()
        words = line.split()
        word_cnt = len(words)
        if max_length < word_cnt:
            max_length = word_cnt

    print(f'Maximum sentence length: {max_length}')

    # Get pose and script file paths
    pose_path = os.path.join(data_dir, 'pose/')
    script_path = os.path.join(data_dir, 'script/')

    if not os.path.exists(pose_path):
        print(f"Error: {pose_path} not found!")
        print("Please create pose/ directory with .mat files containing pose data.")
        return

    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found!")
        print("Please create script/ directory with text descriptions.")
        return

    pose_files = sorted([f for f in os.listdir(pose_path)
                        if os.path.isfile(os.path.join(pose_path, f)) and f.endswith('.mat')])

    print(f"\nFound {len(pose_files)} pose files")

    # Process data
    total_pose_list = []
    total_script_list = []

    # Get vocabulary keys
    if hasattr(w2v_model, 'key_to_index'):
        # Newer gensim version
        vocab_keys = w2v_model.key_to_index.keys()
    else:
        # Older gensim version
        vocab_keys = w2v_model.vocab.keys()

    print("\nProcessing data...")
    for idx, p_file in enumerate(tqdm(pose_files, desc="Processing files")):
        # Load pose data
        pose_data = scio.loadmat(os.path.join(pose_path, p_file))

        # The pose data should be in 'pred_vector' field
        # Adjust this based on your actual .mat file structure
        if 'pred_vector' in pose_data:
            curr_pred = pose_data['pred_vector']
        else:
            print(f"\nWarning: 'pred_vector' not found in {p_file}")
            print(f"Available keys: {pose_data.keys()}")
            continue

        # Load corresponding script
        # Assuming file naming convention: pose_XXXX.mat -> script_XXXX.txt
        script_id = p_file.split('_')[1].split('.')[0] if '_' in p_file else p_file.split('.')[0]
        script_file = os.path.join(script_path, f'script_{script_id}.txt')

        if not os.path.exists(script_file):
            print(f"\nWarning: {script_file} not found, skipping...")
            continue

        with open(script_file, 'r', encoding='utf-8') as s_f:
            lines = s_f.readlines()

        # Process each line
        for line in lines:
            line = line.lower().strip()
            if not line:
                continue

            words = line.split()

            # Create word embedding array
            tmp_word_array = np.zeros((embed_size, len(words)))

            for word_idx, word in enumerate(words):
                if word not in vocab_keys:
                    # Unknown word: use zero vector
                    curr_emb_vec = np.zeros((embed_size,))
                else:
                    curr_emb_vec = w2v_model[word]

                tmp_word_array[:, word_idx] = curr_emb_vec

            total_script_list.append(tmp_word_array)
            total_pose_list.append(curr_pred)

    # Convert to arrays
    print("\nConverting to numpy arrays...")
    pose_array = np.array(total_pose_list)
    print(f"Pose array shape: {pose_array.shape}")

    num_data = pose_array.shape[0]

    # Create script array with padding
    script_array = np.zeros((num_data, embed_size, max_length))
    script_length = np.zeros((num_data,))

    for i in range(num_data):
        tmp_script = total_script_list[i]
        script_length[i] = tmp_script.shape[1]
        script_array[i, :, :tmp_script.shape[1]] = tmp_script

    print(f"Script array shape: {script_array.shape}")
    print(f"Script length array shape: {script_length.shape}")

    # Save processed data
    print('\nDATA READY')
    save_metadata(output_path, pose_array, script_array, script_length, max_length)
    print('DATA SAVED')

    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    print(f"Total samples: {num_data}")
    print(f"Pose dimension: {pose_array.shape[1]}")
    print(f"Action steps: {pose_array.shape[2]}")
    print(f"Embedding dimension: {embed_size}")
    print(f"Max sentence length: {max_length}")
    print(f"Average sentence length: {np.mean(script_length):.2f}")
    print(f"Min sentence length: {int(np.min(script_length))}")
    print(f"Max sentence length: {int(np.max(script_length))}")
    print("="*60)


def verify_data(metadata_path='../data/metadata.npz'):
    """
    Verify the processed data

    Args:
        metadata_path: Path to the metadata file
    """
    print("Verifying processed data...")

    npzfile = np.load(metadata_path, allow_pickle=True)

    train_action = npzfile['arr_0']
    train_script = npzfile['arr_1']
    train_length = npzfile['arr_2']
    sentence_steps = int(npzfile['arr_3'])

    print("\nData shapes:")
    print(f"  Actions: {train_action.shape}")
    print(f"  Scripts: {train_script.shape}")
    print(f"  Lengths: {train_length.shape}")
    print(f"  Max sentence steps: {sentence_steps}")

    # Check for NaN or Inf
    print("\nData quality checks:")
    print(f"  Actions - NaN: {np.isnan(train_action).any()}, Inf: {np.isinf(train_action).any()}")
    print(f"  Scripts - NaN: {np.isnan(train_script).any()}, Inf: {np.isinf(train_script).any()}")

    # Sample data
    print("\nSample action (first frame of first sequence):")
    print(train_action[0, :, 0])

    print("\nSample script embedding (first 5 dims of first word):")
    print(train_script[0, :5, 0])

    print("\nFirst 10 sentence lengths:")
    print(train_length[:10])

    print("\nVerification complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process MSR-VTT data for Text2Action')
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='Directory containing the raw data')
    parser.add_argument('--output', type=str, default='../data/metadata.npz',
                       help='Output path for processed metadata')
    parser.add_argument('--verify', action='store_true',
                       help='Verify existing processed data instead of processing')

    args = parser.parse_args()

    if args.verify:
        verify_data(args.output)
    else:
        preprocess(data_dir=args.data_dir, output_path=args.output)

        # Verify after processing
        print("\n" + "="*60)
        verify_data(args.output)