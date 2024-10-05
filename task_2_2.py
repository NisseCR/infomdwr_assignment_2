import pandas as pd
import numpy as np
import time
import lsh_algorithm as algo

# Set a seed to ensure the permutations are identical across different runs
np.random.seed(45)


def load_data():
    """Read the data from the CSV file into a pandas dataframe"""
    acm_df = pd.read_csv('./data/acm.csv', header=0)
    dblp_df = pd.read_csv('./data/dblp2.csv', header=0,
                          delimiter=',', encoding='latin1')
    match_df = pd.read_csv('./data/DBLP-ACM_perfectMapping.csv',
                           header=0)
    return acm_df, dblp_df, match_df


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by combining all columns for a row into one string,
    making these strings all lower case, and removing multiple spaces.
    """
    # join column into one string
    df = df.apply(lambda row:
                  ' '.join(row.values.astype(str)),
                  axis=1)
    # Change all strings to lowercase
    df = df.str.lower()
    # Convert multiple spaces to single spaces
    df = df.str.replace(r'\s+', ' ', regex=True)
    return df


def preprocess_correction(df: pd.DataFrame) -> pd.DataFrame:
    """
    For the correct match file, preprocess into a dataframe that
    will allow for easy comparison with the candidate pairs. This preprocessing
    includes casting all values to strings, casting all strings to lowercase
    and removing multiple spaces.
    """
    # For the correct match files, we only need to cast everything to strings,
    # The values can stay in their separate columns
    df = df.astype(str)
    # Change all strings to lowercase and remove multiple spaces
    for col in df.columns:
        df[col] = df[col].str.lower()
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
    return df


def get_sentences(df1: pd.DataFrame, df2: pd.DataFrame) -> list:
    """Take two dataframes and combine them into one sentence list"""
    sentences = df1.tolist() + df2.tolist()
    return sentences


# COMPUTE THE SHINGLES AND BUILD THE VOCABULARY


def shingle(text: str, shingle_size: int) -> set:
    """
    Create a set of 'shingles' from the input text using k-shingling.

    Parameters:
        text (str): The input text to be converted into shingles.
        shingle_size (int): The length of the shingles (substring size).

    Returns:
        set: A set containing the shingles extracted from the input text.
    """
    shingle_set = []
    for i in range(len(text) - shingle_size + 1):
        shingle_set.append(text[i:i + shingle_size])
    return set(shingle_set)


def build_shingle_list(sentences: list, shingle_size: int) -> list:
    """
    Create a list of shingle sets from a list of sentences.

    Parameters:
    - sentences (list of str): A list of sentences to be converted into shingles.
    - shingle_size (int): The length of the shingles (substring size).

    Returns:
    - list of set: A list containing sets of shingles for each sentence.
    """
    shingles = []
    for sentence in sentences:
        shingles.append(shingle(sentence, shingle_size))
    return shingles


def build_vocab(shingle_sets: list) -> dict:
    """
    Constructs a vocabulary dictionary from a list of shingle sets.

    This function takes a list of shingle sets and creates a unified vocabulary
    dictionary. Each unique shingle across all sets is assigned a unique
    integer identifier.

    Parameters:
    - shingle_sets (list of set): A list containing sets of shingles.

    Returns:
    - dict: A vocabulary dictionary where keys are the unique shingles and
      values are their corresponding unique integer identifiers.
    """
    full_set = {item for set_ in shingle_sets for item in set_}
    vocabulary = {}
    for i, shin in enumerate(sorted(full_set)):
        vocabulary[shin] = i
    return vocabulary


def one_hot(shingle_set: set, vocabulary: dict):
    """
    Create a one-hot encoded vector from a set of shingles.
    """
    vec = np.zeros(len(vocabulary))
    for shing in shingle_set:
        idx = vocabulary[shing]
        vec[idx] = 1
    return vec


def build_1hot(shingles: list, vocab: dict) -> np.ndarray:
    """
    Create a one-hot encoded matrix from a list of shingle sets.

    Parameters:
    - shingles (list of set): A list containing sets of shingles.

    Returns:
    - np.ndarray: A binary matrix where each row corresponds to a sentence
      and each column corresponds to a shingle in the vocabulary.
    """
    one_hot_matrix = []
    for shingle_set in shingles:
        one_hot_matrix.append(one_hot(shingle_set, vocab))
    return np.stack(one_hot_matrix)

# Compute the minhash signature


def get_minhash_arr(num_hashes: int, vocabulary: dict):
    """
    Generates a MinHash array for the given vocabulary.

    This function creates an array where each row represents a hash function
    and each column corresponds to a word in the vocabulary. The values are
    permutations of integers representing the hashed value of each word for
    that particular hash function.

    Parameters:
    - num_hashes (int): The number of hash functions (rows) to generate for
    the MinHash array.
    - vocab (dict): The vocabulary where keys are words and values can be
    any data
      (only keys are used in this function).

    Returns:
    - np.ndarray: The generated MinHash array with `num_hashes` rows and
    columns equal
      to the size of the vocabulary. Each cell contains the hashed value of
      the corresponding
      word for the respective hash function.
    """

    length = len(vocabulary.keys())
    arr = np.zeros((num_hashes, length))
    for i in range(num_hashes):
        permutation = np.random.permutation(len(vocabulary.keys())) + 1
        arr[i, :] = permutation.copy()
    return arr.astype(int)


def get_signature(minhash: np.ndarray, vector: np.ndarray):
    """
    Computes the signature of a given vector using the provided MinHash matrix.

    The function finds the nonzero indices of the vector, extracts the
    corresponding columns from the MinHash matrix, and computes the signature
    as the minimum value across those columns for each row of the matrix.

    Parameters:
        - minhash (np.ndarray): The MinHash matrix where each column
    represents a shingle and each row represents a hash function.
    - vector (np.ndarray): A vector representing the presence
    (non-zero values) or absence (zero values) of shingles.

    Returns:
    - np.ndarray: The signature vector derived from the MinHash matrix for
    the provided vector.
    """
    idx = np.nonzero(vector)[0].tolist()
    sel_shingles = minhash[:, idx]
    signature = np.min(sel_shingles, axis=1)
    return signature


def build_signatures(minhash: np.ndarray,
                     shingles_1hot: np.ndarray) -> list:
    """
    Create the MinHash signatures for the given sentences.

    Parameters:
    - sentences (list of str): A list of sentences to be converted into
    MinHash signatures.
    - minhash (np.ndarray): The MinHash matrix where each column represents
    a shingle and each row represents a hash function.
    - shingles_1hot (np.ndarray): A binary matrix where each row corresponds
    to a sentence and each column corresponds to a shingle in the vocabulary.

    Returns:
    - list of np.ndarray: A list of MinHash signatures for each sentence.
    """
    signatures = []
    for sel_vector in shingles_1hot:
        signatures.append(get_signature(minhash, sel_vector))
    return signatures


# COMPUTE THE SIMILARITY SCORES


def jaccard_similarity(set1, set2):
    """Compute the Jaccard similarity of two sets"""
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    return intersection_size / union_size if union_size != 0 else 0.0


def compute_signature_similarity(signature_1, signature_2):
    """
    Calculate the similarity between two signature matrices using MinHash.

    Parameters:
    - signature_1: First signature matrix as a numpy array.
    - signature_matrix2: Second signature matrix as a numpy array.

    Returns:
    - Estimated Jacquard similarity.
    """
    # Ensure the matrices have the same shape
    if signature_1.shape != signature_2.shape:
        raise ValueError("Both signature matrices must have the same shape.")
    # Count the number of rows where the two matrices agree
    agreement_count = np.sum(signature_1 == signature_2)
    # Calculate the similarity
    similarity = agreement_count / signature_2.shape[0]

    return similarity


def compute_similarity_matrix(signatures: list) -> np.matrix:
    """
    Build a similarity matrix filled with the estimated jaccard similarity
    of two signatures.

    :param signatures: The list containing signatures that need to be
    cross-compared
    :return: The similarity matrix of all signatures
    """
    # Compute the similarity scores between all pairs of signatures
    # and store them in a matrix
    sim_scores = np.zeros((len(signatures), len(signatures)))
    for m in range(len(signatures)):
        for n in range(m + 1, len(signatures)):
            # If already visibly visited, skip
            if sim_scores[m, n] != 0:
                continue
            # If not yet visited, or the score is 0, compute
            sim_scores[m, n] = compute_signature_similarity(signatures[m],
                                                            signatures[n])
            sim_scores[n, m] = sim_scores[m, n]
    print(sim_scores)


def create_candidate_df(candidate_pairs: set, sentences: list) -> pd.DataFrame:
    """
    Create a dataframe of the candidate pairs for easy comparison with the
    true matches.
    """
    # Extract the id's of the candidate pairs and put them in a new dataframe
    # with the columns 'first_id' and 'second_id'
    candidate_df = pd.DataFrame(columns=['first_id', 'second_id'])
    number_of_candidates = 2224
    for candidate in list(candidate_pairs)[:number_of_candidates]:
        first_id = sentences[candidate[0]].split()[0]
        second_id = sentences[candidate[1]].split()[0]
        candidate_df = candidate_df._append({'first_id': second_id,
                                             'second_id': first_id},
                                            ignore_index=True)
    return candidate_df


def compute_precision(candidate_df: pd.DataFrame,
                      match_df: pd.DataFrame) -> None:
    """
    Compute the precision of the candidate pairs by comparing them to the
    true matches.
    """
    # Sort the row values to ensure that the order of the values is the same in
    # both the candidate and match dataframes
    candidate_df_sorted = candidate_df.apply(lambda row: sorted(row),
                                             axis=1)
    match_df_sorted = match_df.apply(lambda row: sorted(row), axis=1)

    # Convert the sorted dataframes to sets of tuples for easy comparison
    candidate_pairs_set = set([tuple(x) for x in candidate_df_sorted.values])
    match_pairs_set = set([tuple(x) for x in match_df_sorted.values])

    # Compute the precision by finding the intersection of the
    # candidate and match
    correct_pairs_set = candidate_pairs_set & match_pairs_set
    precision = len(correct_pairs_set) / len(match_pairs_set)
    print("Precision: ", precision)


def show_time(start_time, end_time) -> None:
    """Print the elapsed time based on start and end time"""
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    seconds = round(seconds, 2)

    print(f"The program took {int(minutes)} minute(s) and {seconds} "
          f"seconds to run.")


def main():
    # Load and preprocess the data
    acm_df, dblp_df, match_df = load_data()
    acm_df = preprocess_df(acm_df)
    dblp_df = preprocess_df(dblp_df)
    match_df = preprocess_correction(match_df)

    # Combine the dataframes into a list of sentences and start the timer
    sentences = get_sentences(acm_df, dblp_df)
    start_time = time.time()

    # Shingle the sentences and build the vocabulary
    k = 5
    shingles = build_shingle_list(sentences, k)
    vocab = build_vocab(shingles)
    shingles_1hot = build_1hot(shingles, vocab)

    # Generate the MinHash array and compute the signatures
    hash_num = 800
    min_arr = get_minhash_arr(hash_num, vocab)
    signatures = build_signatures(min_arr, shingles_1hot)

    # Compute the similarity matrix between the signatures
    compute_similarity_matrix(signatures)

    # Apply the LSH algorithm
    num_buck = 100
    lsh = algo.LSH(num_buck)
    for sig in signatures:
        lsh.add_hash(sig)
    candidate_pairs = lsh.check_candidates()

    # Create a dataframe of the candidate pairs and compute the precision
    candidate_df = create_candidate_df(candidate_pairs, sentences)
    compute_precision(candidate_df, match_df)

    # End timer and print the elapsed time
    end_time = time.time()
    show_time(start_time, end_time)


if __name__ == '__main__':
    main()
