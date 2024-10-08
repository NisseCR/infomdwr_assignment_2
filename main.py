import numpy as np
import pandas as pd
import py_stringmatching as sm
import seaborn as sb
from itertools import combinations
import time
import matplotlib.pyplot as plt
from collections import Counter
import scipy


from pandas import DataFrame


# TASK 1
def task_1():
    # loading the dataset
    excs_df = pd.read_excel('./data/excerpt_topics_total_7.xlsx')

    # explore the head and tail of the dataframe in order to check if the dataset is loaded in correctly
    print(excs_df.head)
    print(excs_df.columns)

    def primary_key_check(df: pd.DataFrame):
        id_col = df['ID']
        text_col = df['text']

        len_id = len(id_col)
        dist_id_len = len(id_col.unique())
        dist_text_len = len(text_col.unique())

        # unique text n / unique ID n
        if dist_text_len / dist_id_len == 1:
            print(f'The unique texts of the excerpts as long as the unique IDs')
        else:
            print(f'The text are not unique:  {dist_text_len}/ {dist_id_len}')

        # unique ID's / total ID's
        if dist_id_len / len_id == 1:
            print(f'ID is a primary key \n'
                  f'and the number of excerpts is {len_id}')
        else:
            print(f'ID is not a primary key \n'
                  f'and the number of unique excerpts_ids is {dist_id_len}')

    primary_key_check(excs_df)

    '''
    Now we know that the ID is the primary key and the text are a candidate key and thus that we work with unique excerpts. 
    '''

    '''
    To get a better understanding of the size of the text were working with, we wanted to know the minimal and maximal length of words. 
    To be more, precise we created a boxplot of the number of words in all excerpts. This allows us to know how large the 
    snippets of texts are, that we're working with. 
    '''

    def boxplot_words_excs(texts: pd.Series):
        # count the number of elements distinguished by spaces in the excerpts
        word_count = [len(exc.split(' ')) for exc in texts]

        print(f'The range of words are between {np.min(word_count)} and {np.max(word_count)}\n'
              f'with an average of {round(np.mean(word_count), 4)} and\n'
              f'with 95% of the excerpts between {np.percentile(word_count, q=[2.5, 97.5])}')

        plt.boxplot(word_count)
        plt.ylabel('Word (n)')
        plt.title('Boxplot of the words per excerpts')
        # plt.show()

    boxplot_words_excs(excs_df['text'])

    '''
    The number of words within the excerpts can differ strongly (4 to 1093 words). 
    However, 95% of the data remains between 24 and 180 words. Thus we work with small sentences to small paragraphs. 
    '''

    '''
    The dataset has two important attributes that can be utilised in an analysis:
    The century in which an excerpt was probably written and the religious keyword that is used. 
    It is important to know the representativeness of data in different centuries if the dataset is analysed as a whole. 
    It is also import to how many excerpts are anonymously written. If a temporal analysis would be conducted, we should make
    a choice what we would do with these datapoints.
    Furthermore, it is necessary to investigate if multiple keywords are not covered in one excerpt entry. The main concern
    of this dataset is the difference in excerpts with these keywords in it. If keywords would overlap, it could influence the results.
    '''

    def histo_cent(cent: pd.Series):
        print(f'Number of NAs in century: {cent.isna().sum()}')

        # histogram of the centuries with a bin for every century
        plt.hist(cent, bins=len(cent.unique()))
        plt.xlabel('Century')
        plt.ylabel('Excerpts (n)')
        plt.title('Histogram of excerpts in Late Antiquity')
        # plt.show()

    histo_cent(cent=excs_df['century'])

    '''
    Apparently, the unknown authors are dated or they were excluded from the dataset. 
    The authors of these excerpts were mostly writing in the 4th to the 6th century. 
    This could resemble an imbalance in the digitization process or gathering of the data,
    or could resemble the dominance of Greek in the 2th and 3th century in intellectual circles 
    and a steep decease of Latin writing ofter the collapse of the administration of Rome in 476. 
    '''

    def keyword_labeling(df):
        # slice df to lemmitized tokens of the text and the religious keywords
        token_kw_df = df[['tokens', 'keyword']]

        # get unique keywords
        uniq_kws = token_kw_df['keyword'].unique()

        search_tokens = ['synagoga', 'ecclesia', 'domus']

        results = []
        for kw in uniq_kws:
            # get subsets of the tokens with the unique keywords
            sub_tokens = token_kw_df[token_kw_df['keyword'] == kw]['tokens'].tolist()

            sub_result = []
            for s_tok in search_tokens:
                # if the religious word is in tokens than add 1 else 0 and sum the result
                sum_search_words = np.sum(1 if s_tok in token else 0 for token in sub_tokens)
                sub_result.append(sum_search_words)

            results.append(sub_result)

        # print the dataframe of the results
        results_df = pd.DataFrame(results, columns=search_tokens, index=uniq_kws)

        print(results_df.to_string())

    keyword_labeling(excs_df)

    '''
    The keywords as labeled the data are not exclusionary for the other keywords. They are almost exclusionary but not fully. 
    For further analysis, we should consider the overlapping excerpts. 
    '''

    '''
    So far, we have interrogated the structure of the data, the uniqueness and size of the excerpts, and the distributions and quality of the attributes keywords and centuries.
    The last phase of the EDA concerns the Sentiment Analysis (SA). The SA performed in this dataset was a static SA, which 
    considers a labeled list of Latin word with an emotional connotation. The sum of the SA labeled tokens per excerpt were 
    noted under SA score. 

    1. Does it looks like data from the real world, and for this we will utilise Benford's Law. 
    This will easily check if the scores are conform Berford's law. 
    2. The assumption of the SA score is that it remains similar over the centuries. 
    This can be tested with a scatterplot and a regression line. 
    '''

    def berford_sa_score(sa: pd.Series):
        # Convert values to string, remove negative sign, and select first character
        leading_digits = [int(str(abs(score))[0]) for score in sa.dropna() if isinstance(score, float)]

        # Compute the actual frequencies of leading digits
        actual_count = Counter(leading_digits)
        total_count = sum(actual_count.values())  # Total number of valid leading digits
        actual_freq = [actual_count[d] / total_count for d in range(1, 10)]  # Normalize to proportions

        # Compute the expected frequencies according to Benford's Law
        benford_freq = [np.log10(1 + 1 / d) for d in range(1, 10)]

        # Compare actual vs expected (Plotting)
        digits = range(1, 10)
        plt.figure(figsize=(10, 6))
        plt.bar(digits, actual_freq, width=0.4, label='Actual', align='center', alpha=0.7)
        plt.bar(digits, benford_freq, width=0.4, label='Benford', align='edge', alpha=0.7)
        plt.xlabel('Leading Digit')
        plt.ylabel('Proportion')
        plt.title('Comparison of Leading Digit Distribution with Benford\'s Law')
        plt.xticks(digits)
        plt.legend()
        plt.show()

    # Example usage
    berford_sa_score(excs_df['sa_score'])

    '''
    Benford\'s Law remains preserved. The distribution of first digits is proportional to the log of 1 + 1/digit. 
    This means that the distribution of this SA score has probability not serious errors. 
    '''

    def scatter_with_regression(df):
        # remove na's
        clean_df = df[['century', 'sa_score']].dropna()
        cent = clean_df['century']
        sa_score = clean_df['sa_score']

        # Create a scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(cent, sa_score, color='blue', label='Data Points', alpha=0.2)

        # Fit a regression line
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(cent, sa_score)

        # Regression line
        x_vals = np.array(cent)
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, color='red', label=f'Regression Line (slope={slope:.2f})')

        # Annotating the slope on the plot
        plt.text(x=cent.mean(), y=sa_score.min(), s=f'Slope: {slope:.2f}', color='red', fontsize=12)

        # Labels and title
        plt.xlabel('Century')
        plt.ylabel('SA Score')
        plt.title('Scatter Plot of SA Score vs Century with Regression Line')
        plt.legend()

        # Show plot
        plt.show()

        print(f'Slope: {slope:.2f}')

    # Call the function to create the plot
    scatter_with_regression(excs_df)

    '''
    A temporal analysis on the sentiments of the excerpts show a increase in the total sentiment over the century (r=0.28).
    This means that a normalisation step needs to be performed on the SA when other attributes like dominant topics or keywords 
    are considered. 
    This means that the total number of words with a positive connotation are used in later centuries, compared to earlier ones. 
    This might have a historical reason which needs further investigation. 
    '''


# TASK 2.1
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def load_data_21() -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Load the data from the 3 cvs files.
    :return: 3 dataframes
    """
    df_acm = pd.read_csv('./data/ACM.csv', header=0)
    df_dblp = pd.read_csv('./data/DBLP2.csv', header=0, encoding='latin1')
    df_mapping = pd.read_csv('./data/DBLP-ACM_perfectMapping.csv', header=0)
    return df_acm, df_dblp, df_mapping


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set the id columns as the index and preprocess text columns. Preprocessing includes removing whitespaces,
    converting to lowercase and replacing missing values with the empty word "".
    :param df: Dataframe containing raw data
    :return: Dataframe with preprocessed data
    """
    df = df.set_index('id')

    for col in df.columns:
        if df.dtypes[col] != 'object':
            continue

        # Lowercase
        df[col] = df[col].str.lower()

        # Whitespaces
        df[col] = df[col].str.replace(r'\s+', ' ')

        # NA values
        df[col] = df[col].fillna("")

    return df


def year_sim(s, t) -> int:
    """
    Match / mismatch between two year instances. Returns 1 if matched.
    :param s: Year instance left
    :param t: Year instance right
    :return: Match result
    """
    if s == t:
        return 1

    return 0


def record_sim(r: pd.Series, lev: sm.Levenshtein, jaro: sm.Jaro, aff: sm.Affine) -> pd.Series:
    """
    Apply py_stringmatching built-in similarity scores to a single record. Note that venue (using the affine similarity)
    is not normalised.
    :param r: data record
    :param lev: Levenshtein class instance
    :param jaro: Jaro class instance
    :param aff: Affine class instance
    :return: Record including similarity scores per attribute
    """
    r['sim_title'] = lev.get_sim_score(r['title_x'], r['title_y'])
    r['sim_authors'] = jaro.get_sim_score(r['authors_x'], r['authors_y'])
    r['sim_venue'] = aff.get_raw_score(r['venue_x'], r['venue_y'])
    r['sim_year'] = year_sim(r['year_x'], r['year_y'])
    return r


def pairwise_comparison(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Cross join the two dataframes and perform pairwise comparison through similarity scores. The venue score
    (calculated by the affine similarity) is normalised using the min-max formula. Similarity scores are applied in a
    vectorised method to increase performance for the O(n^2) algorithm.
    Finally, all the attribute scores are aggregated using custom weights. The resulting score has a range from 0 to 1.
    :param df1: Dataframe left
    :param df2: Dataframe right
    :return: Cross-joined data with similarity score between records
    """
    # Reset ids
    df1 = df1.reset_index(names='id')
    df2 = df2.reset_index(names='id')

    # Initiate models.
    lev = sm.Levenshtein()
    jaro = sm.Jaro()
    aff = sm.Affine()

    # Cross join for pairwise comparison.
    df = pd.merge(df1, df2, how='cross')

    # Apply sim score calculations to rows (vectorised).
    df['sim_title'], df['sim_authors'], df['sim_venue'], df['sim_year'] = np.nan, np.nan, np.nan, np.nan
    df = df.apply(lambda r: record_sim(r, lev, jaro, aff), axis=1)

    # Apply min-max scaling (normalisation) to sim_venue.
    df['sim_venue'] = (df['sim_venue'] - df['sim_venue'].min()) / (df['sim_venue'].max() - df['sim_venue'].min())

    # Calculate similarity score.
    df['sim_score'] = 0.45 * df['sim_title'] + 0.45 * df['sim_authors'] + 0.05 * df['sim_venue'] + 0.05 * df['sim_year']
    return df


def join_mapping(df_sims: pd.DataFrame, df_mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Join the results of similar records to the true duplicate record data. A 'match' column is added to indicate whether
    a match was correct.
    :param df_sims: Dataframe containing similarity scores of cross-join
    :param df_mapping: Dataframe containing the true duplicate records
    :return: Dataframe with indicator of correct results
    """
    # Retrieve ids only.
    df_sims = df_sims[['id_x', 'id_y']].rename(columns={'id_x': 'idDBLP', 'id_y': 'idACM'})

    # Left join to indicate correct predictions.
    df_mapping['match'] = 1
    df_match = pd.merge(df_sims, df_mapping, how='left', on=['idDBLP', 'idACM'])
    df_match['match'] = df_match['match'].fillna(0)
    return df_match


def task_2_1():
    df_acm, df_dblp, df_mapping = load_data_21()

    # Preprocess text.
    df_acm = preprocess(df_acm)
    df_dblp = preprocess(df_dblp)

    # Find similar records.
    time_start = time.time()
    df_sims = pairwise_comparison(df_dblp, df_acm)
    duration = round(time.time() - time_start, 2)

    # Filter similar records.
    df_sims = df_sims[df_sims['sim_score'] > 0.7]

    # Calculate precision
    df_match = join_mapping(df_sims, df_mapping)
    n_matches = df_match['match'].sum()
    precision = round(n_matches / len(df_match), 2)

    # Print results.
    print(df_match.head())
    print(df_match.describe())
    print(df_match.info())
    print(f"Precision is {precision} with {n_matches} correct matches.")

    # Runtime
    print(f"Runtime of the pairwise similarity comparison is {duration} seconds.")



# TASK 2.1
# Set a seed to ensure the permutations are identical across different runs
np.random.seed(45)

class LSH:
    """
    Implements the Locality Sensitive Hashing (LSH) technique for approximate
    nearest neighbor search.
    """
    buckets = []
    counter = 0

    def __init__(self, b: int):
        """
        Initializes the LSH instance with a specified number of bands.

        Parameters:
        - b (int): The number of bands to divide the signature into.
        """
        self.b = b
        for i in range(b):
            self.buckets.append({})

    def make_subvecs(self, signature: np.ndarray) -> np.ndarray:
        """
        Divides a given signature into subvectors based on the number of bands.

        Parameters:
        - signature (np.ndarray): The MinHash signature to be divided.

        Returns: - np.ndarray: A stacked array where each row is a subvector
        of the signature.
        """
        length = len(signature)
        assert length % self.b == 0
        r = int(length / self.b)
        subvecs = []
        for i in range(0, length, r):
            subvecs.append(signature[i:i + r])
        return np.stack(subvecs)

    def add_hash(self, signature: np.ndarray):
        """
        Adds a signature to the appropriate LSH buckets based on its
        subvectors.

        Parameters:
        - signature (np.ndarray): The MinHash signature to be hashed and added.
        """
        subvecs = self.make_subvecs(signature).astype(str)
        for i, subvec in enumerate(subvecs):
            subvec = ','.join(subvec)
            if subvec not in self.buckets[i].keys():
                self.buckets[i][subvec] = []
            self.buckets[i][subvec].append(self.counter)
        self.counter += 1

    def check_candidates(self) -> set:
        """
        Identifies candidate pairs from the LSH buckets that could be
        potential near duplicates.

        Returns:
        - set: A set of tuple pairs representing the indices of candidate
        signatures.
        """
        candidates = []
        for bucket_band in self.buckets:
            keys = bucket_band.keys()
            for bucket in keys:
                hits = bucket_band[bucket]
                if len(hits) > 1:
                    candidates.extend(combinations(hits, 2))
        return set(candidates)


def load_data_22():
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


def task_2_2():
    # Load and preprocess the data
    acm_df, dblp_df, match_df = load_data_22()
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
    lsh = LSH(num_buck)
    for sig in signatures:
        lsh.add_hash(sig)
    candidate_pairs = lsh.check_candidates()

    # Create a dataframe of the candidate pairs and compute the precision
    candidate_df = create_candidate_df(candidate_pairs, sentences)
    compute_precision(candidate_df, match_df)

    # End timer and print the elapsed time
    end_time = time.time()
    show_time(start_time, end_time)


# TASK 3
def task_3():
    diabetes_data = pd.read_csv('diabetes.csv')

    # Step 1
    diabetes_data.drop(columns=['Outcome'])

    correlation_before = diabetes_data.corr()
    print("Correlation before filling missing values:")
    print(correlation_before)

    sb.heatmap(correlation_before.corr(numeric_only=True), cmap="YlGnBu", annot=True)

    # Step 2: Replace 0 values in 'BloodPressure', 'SkinThickness', and 'BMI' with NaN
    columns_to_replace = ['BloodPressure', 'SkinThickness', 'BMI']
    diabetes_data[columns_to_replace] = diabetes_data[columns_to_replace].replace(0, pd.NA)

    diabetes_data.head(11)

    # Step 3
    for column in columns_to_replace:
        diabetes_data[column] = diabetes_data.groupby('Outcome')[column].transform(lambda x: x.fillna(x.mean()))

    diabetes_data.head()

    # Step 4
    correlation_after = diabetes_data.corr()

    sb.heatmap(correlation_after.corr(numeric_only=True), cmap="YlGnBu", annot=True)

    # Step 5: Compare the correlation matrices and show the changes
    correlation_change = correlation_after - correlation_before
    correlation_change.head()

    # Convert the correlation change matrix to a table
    correlation_change_df = pd.DataFrame(correlation_change)

    # Displaying the correlation change as a table
    print("\nDifference in Correlation Matrices (After - Before):")
    correlation_change_df.head(10)

    correlation_before_df = pd.DataFrame(correlation_before)
    correlation_before_df.head(10)

    correlation_after_df = pd.DataFrame(correlation_after)
    correlation_after_df.head(10)


# Execute all code
task_2_1()
task_2_2()
task_3()
