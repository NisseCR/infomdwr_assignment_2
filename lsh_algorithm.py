import numpy as np

from itertools import combinations

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
