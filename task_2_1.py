import pandas as pd
import numpy as np
import csv


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def load_data():
    df_acm = pd.read_csv('./data/ACM.csv', header=0)
    df_dblp = pd.read_csv('./data/DBLP2.csv', header=0, encoding='latin1')
    return df_acm, df_dblp


def filter_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index('id')
    return df


def levenshtein_sim(s: str, t: str) -> float:
    rows = len(s) + 1
    cols = len(t) + 1
    dist = [[0 for x in range(cols)] for x in range(rows)]

    # source prefixes can be transformed into empty strings
    # by deletions:
    for i in range(1, rows):
        dist[i][0] = i

    # target prefixes can be created from an empty source string
    # by inserting the characters
    for i in range(1, cols):
        dist[0][i] = i

    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                c = 0
            else:
                c = 2
            dist[row][col] = min(dist[row - 1][col] + 1,  # deletion
                                 dist[row][col - 1] + 1,  # insertion
                                 dist[row - 1][col - 1] + c)  # substitution

    max = max(len(s), len(t))
    med = dist[row][col]

    return 1 - (med / max)


def jaro_sim(s: str, t: str) -> float:
    s_len = len(s)
    t_len = len(t)

    if s_len == 0 and t_len == 0:
        return 1

    match_distance = (max(s_len, t_len) // 2) - 1

    s_matches = [False] * s_len
    t_matches = [False] * t_len

    matches = 0
    transpositions = 0

    for i in range(s_len):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, t_len)

        for j in range(start, end):
            if t_matches[j]:
                continue
            if s[i] != t[j]:
                continue
            s_matches[i] = True
            t_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0

    k = 0
    for i in range(s_len):
        if not s_matches[i]:
            continue
        while not t_matches[k]:
            k += 1
        if s[i] != t[k]:
            transpositions += 1
        k += 1

    return ((matches / s_len) +
            (matches / t_len) +
            ((matches - transpositions / 2) / matches)) / 3


def year_sim(s, t):
    if s == t:
        return 1

    return 0


def record_sim(r1: pd.Series, r2: pd.Series) -> float:
    s_t = levenshtein_sim(r1['title'], r2['title'])
    s_a = jaro_sim(r1['authors'], r2['authors'])
    s_c = jaro_sim(r1['venue'], r2['venue'])
    s_y = year_sim(r1['year'], r2['year'])

    return (s_t + s_a + s_c + s_y) * 0.25


def pairwise_comparison(df: pd.DataFrame) -> pd.DataFrame:
    pass


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df.dtypes[col] != 'object':
            continue

        df[col] = df[col].str.lower()
        # TODO fix this
        df[col] = df[col].str.replace(r'\s+', ' ')
        df[col] = df[col].str.replace(r'\t+', ' ')

    return df


def main():
    df_acm, df_dblp = load_data()

    # Filter id
    df_acm = filter_id(df_acm)
    df_dblp = filter_id(df_dblp)

    # Preprocess text
    df_acm = preprocess_text(df_acm)
    df_dblp = preprocess_text(df_dblp)

    # EDA
    print(df_acm.head())
    print(df_dblp.head())


if __name__ == '__main__':
    main()