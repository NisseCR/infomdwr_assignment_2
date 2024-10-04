from typing import Tuple

import numpy as np
import pandas as pd
import py_stringmatching as sm
import time

from pandas import DataFrame

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def load_data() -> tuple[DataFrame, DataFrame, DataFrame]:
    df_acm = pd.read_csv('./data/ACM.csv', header=0)
    df_dblp = pd.read_csv('./data/DBLP2.csv', header=0, encoding='latin1')
    df_mapping = pd.read_csv('./data/DBLP-ACM_perfectMapping.csv', header=0)
    return df_acm, df_dblp, df_mapping


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
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
    if s == t:
        return 1

    return 0


def record_sim(r: pd.Series, lev: sm.Levenshtein, jaro: sm.Jaro, aff: sm.Affine) -> pd.Series:
    r['sim_title'] = lev.get_sim_score(r['title_x'], r['title_y'])
    r['sim_authors'] = jaro.get_sim_score(r['authors_x'], r['authors_y'])
    r['sim_venue'] = aff.get_raw_score(r['venue_x'], r['venue_y'])
    r['sim_year'] = year_sim(r['year_x'], r['year_y'])
    return r


def pairwise_comparison(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
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


def read_mapping() -> pd.DataFrame:
    df = pd.read_csv('./data/DBLP-ACM_perfectMapping.csv', header=0)
    return df


def join_mapping(df_sims: pd.DataFrame, df_mapping: pd.DataFrame) -> pd.DataFrame:
    # Retrieve ids only.
    df_sims = df_sims[['id_x', 'id_y']].rename(columns={'id_x': 'idDBLP', 'id_y': 'idACM'})
    print(df_sims.head())
    print(df_sims.dtypes)
    print(df_mapping.dtypes)

    # Left join to indicate correct predictions.
    df_mapping['match'] = 1
    df_match = pd.merge(df_sims, df_mapping, how='left', on=['idDBLP', 'idACM'])
    df_match['match'] = df_match['match'].fillna(0)
    return df_match


def main():
    df_acm, df_dblp, df_mapping = load_data()

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


if __name__ == '__main__':
    main()