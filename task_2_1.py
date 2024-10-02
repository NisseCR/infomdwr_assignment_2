import pandas as pd
import py_stringmatching as sm
import time


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def load_data():
    df_acm = pd.read_csv('./data/ACM.csv', header=0)
    df_dblp = pd.read_csv('./data/DBLP2.csv', header=0, encoding='latin1')
    return df_acm, df_dblp


def filter_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index('id')
    return df


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df.dtypes[col] != 'object':
            continue

        # Lowercase
        df[col] = df[col].str.lower()

        # Whitespaces
        # TODO fix this
        df[col] = df[col].str.replace(r'\s+', ' ')
        df[col] = df[col].str.replace(r'\t+', ' ')

        # NA values
        df[col] = df[col].fillna("")

    return df


def year_sim(s, t):
    if s == t:
        return 1

    return 0


def record_sim(r: pd.Series, lev: sm.Levenshtein, jaro: sm.Jaro, aff: sm.Affine) -> float:
    s_t = lev.get_sim_score(r['title_x'], r['title_y'])
    s_a = jaro.get_sim_score(r['authors_x'], r['authors_y'])
    s_c = aff.get_raw_score(r['venue_x'], r['venue_y'])
    s_y = year_sim(r['year_x'], r['year_y'])
    return 1


def pairwise_comparison(df1: pd.DataFrame, df2: pd.DataFrame) -> list:
    ids = []

    # Initiate models
    lev = sm.Levenshtein()
    jaro = sm.Jaro()
    aff = sm.Affine()

    # Vectorised way
    df = pd.merge(df1, df2, how='cross')
    df = df.iloc[0:100000]

    time_s = time.time()
    df.apply(lambda r: record_sim(r, lev, jaro, aff), axis=1)
    print(f"Duration is {time.time() - time_s} seconds")

    print(df.head())
    print(len(df))

    return ids


def read_mapping() -> pd.DataFrame:
    df = pd.read_csv('./data/DBLP-ACM_perfectMapping.csv', header=0)
    return df


def join_mapping(ids: list) -> pd.DataFrame:
    df_mapping = read_mapping()
    df_results = pd.DataFrame(data=ids, columns=['idDBLP', 'idACM'])
    df_results['match'] = 1

    # TODO remove
    df_results.iloc[0] = ['conf/sigmod/SlivinskasJS01', 375678, 1]

    df_match = pd.merge(df_mapping, df_results, how='left', on=['idDBLP', 'idACM'])
    df_match['match'] = df_match['match'].fillna(0)
    return df_match


def main():
    df_acm, df_dblp = load_data()

    # Filter id
    df_acm = filter_id(df_acm)
    df_dblp = filter_id(df_dblp)

    # Preprocess text
    df_acm = preprocess_text(df_acm)
    df_dblp = preprocess_text(df_dblp)

    # Find similar records
    time_start = time.time()

    # TODO error continue, they wont run, no catchy
    df_acm_sub = df_acm.iloc[0:4]
    df_dblp_sub = df_dblp.iloc[0:300]
    record_ids = pairwise_comparison(df_dblp, df_acm)
    duration = round(time.time() - time_start, 2)

    # Calculate precision
    df_match = join_mapping(record_ids)
    n_matches = df_match['match'].sum()
    precision = round(n_matches / len(df_match), 2)
    print(f"Precision is {precision}")

    # Runtime
    print(f"Runtime of the pairwise similarity comparison {duration} seconds.")


if __name__ == '__main__':
    main()