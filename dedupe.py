import operator
import itertools
from functools import reduce
import numpy as np
import pandas as pd
import jellyfish
from matching.fellegi_sunter import fs_em, gamma_pattern
from matching.cluster import clusterdf, ngram_index
from matching.similarity import shingle


def canopy(n):
    df = fl_data()
    df2 = oge_data()

    x = ngram_index(df, 'nname', n=n)

    for s in shingle(df2.nname[0],k=n):
        print(len(x[s]))

def simple_score(r_a, r_b):
    return (jellyfish.jaro_winkler(r_a[0], r_b[0]) + 
            (1.0 if r_a[1] == r_b[1] else 0) + 
            (1.0 if r_a[2] == r_b[2] else 0))


def match_score():
    df_oge_d = pd.read_csv('./data/ogesdw.whd.whisard.fl-dedupe.csv')

    df_fl_d = pd.read_csv('./data/fl.restaurant-inspections-dedupe.csv')

    dfz = df_oge_d.merge(df_fl_d, on='nzip')
    
    mh = np.array([ 0.08910887,  0.98090761,  0.88638131])
    uh = np.array([ 0.00019569,  0.00198455,  0.0003152 ])

    dfz['score'] = dfz[['nname_x', 'snum_x', 'nzip', 'nname_y', 'snum_y', 'nzip']].apply(
                       lambda x: log_score(gamma_pattern(((x[0],x[1],x[2]), (x[3],x[4],x[5]))),
                                           mh, uh),
                       axis=1)

    return dfz


def training_set():
    dfz = pd.read_csv('./data/matches.csv')
    dft = dfz[['nname_x','nname_y', 'snum_x', 'snum_y', 'location_city', 'cty_nm', 'match']]
    dft.to_csv('./data/match_train.csv', encoding='utf-8', index=False)


def fl_data():
    df_fl = pd.read_csv('./data/fl.restaurant-inspections.csv')

    df_fl['id'] = df_fl.index
    df_fl_a = df_fl.groupby(['dba', 
                     'location_address', 
                     'location_city', 
                     'location_zip_code',
                     'district',
                     'county_number']).first().reset_index()

    # Process name and zip code fields
    name_prep = lambda x:x.lower().translate({None:"'.,"}) if pd.notnull(x) else ''
    df_fl_a['nname'] = df_fl_a.dba.apply(name_prep)
    df_fl_a['nzip'] = df_fl_a.location_zip_code.apply(lambda x:x[0:5] if pd.notnull(x) else x)
    df_fl_a['snum'] = df_fl_a.location_address.str.split(' ',1).apply(lambda x:x[0])
    return df_fl_a


def oge_data():
    df_oge = pd.read_csv('./data/ogesdw.whd.whisard.fl.csv')

    df_oge_a_fl = df_oge.groupby(['trade_nm', 
                          'legal_name', 
                          'street_addr_1_txt', 
                          'cty_nm', 
                          'st_cd', 
                          'zip_cd',
                          'naic_cd']).first().reset_index()

    # Process name and zip code fields
    name_prep = lambda x:x.lower().translate({None:"'.,"}) if pd.notnull(x) else ''
    df_oge_a_fl['nname'] = df_oge_a_fl.trade_nm.apply(name_prep)
    df_oge_a_fl['nzip'] = df_oge_a_fl.zip_cd.apply(lambda x: str(int(x)))
    df_oge_a_fl['snum'] = df_oge_a_fl.street_addr_1_txt.str.split(' ',1).apply(lambda x:x[0])
    return df_oge_a_fl


def em_oge(df_in):

    df1 = df_in[['nname', 'nzip', 'snum']].sample(frac=0.1)

    df1['k'] = 1

    m_u = [('name', 0.5, 0.0000005),
          ('zip', 0.9, 0.001),
          ('snum', 0.9, 0.0003)]
    df_mu = pd.DataFrame(data=m_u, columns=['Field', 'm_i', 'u_i'])

    record_pairs = []

    print(df1.info())

    for idx, r in df1.merge(df1, on='k').iterrows():
        record_pairs.append(((r.nname_x, r.nzip_x, r.snum_x), (r.nname_y, r.nzip_y, r.snum_y)))

    mh, uh, ph = fs_em(record_pairs, df_mu.m_i.values, df_mu.u_i.values, 1e-6)

    return mh, uh, ph


def scores(mh, uh, p):
    df_scores = pd.DataFrame(data=list(itertools.product([1,0], repeat=3)),
                 columns=['name', 'zip', 'street num'])
    df_scores['m'] = df_scores[['name', 'zip', 'street num']].apply(lambda p: mp(p,mh,uh), axis=1)
    df_scores['u'] = df_scores[['name', 'zip', 'street num']].apply(lambda p: up(p,mh,uh), axis=1)
    df_scores['mu'] = df_scores.m/df_scores.u
    df_scores['w'] = df_scores[['name', 'zip', 'street num']].apply(lambda p: log_score(p,mh,uh), axis=1)
    df_scores.sort('mu', ascending=False, inplace=True)
    df_scores['cu'] = df_scores.u.cumsum()
    df_scores['cm'] = df_scores.m.cumsum()
    return df_scores


def cluster(df):
    df1 = df[['nname', 'nzip', 'snum']].reset_index()
    dfm = df1.merge(df1, on='nzip')
    #dfm = dfm[dfm.index_x != dfm.index_y]
    dfm['g1'] = dfm[['nname_x','nname_y']].apply(lambda x: 1 if jellyfish.jaro_winkler(x[0],x[1]) > 0.8 else 0, axis=1)
    dfm['g2'] = dfm[['snum_x','snum_y']].apply(lambda x: 1 if x[0] == x[1] else 0, axis=1)
    dfc = clusterdf(dfm)
    dd = dfc.groupby(['cluster_id', 'index_x']).first().reset_index()
    ddd = dfc.groupby('cluster_id').first()
    return df.ix[ddd.index_x]


def manual(dfz):
    for idx, r in dfz[(dfz.match==0)&(dfz.score>14)][['dba','location_address','trade_nm','street_addr_1_txt','jw','score']].sort_values(by='jw',ascending=False).iterrows():
        print(r)
        m = input()
        if m == 'm':
            dfz.match[idx] = 1
        if m == 'q':
            break

