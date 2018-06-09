import pandas as pd
from tqdm import tqdm
import numpy as np
import gc
import multiprocessing as mlp
from multiprocessing import Manager
from itertools import combinations,product
from tool import co_prob,condition_prob,condition_stat,num2bin
from process_worker import prob_feature_worker,tfidf_worker,conti_stat_feature_worker
from sklearn.linear_model import Ridge

"""
tem_id                  - Ad id.
user_id                 - User id.
region                  - Ad region.
city                    - Ad city.
parent_category_name    - Top level ad category as classified by Avito's ad model.
category_name           - Fine grain ad category as classified by Avito's ad model.
param_1                 - Optional parameter from Avito's ad model.
param_2                 - Optional parameter from Avito's ad model.
param_3                 - Optional parameter from Avito's ad model.
title                   - Ad title.
description             - Ad description.
price                   - Ad price.
item_seq_number         - Ad sequential number for user.
activation_date         - Date ad was placed.
user_type               - User type.
image                   - Id code of image. Ties to a jpg file in train_jpg. Not every ad has an image.
image_top_1             - Avito's classification code for the image.
"""

def baseFeature(df):
    df['weekday'] = df.activation_date.dt.weekday
    df['month'] = df.activation_date.dt.month
    df['day'] = df.activation_date.dt.day
    df['week'] = df.activation_date.dt.week

    # length of description
    df['description_len'] = df['description'].fillna("").apply(lambda x: len(x.split()))

    # length of title
    df['title_len'] = df['title'].fillna("").apply(lambda x: len(x.split()))

    # param_combined and its length
    df['param_combined'] = df.apply(
        lambda row: ' '.join([str(row['param_1']), str(row['param_2']), str(row['param_3'])]), axis=1)
    df['param_combined_len'] = df['param_combined'].fillna("").apply(lambda x: len(x.split()))

    # charater len of text columns
    df['description_char'] = df['description'].fillna("").apply(len)
    df['title_char'] = df['title'].fillna("").apply(len)
    df['param_char'] = df['param_combined'].fillna("").apply(len)
    for col in ["description", "title"]:
        df[col+'_num_words'] = df[col].fillna("").apply(lambda comment: len(comment.split()))
        df[col+'_num_unique_words']=df[col].fillna("").apply(lambda comment:len(set(w for w in comment.split())))
        df[col+'_words_vs_unique']=df[col+'_num_unique_words']/df[col+'_num_words']*100

    return df

def get_geo_feature(df):
    df['city_region'] = df["city"] + ' ' + df["region"]
    city_region = pd.DataFrame()
    city_region['city_region'] = df['city_region'].drop_duplicates().reset_index(drop=True)
    city_region['latitude'] = np.nan
    city_region['longitude'] = np.nan

    from geopy import geocoders
    g = geocoders.GoogleV3()

    def get_geocode(row):
        try:
            geocode = g.geocode(row['city_region'],timeout=60,language='en')
            row['latitude'] = geocode.latitude
            row['longitude'] = geocode.longitude
        except:
            row['latitude'] = np.nan
            row['longitude'] = np.nan
        return row

    city_region = city_region.apply(get_geocode,axis=1)

    df = df.merge(city_region, on='city_region', how='left')
    df.drop(['city_region'],axis=1,inplace=True)
    return df

def tfidf_feat(train,test,features,n_components):

    print(features,n_components)
    pool = mlp.Pool(len(features))
    results = []
    for feat,k in zip(features,n_components):
        result = pool.apply_async(tfidf_worker,
                                  args=(feat,k,train[feat],test[feat]))
        results.append((feat,k,result))
    pool.close()
    pool.join()

    for feat,k,result in tqdm(results):
        result.get()
        # de_matrix = np.load('./dataset/'+feat+'.npy')
        # cols = [feat[:4] + "_tf_" + str(x) for x in range(k)]
        #
        # for i, col in enumerate(cols):
        #     train[col] = de_matrix[:len(train),i]
        #     test[col] = de_matrix[len(train):,i]

    # return train,test

def agg_time_feat(train,test):
    used_cols = ['item_id', 'user_id']

    train_sub = train[used_cols]
    train_active = pd.read_csv('./dataset/train_active.csv', usecols=used_cols)
    test_sub = test[used_cols]
    test_active = pd.read_csv('./dataset/test_active.csv', usecols=used_cols)

    train_periods = pd.read_csv('./dataset/periods_train.csv', parse_dates=['date_from', 'date_to'])
    test_periods = pd.read_csv('./dataset/periods_test.csv', parse_dates=['date_from', 'date_to'])

    all_samples = pd.concat([
        train_sub,
        train_active,
        test_sub,
        test_active
    ]).reset_index(drop=True)
    all_samples.drop_duplicates(['item_id'], inplace=True)

    all_periods = pd.concat([train_periods,test_periods])
    all_periods['days_up'] = all_periods['date_to'].dt.dayofyear - all_periods['date_from'].dt.dayofyear

    gp = all_periods.groupby(['item_id'])[['days_up']]

    gp_df = pd.DataFrame()
    gp_df['days_up_sum'] = gp.sum()['days_up']
    gp_df['times_put_up'] = gp.count()['days_up']
    gp_df.reset_index(inplace=True)
    gp_df.rename(index=str, columns={'index': 'item_id'})

    all_periods.drop_duplicates(['item_id'], inplace=True)
    all_periods = all_periods.merge(gp_df, on='item_id', how='left')

    all_periods = all_periods.merge(all_samples, on='item_id', how='left')

    gp = all_periods.groupby(['user_id'])[['days_up_sum', 'times_put_up']].mean().reset_index() \
        .rename(index=str, columns={
        'days_up_sum': 'avg_days_up_user',
        'times_put_up': 'avg_times_up_user'
    })

    n_user_items = all_samples.groupby(['user_id'])[['item_id']].count().reset_index() \
                        .rename(index=str, columns={'item_id': 'n_user_items'})
    gp = gp.merge(n_user_items, on='user_id', how='outer')

    train = train.merge(gp, on='user_id', how='left')
    test = test.merge(gp, on='user_id', how='left')

    return train,test

def get_img_info(df,datatype):
    from process_worker import img_info_worker
    if datatype=='tr':
        path = './dataset/train_img/'
    elif datatype=='te':
        path = './dataset/test_img/'
    else:
        raise RuntimeError("don't have this data")

    img_files = df['image'].values
    num_cpu = mlp.cpu_count()
    pool = mlp.Pool(num_cpu)
    num_task = 1+len(img_files)//num_cpu
    results = []

    for i in range(num_cpu):
        result = pool.apply_async(img_info_worker,
                        args=(img_files[i*num_task:(i+1)*num_task],path))
        results.append(result)
    pool.close()
    pool.join()

    img_feat = []
    for result in tqdm(results):
        img_feat += result.get()

    assert len(df) == img_feat
    columns = ['light_percent', 'dark_percent','average_color',
               'blurrness_score','dominant_color','average_pixel_width']
    img_feat = pd.DataFrame(img_feat,columns=columns)
    img_feat.to_csv('./dataset/'+datatype+'_info.csv',index=False)


def get_img_confi(df, datatype):
    from process_worker import img_confi_worker
    if datatype == 'tr':
        path = './dataset/train_img/'
    elif datatype == 'te':
        path = './dataset/test_img/'
    else:
        raise RuntimeError("don't have this data")

    img_files = df['image'].values
    num_cpu = mlp.cpu_count()
    pool = mlp.Pool(num_cpu)
    num_task = 1 + len(img_files) // num_cpu

    img_feat = pd.DataFrame()
    for model in ['resnet50','xception','inception_v3']:
        results = []
        for i in range(num_cpu):
            result = pool.apply_async(img_confi_worker,
                                      args=(img_files[i*num_task:(i + 1)*num_task],model,path))
            results.append(result)
        pool.close()
        pool.join()

        feat = []
        for result in tqdm(results):
            feat += result.get()

        assert len(df) == img_feat
        columns = [model+str(i) for i in range(3)]
        feat = pd.DataFrame(feat, columns=columns)
        img_feat = pd.concat([img_feat,feat],axis=1)

    img_feat.to_csv('./dataset/' + datatype + '_confi.csv', index=False)

def comb_feat():
    #  共现概率特征
    co_features = []
    co_prob_feat = [
        "category_name",
        "city",
        "param_1",
        "param_2",
        "param_3",
        "user_id",
        "weekday",
    ]
    for i in range(1, 4):
        co_features += list(combinations(co_prob_feat, i))

    # 条件概率特征
    from itertools import product
    condi_features = []

    area = ["city"]
    user_type = ["user_type"]
    user_id = ['user_id']
    date = ["activation_date", "weekday"]
    item = ["param_1", "category_name"]

    comb_feat = [area, user_type, date, item, user_id]
    for i in range(1, 2):
        for j in range(1, 3):
            for condi_varis in combinations(comb_feat, i):
                X = list(product(*condi_varis))
                res = [i for i in comb_feat if i not in condi_varis]
                for res_varis in combinations(res, j):
                    Y = list(product(*res_varis))
                    condi_features += list(product(X, Y))

    return co_features,condi_features

def prob_feature(train,test):

    def run_task(train,test,usecols,features,is_co):
        num_cpu = mlp.cpu_count()//2
        pool = mlp.Pool(num_cpu)
        num_task = 1 + len(features) // num_cpu
        results = []

        for i in range(num_cpu):
            result = pool.apply_async(prob_feature_worker,
                                      args=(usecols,features[i * num_task:(i + 1) * num_task],is_co))
            results.append(result)
        pool.close()
        pool.join()

        for result in tqdm(results):
            train_feat,test_feat = result.get()
            train = pd.concat([train,train_feat],axis=1)
            test = pd.concat([test,test_feat],axis=1)

        return train,test




    l_tr = len(train)
    l_te = len(test)
    usecols = [
        "item_id",
        "user_id",
        "region",
        "city",
        "parent_category_name",
        "category_name",
        "param_1",
        "param_2",
        "param_3",
        "activation_date",
        "user_type",
        "weekday"
    ]

    co_features,condi_features = comb_feat()
    train,test = run_task(train,test,usecols,co_features,True)

    train, test = run_task(train,test,usecols,condi_features,False)

    assert l_tr == len(train)
    assert l_te == len(test)

    return train,test

def user_item_matrix(train,test,dims):

    all_samples = pd.concat([
        train,
        test
    ]).reset_index(drop=True)

    def feat_encode(all_samples):

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        all_samples['city_id'] = le.fit_transform(all_samples['city'].values)
        le = LabelEncoder()
        all_samples['user_type_id'] = le.fit_transform(all_samples['user_type'].values)
        le = LabelEncoder()
        all_samples['category_name_id'] = le.fit_transform(all_samples['category_name'].values)

        all_samples['image_top_1_id'] = all_samples['image_top_1']
        all_samples['image_top_1_id'].fillna(all_samples['image_top_1'].max()+1, inplace=True)
        return all_samples

    all_samples = feat_encode(all_samples)

    i = len(all_samples['city_id'].unique())
    j = len(all_samples['user_type_id'].unique())
    assert all_samples['city_id'].max() == i-1
    assert all_samples['user_type_id'].max() == j-1
    num_user = i*j
    all_samples['u_id'] = i*all_samples['user_type_id'] + all_samples['city_id']
    all_samples['u_id'] = all_samples['u_id'].astype(int)

    i = int(all_samples['image_top_1_id'].max()+1)
    j = len(all_samples['category_name_id'].unique())
    all_samples['i_id'] = i * all_samples['category_name_id'] + all_samples['image_top_1_id']
    all_samples['i_id'] = all_samples['i_id'].astype(int)
    num_item = i*j

    ui_graph = all_samples[['u_id','i_id']].values

    u_i_matrix = np.zeros((num_user,num_item))
    print(u_i_matrix.shape)
    for i,j in ui_graph:
        u_i_matrix[i,j] +=1

    from scipy.sparse.linalg import svds
    [u,sigma,v] = svds(u_i_matrix,dims)
    v = v.T

    for i in range(dims):
        all_samples['user_vec_'+str(i)] = 0
        all_samples['item_vec_'+str(i)] = 0
    user_cols = ['user_vec_'+str(i) for i in range(dims)]
    item_cols = ['item_vec_'+str(i) for i in range(dims)]

    print('set vec')
    def set_vec(row):
        row[user_cols] = u[row['u_id']]
        row[item_cols] = v[row['i_id']]
        return row
    usecol = ['city', 'user_type', 'category_name', 'image_top_1']
    all_samples.drop_duplicates(usecol, inplace=True)
    all_samples = all_samples.apply(set_vec,axis=1)

    print('merge')
    all_samples = all_samples[usecol+user_cols+item_cols]
    train = train.merge(all_samples, on=usecol, how='left')
    test = test.merge(all_samples, on=usecol, how='left')

    for data,name in zip([train,test],['tr','te']):
        u = data[user_cols].values
        i = data[item_cols].values
        np.save('./dataset/'+name+'_u_id.csv',u)
        np.save('./dataset/'+name+'_i_id.csv',i)


def get_stat_features():

    cate_cols = [
        "city",
        "category_name",
        "param_1",
        "activation_date",
        "user_type",
        "user_id",
        "weekday",
    ]

    conti_cols = [
        "price",
        'item_seq_number',
    ]

    features = []
    for i in range(1, 3):
        for comb_feat in combinations(cate_cols, i):
            for conti_feat in conti_cols:
                features.append([conti_feat,list(comb_feat)])
    usecols = cate_cols + conti_cols + ['item_id']
    return usecols,features

def conti_stat_feature(train,test):

    l_train = len(train)
    l_test = len(test)

    usecols,features = get_stat_features()

    num_cpu = mlp.cpu_count()//2
    pool = mlp.Pool(num_cpu)
    num_task = 1+len(features)//num_cpu
    results = []

    for i in range(num_cpu):
        result = pool.apply_async(conti_stat_feature_worker,
                        args=(usecols,features[i*num_task:(i+1)*num_task]))
        results.append(result)
    pool.close()
    pool.join()

    for result in tqdm(results):
        train_feat,test_feat = result.get()
        train = pd.concat([train,train_feat],axis=1)
        test = pd.concat([test,test_feat],axis=1)

    assert len(train) == l_train
    assert len(test) == l_test

    return train,test

# def num_to_bin_feature(train,test):
#     samples = train[['price', 'item_seq_number']].append(test[['price', 'item_seq_number']]).reset_index(drop=True)
#
#     for num_bin,col in [()]
#     samples = num2bin(samples, ['price', 'item_seq_number'], 500)
#     train['price_bin'] = samples.loc[:len(train) - 1, 'price_bin']
#     test['price_bin'] = samples.loc[len(train):, 'price_bin']
#     train["item_seq_number_bin"] = samples.loc[:len(train) - 1, "item_seq_number_bin"]
#     test["item_seq_number_bin"] = samples.loc[len(train):, "item_seq_number_bin"]
#
#     return train, test

def target_encode(train,test):
    cate_cols = [
        "city",
        "category_name",
        "param_1",
        "activation_date",
        "user_type",
        "weekday",
        "image_top_1",
        # "price_bin",
        # "item_seq_number_bin"
    ]

    for col in tqdm(cate_cols):
        feat = condition_stat(train[[col,'deal_probability']],'deal_probability',col,col+'_TE_mean','mean')
        feat = feat[[col,col+'_TE_mean']].drop_duplicates([col]).reset_index(drop=True)
        train = train.merge(feat,how='left',on=[col])
        test = test.merge(feat, how='left', on=[col])

        feat = condition_stat(train[[col, 'deal_probability']], 'deal_probability', col, col +'_TE_std', 'std')
        feat = feat[[col,col+'_TE_std']].drop_duplicates([col]).reset_index(drop=True)
        train = train.merge(feat, how='left', on=[col])
        test = test.merge(feat, how='left', on=[col])
    return train,test

def oof_feature(train,test,alpha):
    NFOLDS = 5
    SEED = 42

    class SklearnWrapper(object):
        def __init__(self, clf, seed=0, params=None, seed_bool=True):
            if (seed_bool == True):
                params['random_state'] = seed
            self.clf = clf(**params)

        def train(self, x_train, y_train):
            self.clf.fit(x_train, y_train)

        def predict(self, x):
            return self.clf.predict(x)

    def get_oof(clf, x_train, y, x_test):

        from sklearn.cross_validation import KFold

        oof_train = np.zeros((len(y),))
        oof_test = np.zeros((len(test),))
        oof_test_skf = np.empty((NFOLDS,len(test)))

        kf = KFold(len(y), n_folds=NFOLDS, shuffle=True, random_state=SEED)
        for i, (train_index, test_index) in enumerate(kf):
            print('\nFold {}'.format(i))
            x_tr = x_train[train_index]
            y_tr = y[train_index]
            x_te = x_train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    print('load data')
    desc_matrix = np.load('./dataset/description.npy')
    titl_matrix = np.load('./dataset/title.npy')
    ready_df = np.concatenate((desc_matrix,titl_matrix),axis=1)

    print('train')
    ridge_params = {'alpha':alpha, 'fit_intercept': True, 'normalize': False, 'copy_X': True,
                    'max_iter': None, 'tol': 0.0001, 'solver': 'auto', 'random_state': SEED}
    y = train['deal_probability'].values
    ridge = SklearnWrapper(clf=Ridge, seed=SEED, params=ridge_params)
    ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:len(train)], y, ready_df[len(train):])

    from sklearn.metrics import mean_squared_error
    from math import sqrt
    rms = sqrt(mean_squared_error(y, ridge_oof_train))
    print('Ridge OOF RMSE: {}'.format(rms))

    train['ridge_preds'] = ridge_oof_train
    test['ridge_preds'] = ridge_oof_test

    return train,test


def pipeline():


    train = pd.read_csv("./dataset/train.csv",parse_dates=["activation_date"])
    test = pd.read_csv("./dataset/test.csv",parse_dates=["activation_date"])

    # train,test = baseFeature(train),baseFeature(test)
    # train,test = get_geo_feature(train),get_geo_feature(test)

    print('tfidf')
    tfidf_feat(train,test,['description','title'],[800,500])


    # train, test = agg_time_feat(train, test)

    # train,test = num_to_bin_feature(train,test)
    # train,test = target_encode(train,test)


    # train, test = oof_feature(train, test,20)


    # train, test = prob_feature(train, test)
    # train,test = conti_stat_feature(train,test)
    # user_item_matrix(train,test,100)

    # print('save')
    # train.to_csv("./dataset/train.csv",index=False)
    # test.to_csv("./dataset/test.csv",index=False)
    # print('save 2')
    # train.to_csv("./dataset/train.csv.gz", index=False,compression='gzip')
    # test.to_csv("./dataset/test.csv.gz", index=False,compression='gzip')

if __name__ == '__main__':
    pipeline()


















