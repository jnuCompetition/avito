import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import combinations

def co_prob(dataset,cols,feature_name,normilize=False,return_col=False):

    if isinstance(cols,str) :
        cols = [cols]
    cols = list(cols)
    X = dataset[cols]
    X[feature_name] = list(range(len(dataset)))
    X = X.groupby(by=cols, as_index=False).count()
    if normilize:
        X[feature_name] = X[feature_name]/len(dataset)
    dataset = dataset.merge(X,how='left',on=cols)
    if return_col:
        return dataset[feature_name]
    return dataset

def condition_prob(dataset,Y,X,feature_name,return_col=False):

    if isinstance(Y,str):
        Y = [Y]
    if isinstance(Y,str):
        X = [X]
    X = list(X)
    Y = list(Y)
    XY = list(set(X + Y))
    prob_x = co_prob(dataset,X,'prob_x',return_col=True)
    prob_xy = co_prob(dataset,XY,'prob_xy', return_col=True)
    dataset[feature_name] = prob_xy/prob_x

    if return_col:
        return dataset[feature_name]
    return dataset

def condition_stat(dataset,target_col,cols,feature_name,stat_f,return_col=False):

    if isinstance(cols,str) :
        cols = [cols]
    cols = list(cols)
    X = dataset[[target_col]+cols]

    if stat_f == 'mean':
        X = X.groupby(by=cols, as_index=False).aggregate({target_col:'mean'})
    elif stat_f == 'std':
        X = X.groupby(by=cols, as_index=False).aggregate({target_col:'std'})
    elif hasattr(stat_f, '__call__'):
        X = X.groupby(by=cols, as_index=False).aggregate({target_col:stat_f})
    # X = pd.DataFrame()
    X = X.rename(columns={target_col:feature_name})


    dataset = dataset.merge(X,how='left',on=cols)
    if return_col:
        return dataset[feature_name]
    return dataset

def num2bin(dataset,cols,num_bins):

    if isinstance(cols,str):
        dataset[cols + '_bin'] = pd.cut(dataset[cols], num_bins,
                                labels=['bin' + str(i) for i in range(num_bins)])
    else:
        for col in cols:
            dataset[col+'_bin'] = pd.cut(dataset[col],num_bins,
                                    labels=['bin'+str(i) for i in range(num_bins)])
    return dataset

def drop_cols(cols):
    train = pd.read_csv("./dataset/train.csv", parse_dates=["activation_date"])
    test = pd.read_csv("./dataset/test.csv", parse_dates=["activation_date"])

    train.drop(cols,axis=1,inplace=True)
    test.drop(cols,axis=1,inplace=True)

    train.to_csv("./dataset/train.csv", index=False)
    test.to_csv("./dataset/test.csv", index=False)

def save_pickle():
    train = pd.read_csv("./dataset/train.csv", parse_dates=["activation_date"])
    test = pd.read_csv("./dataset/test.csv", parse_dates=["activation_date"])

    usecols = [
        # "item_id",
        # "user_id",
        "region",
        "city",
        "parent_category_name",
        "category_name",
        "param_1",
        "param_2",
        "param_3",
        # "title",
        # "description",
        "price",
        "item_seq_number",
        # "activation_date",
        "user_type",
        # "image",
        "image_top_1",

        "weekday",
        # "month",     0 gain
        "day",
        # "week",
        "description_len",
        "title_len",
        "param_combined",
        "param_combined_len",
        "description_char",
        "title_char",
        "param_char",

        # "latitude",
        # "longitude",

        'avg_days_up_user',
        'avg_times_up_user',

        # 'days_up_sum',
        # 'times_put_up',
        'ridge_preds',
        'n_user_items',
    ]

    dtypes = {
        "item_id": 'category',
        "user_id": 'category',
        "region": 'category',
        "city": 'category',
        "parent_category_name": 'category',
        "category_name": 'category',
        "param_1": 'category',
        "param_2": 'category',
        "param_3": 'category',
        "title": 'category',
        "description": 'category',
        "price": np.float64,
        "item_seq_number": np.float64,
        "activation_date": 'category',
        "user_type": 'category',
        "image": 'category',
        "image_top_1": 'category',

        "weekday": np.float64,
        "month": np.float64,
        "day": np.float64,
        "week": np.float64,
        "description_len": np.float64,
        "title_len": np.float64,
        "param_combined": 'category',
        "param_combined_len": np.float64,
        "description_char": np.float64,
        "title_char": np.float64,
        "param_char": np.float64,

        "latitude": np.float64,
        "longitude": np.float64,

        'avg_days_up_user': np.float64,
        'avg_times_up_user': np.float64,

        # 'days_up_sum',
        # 'times_put_up',
        'ridge_preds':np.float64,
        'n_user_items':np.float64,
    }

    for col in usecols:
        train[col] = train[col].astype(dtypes[col])
        test[col] = test[col].astype(dtypes[col])

    print(train['city'])
    train.to_pickle("./dataset/train.pkl")
    test.to_pickle("./dataset/test.pkl")

    train = pd.read_pickle("./dataset/train.pkl")
    test = pd.read_csv("./dataset/test.pkl")
    print(train.info())
    print(test.info())


def norm_text(sentences):
    import re
    def remove_dots_from_shortenings(match):
        s = match.string[match.start():match.end()]
        return s.replace('.', '')

    # shortenings = [u'мм', u'см', u'м', u'км', u'мл', u'л', u'г', u'кг', u'т', u'лит',
    #                u'р', u'руб', u'сот', u'га', u'шт', u'ш', u'дб', u'вт',
    #                u'ул', u'пр', u'д', u'кв', u'чел', u'жен', u'муж', u'тыс',
    #                u'др']
    # shortenings = '|'.join(shortenings)

    def normalize(text):
        text = text.lower()
        # text = text.replace(u'ё', u'е')
        # text = text.replace(u'²', '2')
        # text = re.sub('(\d+)[.](\d+)', r'\1,\2', text)
        # text = re.sub('([a-zа-я][.]){2,}', remove_dots_from_shortenings, text)
        # text = re.sub('(?<=[^а-я])(' + shortenings + ')[.]', '\1 ', text)
        # re.sub(r'(\d+),(\d+)', r'\1.\2', text)
        #
        # text = re.sub(u'[^a-zа-я0-9.]', ' ', text)
        text = re.sub('\s+', ' ', text)
        return text.strip()

    result = []
    for sent in tqdm(sentences):
        result.append(normalize(sent))

    return result

def feature_selective():
    from config import USE_FEAT

    usecols = [
        # "item_id",
        # "user_id",
        "region",
        "city",
        "parent_category_name",
        "category_name",
        "param_1",
        "param_2",
        "param_3",
        # "title",
        # "description",
        "price",
        "item_seq_number",
        # "activation_date",
        "user_type",
        # "image",
        "image_top_1",

        "weekday",
        # "month",     0 gain
        "day",
        # "week",
        "description_len",
        "title_len",
        # "param_combined",
        "param_combined_len",
        "description_char",
        "title_char",
        "param_char",

        # "latitude",
        # "longitude",

        'avg_days_up_user',
        'avg_times_up_user',

        # 'days_up_sum',
        # 'times_put_up',
        'ridge_preds',
        'n_user_items',
    ]

    dtypes = {
        "item_id": 'category',
        "user_id": 'category',
        "region": 'category',
        "city": 'category',
        "parent_category_name":'category' ,
        "category_name": 'category',
        "param_1": 'category',
        "param_2": 'category',
        "param_3": 'category',
        "title": 'category',
        "description": 'category',
        "activation_date": 'category',
        "user_type": 'category',
        "image": 'category',
        "image_top_1": 'category',
        "param_combined":'category',
        "weekday": 'category',

        "day": 'category',
        "week": 'category',

        "price": np.float64,
        "item_seq_number": np.float64,
        "month": np.float16,


        "description_len": np.float16,
        "title_len": np.float16,
        "param_combined_len": np.float16,
        "description_char": np.float16,
        "title_char": np.float16,
        "param_char": np.float16,

        "latitude": np.float16,
        "longitude": np.float16,
    }

    for k, i in list(dtypes.items()):
        if k not in usecols:
            dtypes.pop(k)

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
        'item_seq_number'
    ]



    for i in range(1,3):
        for comb_feat in combinations(cate_cols, i):
            for conti_feat in conti_cols:
                feat_mean = conti_feat + '_mean_' + '_'.join(comb_feat)
                feat_norm = conti_feat + '_norm_' + '_'.join(comb_feat)
                feat_std = conti_feat + '_std_' + '_'.join(comb_feat)
                usecols+=[feat_mean,feat_std,feat_norm]

    from extractFeature import comb_feat
    co_features, condi_features = comb_feat()
    for cols_x,cols_y in condi_features:
        usecols.append("_".join(cols_y) + "_by_" + "_".join(cols_x))
    for cols in co_features:
        usecols.append("co_"+"_".join(cols))

    usecols +=USE_FEAT

    for col in ["description", "title"]:
        usecols += [col+'_num_words',col+'_num_unique_words',col+'_words_vs_unique']


    cate_cols = [
        "city",
        "category_name",
        "param_1",
        "activation_date",
        "user_type",
        "weekday",
        "image_top_1",
    ]
    for col in cate_cols:
        usecols.append(col+'_TE_mean')
        usecols.append(col+'_TE_std')


    feat_imp = pd.read_csv('feat.csv')
    feat = feat_imp['name'].values.tolist()
    feat = feat[-250:]
    #
    usecols = feat

    # print(feat_imp.shape)
    # drop_feat = feat_imp['name'].values
    # usecols = [col for col in usecols if col in feat]
    # usecols = feat

    for col in usecols:
        if col not in dtypes:
            dtypes[col] = np.float16

    return usecols,dtypes

if __name__ == '__main__':

    # usecols = []
    # for feat in ['description','title','param_combined']:
    #     usecols += [feat[:4]+"_tf_" + str(x) for x in range(200)]
    # drop_cols(usecols)
    #
    # pipeline()

    # import dask.dataframe as dd
    save_pickle()

    # print(pd.get_dummies(d['char'],prefix='aaa'))