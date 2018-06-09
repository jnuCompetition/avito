
import pandas as pd
from itertools import combinations,product
# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold
import numpy as np

NFOLDS = 10
SEED = 42


class ModelWrap:


    def __init__(self,eval_set):


        self.baseModel = lgb.LGBMModel(
            objective = 'regression',
            n_jobs=-1,
            max_depth=18,
            learning_rate=0.07,
            colsample_bytree=0.75,
            subsample=0.75,
            num_leaves=2**14,
            n_estimators=1000,
            metrics='mse',
            silent=False
        )

        self.eval_set = eval_set

    def fit(self,X,Y):

        self.baseModel.fit(X,Y,verbose=2,eval_set=self.eval_set)

    def predict(self,X):

        return self.predict(X)


def main(train_df,Y,test_df,K):
    from sklearn.cross_validation import KFold

    kf = KFold(len(train_df), n_folds=K, shuffle=True,random_state=42)

    param = {
        'objective' : 'regression',
        'boosting_type': 'gbdt',
        'metric' : 'rmse',
        # 'max_bin':50,
        'num_leaves': 128,
        'feature_fraction': 0.4,
        'bagging_fraction': 0.8,
        'bagging_freq': 2,
        'learning_rate': 0.017,
        # 'device':'gpu',
        # 'max_depth':15,

        'lambda_l1':0,
        'lambda_l2':0,
        'nthread':30,
    }

    results = np.zeros((len(test_df,)))
    oof_preds = np.zeros(train_df.shape[0])

    for i, (tr_idx, val_idx) in enumerate(kf):
        print('第{}次训练...'.format(i))

        train_X = train_df.loc[tr_idx,:]
        train_Y = Y[tr_idx]
        train_dataset = lgb.Dataset(train_X,train_Y)

        del train_X,train_Y
        gc.collect()

        val_X = train_df.loc[val_idx,:]
        val_Y = Y[val_idx]
        val_dataset = lgb.Dataset(val_X,val_Y)

        del val_Y
        gc.collect()

        model = lgb.train(params=param,train_set=train_dataset,num_boost_round=10000,
                          valid_sets=[val_dataset],verbose_eval=50,early_stopping_rounds=100)


        if i==0:
            feature_score = pd.DataFrame()
            feature_score['name'] = model.feature_name()
            feature_score['importance1'] = model.feature_importance()
            feature_score['importance2'] = model.feature_importance('gain')
            feature_score = feature_score.loc[feature_score['importance1']>0]
            feature_score['importance3'] = feature_score['importance2']/(feature_score['importance1']**0.5)
            feature_score = feature_score.sort_values(by=['importance3','importance2','importance1'])
            print(feature_score)
            feature_score.to_csv('feat_importance.csv',index=False)

        test_pred = model.predict(test_df,num_iteration=model.best_iteration-2)  + \
                    model.predict(test_df,num_iteration=model.best_iteration-1) + \
                    model.predict(test_df,num_iteration=model.best_iteration) + \
                    model.predict(test_df,num_iteration=model.best_iteration+1) + \
                    model.predict(test_df,num_iteration=model.best_iteration+2)

        test_pred /=5
        test_pred[test_pred<0] = 0
        test_pred[test_pred>1] = 1
        results += test_pred

        oof_preds[val_idx] =model.predict(val_X,num_iteration=model.best_iteration-2)  + \
                            model.predict(val_X,num_iteration=model.best_iteration-1) + \
                            model.predict(val_X,num_iteration=model.best_iteration) + \
                            model.predict(val_X,num_iteration=model.best_iteration+1) + \
                            model.predict(val_X,num_iteration=model.best_iteration+2)
        oof_preds[val_idx]/=5

        del model
        gc.collect()

    results/=K
    sub_df = pd.read_csv('./dataset/sample_submission.csv')
    sub_df["deal_probability"] = results
    sub_df.to_csv("baseline.csv.gz",index=False,compression='gzip')

    print('save oof ')
    tr_oof = pd.read_csv("./dataset/train.csv")
    tr_oof['TARGET_oof'] = oof_preds.copy()
    tr_oof.to_csv("./dataset/train.csv", index=False)

    te_oof = pd.read_csv("./dataset/test.csv")
    te_oof['TARGET_oof'] = results
    te_oof.to_csv("./dataset/test.csv", index=False)


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

        "price": np.float64,
        "item_seq_number": np.float32,


        "weekday": np.float16,
        "month": np.float16,
        "day": np.float16,
        "week": np.float16,
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
    feat = feat[-200:]
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
    import gc
    from sklearn.preprocessing import LabelEncoder
    from tqdm import tqdm

    usecols,dtypes = feature_selective()

    df = pd.read_csv("./dataset/train.csv",usecols=usecols+['deal_probability'],dtype=dtypes)
    test = pd.read_csv("./dataset/test.csv",usecols=usecols,dtype=dtypes)


    # df = pd.read_pickle("./dataset/train.pkl")[usecols+['deal_probability']]
    # test = pd.read_csv("./dataset/test.pkl")[usecols]
    # for col in usecols:
    #     df[col] = df[col].astype(dtypes[col])
    #     test[col] = test[col].astype(dtypes[col])

    for feat,k in zip(['description', 'title'],[800,500]):
        de_matrix = np.load('./dataset/' + feat + '.npy')
        de_matrix = de_matrix.astype(np.float16)
        de_matrix = de_matrix[:,:k]
        cols = [feat[:4] + "_tf_" + str(x) for x in range(k)]
        tr_de_matrix = pd.DataFrame(de_matrix[:len(df)],columns=cols)
        te_de_matrix = pd.DataFrame(de_matrix[len(df):],columns=cols)

        df = pd.concat([df,tr_de_matrix],axis=1)
        test = pd.concat([test,te_de_matrix],axis=1)
        del de_matrix,tr_de_matrix,te_de_matrix
        gc.collect()

    for name in ['tr','te']:
        for feat in ['u','i']:
            matrix = np.load('./dataset/'+name+'_'+feat+'_id.csv.npy')
            matrix = matrix.astype(np.float16)
            # matrix = matrix[:,-50:]
            matrix = pd.DataFrame(matrix,columns=[feat+'_'+str(i) for i in range(100)])
            if name == 'tr':
                df = pd.concat([df,matrix], axis=1)
            else:
                test = pd.concat([test,matrix], axis=1)


    print(df.info())
    from scipy.sparse import hstack, csr_matrix

    Y = df['deal_probability'].values
    df.drop(['deal_probability'],axis=1,inplace=True)
    main(df,Y,test,K=5)











