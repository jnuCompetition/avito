
import pandas as pd


import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold
import numpy as np

NFOLDS = 10
SEED = 42

def main(train_df,Y,test_df,K,sparse):

    kf = KFold(len(train_df), n_folds=K, shuffle=True,random_state=42)

    param = {
        'objective' : 'regression',
        'boosting_type': 'gbdt',
        'metric' : 'rmse',
        # 'max_bin':50,
        'num_leaves': 800,
        'feature_fraction': 0.4,
        'bagging_fraction': 0.9,
        'bagging_freq': 2,
        'learning_rate': 0.017,
        # 'device':'gpu',
        # 'max_depth':15,

        'lambda_l1':0,
        'lambda_l2':0,
        'nthread':30,
    }

    results = np.zeros((len(test_df,)))

    for i, (tr_idx, val_idx) in enumerate(kf):
        print('第{}次训练...'.format(i))

        if sparse:
            train_X = train_df[tr_idx]
            val_X = train_df[val_idx]
        else:
            train_X = train_df.loc[tr_idx,:]
            val_X = train_df.loc[val_idx, :]

        train_Y = Y[tr_idx]
        train_dataset = lgb.Dataset(train_X,train_Y)

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

        test_pred = model.predict(test_df,num_iteration=model.best_iteration)

        test_pred[test_pred<0] = 0
        test_pred[test_pred>1] = 1
        results += test_pred

    results/=K
    sub_df = pd.read_csv('./dataset/sample_submission.csv')
    sub_df["deal_probability"] = results
    sub_df.to_csv("baseline.csv.gz",index=False,compression='gzip')

def main_sparse(train_df,Y,test_df,K,cate,cols):
    kf = KFold(train_df.shape[0],n_folds=K, shuffle=True,random_state=42)

    param = {
        'objective' : 'regression',
        'boosting_type': 'gbdt',
        'metric' : 'rmse',
        # 'max_bin':50,
        'num_leaves': 512,
        'feature_fraction': 0.4,
        'bagging_fraction': 0.9,
        'bagging_freq': 2,
        'learning_rate': 0.016,
        # 'device':'gpu',
        # 'max_depth':15,
        'lambda_l1':0,
        'lambda_l2':0,
        'nthread':30,
    }

    results = np.zeros((train_df.shape[0]))

    for i, (tr_idx, val_idx) in enumerate(kf):
        print('第{}次训练...'.format(i))
        print(tr_idx,tr_idx.dtype)

        train_X = train_df[tr_idx]
        val_X = train_df[val_idx]

        train_Y = Y[tr_idx]
        train_dataset = lgb.Dataset(train_X,train_Y,categorical_feature=cate,feature_name=cols)

        val_Y = Y[val_idx]
        val_dataset = lgb.Dataset(val_X,val_Y,categorical_feature=cate,feature_name=cols)

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

        test_pred = model.predict(test_df,num_iteration=model.best_iteration)

        test_pred[test_pred<0] = 0
        test_pred[test_pred>1] = 1
        results += test_pred

    results/=K
    sub_df = pd.read_csv('./dataset/sample_submission.csv')
    sub_df["deal_probability"] = results
    sub_df.to_csv("baseline.csv.gz",index=False,compression='gzip')

if __name__ == '__main__':
    import gc
    from sklearn.preprocessing import LabelEncoder
    from tqdm import tqdm
    from tool import feature_selective
    from scipy.sparse import load_npz,hstack,csr_matrix

    usecols,dtypes = feature_selective()
    for col in usecols:
        if dtypes[col] == 'category':
            dtypes[col] = str
        elif dtypes[col] == np.float16:
            dtypes[col] = np.float32

    df = pd.read_csv("./dataset/train.csv",usecols=usecols+['deal_probability'],dtype=dtypes)
    test = pd.read_csv("./dataset/test.csv",usecols=usecols,dtype=dtypes)


    # df = pd.read_pickle("./dataset/train.pkl")[usecols+['deal_probability']]
    # test = pd.read_csv("./dataset/test.pkl")[usecols]
    # for col in usecols:
    #     df[col] = df[col].astype(dtypes[col])
    #     test[col] = test[col].astype(dtypes[col])

    # for feat,k in zip(['description', 'title'],[1000,500]):
    #     de_matrix = np.load('./dataset/pca' + feat + '_word.npy')
    #     de_matrix = de_matrix.astype(np.float32)
    #     de_matrix = de_matrix[:,:k]
    #     cols = [feat[:4] + "_tf_" + str(x) for x in range(k)]
    #     tr_de_matrix = pd.DataFrame(de_matrix[:len(df)],columns=cols)
    #     te_de_matrix = pd.DataFrame(de_matrix[len(df):],columns=cols)
    #
    #     df = pd.concat([df,tr_de_matrix],axis=1)
    #     test = pd.concat([test,te_de_matrix],axis=1)
    #     del de_matrix,tr_de_matrix,te_de_matrix
    #     gc.collect()

    Y = df['deal_probability'].values
    df.drop(['deal_probability'], axis=1, inplace=True)
    for name in ['tr','te']:
        for feat in ['u','i']:
            matrix = np.load('./dataset/'+name+'_'+feat+'_id.csv.npy')
            matrix = matrix.astype(np.float32)
            # matrix = matrix[:,-50:]
            matrix = pd.DataFrame(matrix,columns=[feat+'_'+str(i) for i in range(100)])
            if name == 'tr':
                df = pd.concat([df,matrix], axis=1)
            else:
                test = pd.concat([test,matrix], axis=1)
    cols = list(df.columns)
    num_tr = len(df)
    all_samples = df.append(test).reset_index(drop=True)

    print('prepocess')
    le = LabelEncoder()
    cate = []
    for col in tqdm(usecols):
        if dtypes[col] == str:
            all_samples[col].fillna('unk', inplace=True)
            all_samples[col] = le.fit_transform(all_samples[col].values)
            cate.append(col)

    df = all_samples.values[:num_tr]
    test = all_samples.values[num_tr:]

    print('combine sparse')
    desc = load_npz('./dataset/sparsedescription.npz').astype(np.float32)
    title = load_npz('./dataset/sparsetitle.npz').astype(np.float32)

    df = hstack([csr_matrix(df),desc[0:num_tr],title[0:num_tr]])
    df = df.tocsr()
    test = hstack([csr_matrix(test),desc[num_tr:],title[num_tr:]])
    test = test.tocsr()
    print('train')
    cols += ['desc' + str(i) for i in range(desc.shape[1])]
    cols += ['title' + str(i) for i in range(title.shape[1])]
    del desc,title

    # main(df,Y,test,K=5,sparse=True)
    main_sparse(df,Y,test,5,cate,cols)










