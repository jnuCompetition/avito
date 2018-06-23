import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold
import numpy as np

NFOLDS = 10
SEED = 42

def main_sparse(train_df,Y,test_df,K,cate,cols):
    kf = KFold(train_df.shape[0],n_folds=K, shuffle=True,random_state=42)

    param = {
        'objective' : 'regression',
        'boosting_type': 'gbdt',
        'metric' : 'rmse',
        # 'max_bin':50,
        'num_leaves': 800,
        'feature_fraction': 0.4,
        'bagging_fraction': 0.9,
        'bagging_freq': 2,
        'learning_rate': 0.016,
        # 'device':'gpu',
        # 'max_depth':15,
        'lambda_l1':0,
        'lambda_l2':0,
        'nthread':32,
    }

    results = np.zeros(test_df.shape[0])

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



    df = pd.read_csv("./dataset/train.csv",usecols=usecols+['deal_probability'],dtype=dtypes)
    test = pd.read_csv("./dataset/test.csv",usecols=usecols,dtype=dtypes)

    Y = df['deal_probability'].values
    df.drop(['deal_probability'], axis=1, inplace=True)
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
    # for feat,k in zip(['description', 'title'],[100,100]):
    #     de_matrix = np.load('./dataset/pca' + feat + '_word.npy')
    #     de_matrix = de_matrix.astype(np.float16)
    #     de_matrix = de_matrix[:,:k]
    #     cols = [feat[:4] + "_tfword_" + str(x) for x in range(k)]
    #     tr_de_matrix = pd.DataFrame(de_matrix[:len(df)],columns=cols)
    #     te_de_matrix = pd.DataFrame(de_matrix[len(df):],columns=cols)
    #     df = pd.concat([df,tr_de_matrix],axis=1)
    #     test = pd.concat([test,te_de_matrix],axis=1)
    #     del de_matrix,tr_de_matrix,te_de_matrix
    #     gc.collect()
    #
    # for feat,k in zip(['description', 'title'],[100,100]):
    #     de_matrix = np.load('./dataset/pca' + feat + '_char.npy').astype(np.float16)
    #     de_matrix = de_matrix[:, :k]
    #     cols = [feat[:4] + "_tfchar_" + str(x) for x in range(k)]
    #     tr_de_matrix = pd.DataFrame(de_matrix[:len(df)],columns=cols)
    #     te_de_matrix = pd.DataFrame(de_matrix[len(df):],columns=cols)
    #     df = pd.concat([df,tr_de_matrix],axis=1)
    #     test = pd.concat([test,te_de_matrix],axis=1)
    #     del de_matrix,tr_de_matrix,te_de_matrix
    #     gc.collect()

    # img_feat = np.load('./dataset/img_feat_demen.npy').astype(np.float16)
    # cols = ["img" + str(x) for x in range(512)]
    # tr_feat= pd.DataFrame(img_feat[:len(df)], columns=cols)
    # te_feat = pd.DataFrame(img_feat[len(df):], columns=cols)
    # df = pd.concat([df, tr_feat], axis=1)
    # test = pd.concat([test, te_feat], axis=1)
    # del img_feat,tr_feat,te_feat
    # gc.collect()

    cols = list(df.columns)
    num_tr = len(df)
    all_samples = df.append(test).reset_index(drop=True)

    print('prepocess')
    le = LabelEncoder()
    cate = []
    for col in tqdm(usecols):
        if dtypes[col] == str:
            all_samples[col].fillna('unk', inplace=True)
            all_samples[col] = le.fit_transform(all_samples[col].values).astype(np.uint16)
            cate.append(col)

    df = all_samples.values[:num_tr]
    test = all_samples.values[num_tr:]
    #
    print('combine sparse')
    desc = load_npz('./dataset/sparsedescription.npz').astype(np.float16)
    title = load_npz('./dataset/sparsetitle.npz').astype(np.float16)
    desc3 = load_npz('./dataset/sparse3descriptionword.npz').astype(np.float16)
    title3 = load_npz('./dataset/sparse3titleword.npz').astype(np.float16)
    param= load_npz('./dataset/sparseparam_combined.npz').astype(np.float16)
    descchar = load_npz('./dataset/sparsedescriptionchar.npz').astype(np.float16)
    titlechar = load_npz('./dataset/sparsetitlechar.npz').astype(np.float16)

    df = hstack([csr_matrix(df),
                 desc[:num_tr],title[:num_tr],param[:num_tr],
                 desc3[:num_tr], title3[:num_tr],
                 descchar[:num_tr],titlechar[:num_tr]])
    df = df.tocsr()
    test = hstack([csr_matrix(test),
                   desc[num_tr:],title[num_tr:],param[num_tr:],
                   desc3[num_tr:], title3[num_tr:],
                   descchar[num_tr:],titlechar[num_tr:]])
    test = test.tocsr()
    # df = csr_matrix(df)
    # test = csr_matrix(test)
    print('train')
    cols += ['desc' + str(i) for i in range(desc.shape[1])]
    cols += ['title' + str(i) for i in range(title.shape[1])]
    cols += ['desc3' + str(i) for i in range(desc3.shape[1])]
    cols += ['title3' + str(i) for i in range(title3.shape[1])]
    cols += ['param' + str(i) for i in range(param.shape[1])]
    cols += ['descchar' + str(i) for i in range(descchar.shape[1])]
    cols += ['titlechar' + str(i) for i in range(titlechar.shape[1])]
    del desc,title,param,descchar,titlechar,all_samples,desc3,title3
    gc.collect()
    # main(df,Y,test,K=5,sparse=True)
    main_sparse(df,Y,test,5,cate,cols)










