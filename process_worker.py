import pandas as pd
from tqdm import tqdm
from itertools import combinations,product
from tool import co_prob,condition_prob
import gc
import numpy as np

def prob_feature_worker(usecols,features,is_co_prob):

    train = pd.read_csv('./dataset/train.csv', usecols=usecols, parse_dates=["activation_date"])
    train_active = pd.read_csv('./dataset/train_active.csv', usecols=usecols, parse_dates=["activation_date"])
    test = pd.read_csv('./dataset/test.csv', usecols=usecols, parse_dates=["activation_date"])
    test_active = pd.read_csv('./dataset/test_active.csv', usecols=usecols, parse_dates=["activation_date"])

    dataset = pd.concat([
        train,
        test,
        train_active,
        test_active
    ]).reset_index(drop=True)

    del test_active, train_active
    gc.collect()

    gen_cols = []
    if is_co_prob:
        for cols in tqdm(features):
            name = "co_"+"_".join(cols)
            gen_cols.append(name)
            sub_cols = list(set(list(cols)+['user_id', 'item_id']))
            feat = co_prob(dataset[sub_cols],cols,name)
            train = train.merge(feat[['user_id', 'item_id', name]], on=['user_id', 'item_id'], how='left')
            test = test.merge(feat[['user_id', 'item_id', name]], on=['user_id', 'item_id'], how='left')
            gc.collect()
    else:
        for cols_x,cols_y in tqdm(features):
            name = "_".join(cols_y) + "_by_"+"_".join(cols_x)
            gen_cols.append(name)
            cols = list(set(list(cols_x) + list(cols_y) + ['user_id','item_id']))
            feat = condition_prob(dataset[cols],cols_y,cols_x,name)
            train = train.merge(feat[['user_id','item_id',name]], on=['user_id', 'item_id'], how='left')
            test = test.merge(feat[['user_id','item_id',name]], on=['user_id', 'item_id'], how='left')
            gc.collect()
    return train[gen_cols],test[gen_cols]

def conti_stat_feature_worker(usecols,features):
    from tool import condition_stat

    train = pd.read_csv('./dataset/train.csv', usecols=usecols, parse_dates=["activation_date"])
    test = pd.read_csv('./dataset/test.csv', usecols=usecols, parse_dates=["activation_date"])
    train_active = pd.read_csv('./dataset/train_active.csv', usecols=usecols, parse_dates=["activation_date"])
    test_active = pd.read_csv('./dataset/test_active.csv', usecols=usecols, parse_dates=["activation_date"])

    dataset = pd.concat([
        train,
        test,
        train_active,
        test_active
    ]).reset_index(drop=True)

    del test_active, train_active
    gc.collect()

    cols = []
    for conti_feat,comb_feat in tqdm(features):
        sub_cols = list(set([conti_feat]+comb_feat+['user_id', 'item_id']))
        all_samples = dataset[sub_cols]

        feat_mean = conti_feat + '_mean_' + '_'.join(comb_feat)
        all_samples = condition_stat(all_samples, conti_feat, comb_feat, feat_mean, 'mean')

        feat_std = conti_feat + '_std_' + '_'.join(comb_feat)
        all_samples = condition_stat(all_samples, conti_feat, comb_feat, feat_std, 'std')

        train = train.merge(all_samples[[feat_mean, feat_std, 'user_id', 'item_id']],
                                            on=['user_id', 'item_id'], how='left')
        test = test.merge(all_samples[[feat_mean, feat_std,'user_id', 'item_id']],
                                            on=['user_id', 'item_id'], how='left')

        feat_norm = conti_feat + '_norm_' + '_'.join(comb_feat)
        train[feat_norm] = (train[conti_feat] - train[feat_mean])/train[feat_std]
        test[feat_norm] = (test[conti_feat] - test[feat_mean]) / test[feat_std]

        cols.extend([feat_mean,feat_std,feat_norm])

        del all_samples
        gc.collect()

    return train[cols],test[cols]


def tfidf_worker(feat,k,train_text,test_text):
    from sklearn.decomposition import KernelPCA, PCA, TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from nltk.corpus import stopwords

    def pca_compression(tfidf_matrix, n_components):
        model = TruncatedSVD(n_components=n_components,algorithm='arpack',n_iter=20)
        de_matrix = model.fit_transform(tfidf_matrix)

        total_ratio = []
        ratio = 0
        for i in model.explained_variance_ratio_:
            ratio+=i
            total_ratio.append(ratio)
        print(total_ratio)
        print(feat)
        return de_matrix

    text = train_text.fillna('_unk_').values.tolist() + \
           test_text.fillna('_unk_').values.tolist()

    tfv = TfidfVectorizer(max_features=15000,strip_accents='unicode', analyzer='char',
                          min_df=15,stop_words=set(stopwords.words('russian')),max_df = 0.4,
                          ngram_range=(1,2),use_idf=1, smooth_idf=False, sublinear_tf=True)
    tfidf_feat = tfv.fit_transform(text)

    # 获取pca后的np
    de_matrix = pca_compression(tfidf_feat, n_components=k)
    np.save('./dataset/'+feat+'.npy',de_matrix)

    return True

def img_info_worker(img_files,path):
    from img_tool import  perform_color_analysis,average_pixel_width,get_average_color,get_dominant_color,get_blurrness_score

    img_samples = []
    for i,img_f in tqdm(enumerate(img_files)):

        img_feat = []
        if isinstance(img_f,str):
            img_path = path+img_f+'.jpg'
            light_percent, dark_percent = perform_color_analysis(img_path,'all')
            img_feat += [light_percent,dark_percent]
            img_feat.append(get_average_color(img_path))
            img_feat.append(get_blurrness_score(img_path))
            img_feat.append(get_dominant_color(img_path))
            img_feat.append(average_pixel_width(img_path))

        img_samples.append(img_feat)

    return img_samples

def img_confi_worker(img_files,model,path):
    if model == 'resnet50':
        import keras.applications.resnet50 as resnet50
        model = resnet50.ResNet50(weights='imagenet')
        pak = resnet50
    elif model == 'xception':
        import keras.applications.xception as xception
        model = xception.Xception(weights='imagenet')
        pak = xception
    elif model == 'inception_v3':
        import keras.applications.inception_v3 as inception_v3
        model = inception_v3.InceptionV3(weights='imagenet')
        pak = inception_v3
    else:
        raise RuntimeError("don't have this model")
    from img_tool import image_classify

    img_confi = []

    for i,img_f in tqdm(enumerate(img_files)):
        img_feat = []
        if isinstance(img_f, str):
            img_feat+=list(image_classify(model,pak,path+img_f+'.jpg'))
        img_confi.append(img_feat)

    return img_confi












