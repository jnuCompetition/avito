import pandas as pd
from tqdm import tqdm
from text_embedding import pocess_text
from config import img_path
import cv2
import numpy as np
from keras.layers import Embedding,Input,concatenate,Dense,Flatten,CuDNNGRU,Bidirectional
from keras.layers import Conv2D,GlobalMaxPooling1D,GlobalAveragePooling1D,Activation
from keras.layers import MaxPool2D,GlobalAveragePooling2D,BatchNormalization,Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Nadam
from config import img_size,title_len,desc_len
from keras import backend as K
from sklearn.cross_validation import KFold

def get_data():
    from tool import feature_selective
    from sklearn.preprocessing import LabelEncoder

    usecols, dtypes = feature_selective()
    usecols += ['description', 'title','image']
    dtypes['description'] = 'str'
    dtypes['title'] = 'str'
    dtypes['image'] = 'str'
    for col in usecols:
        dtypes[col] = str if dtypes[col] == 'category' else dtypes[col]



    print('read data ..')
    train = pd.read_csv("./dataset/train.csv", usecols=usecols+['deal_probability'], dtype=dtypes)
    test = pd.read_csv("./dataset/test.csv", usecols=usecols, dtype=dtypes)

    Y = train['deal_probability']
    train.drop(['deal_probability'],axis=1,inplace=True)

    num_tr = len(train)
    del train,test
    all_samples = train.append(test).reset_index(drop=True)


    print('prepocess')
    le = LabelEncoder()
    conti = []
    cate = []
    for col in tqdm(usecols):
        if dtypes[col] not in ['category','str']:
            all_samples[col].fillna(all_samples[col].median(),inplace=True)
            all_samples[col] = (all_samples[col] - all_samples[col].mean())/all_samples[col].std()
            conti.append(col)
        elif dtypes[col] == 'category' and col not in ['description', 'title','image']:
            all_samples[col].fillna('unk',inplace=True)
            all_samples[col] = le.fit_transform(all_samples[col].values)
            cate.append(col)
    cate.append('image')
    all_samples['image'].fillna('unk',inplace=True)
    all_samples['image'] = img_path+all_samples['image']+'.jpg'

    desc, desc_embed, title, title_embed = pocess_text(all_samples['description', 'title'])

    embed = {'desc':desc_embed,'title':title_embed}
    tr = {}
    te = {}
    for col in cate:
        tr[col] = all_samples[col].values[:num_tr]
        te[col] = all_samples[col].values[num_tr:]
    tr['conti'] = all_samples[conti].values[:num_tr]
    te['conti'] = all_samples[conti].values[num_tr:]
    tr['desc'] = desc[:num_tr]
    te['desc'] = desc[num_tr:]
    tr['title'] = title[:num_tr]
    te['title'] = title[num_tr:]

    print('read image')
    imgs = []
    for i in tqdm(all_samples['image'].values):
        imgs.append(cv2.imread(i))
    imgs = np.array(imgs)
    tr['image'] = imgs[:num_tr]
    te['image'] = imgs[num_tr:]

    return tr,Y,te,embed


def get_model(embed,num_conti):
    def block_wrap(x,filters):
        x = Conv2D(filters,kernel_size=3,padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_true - y_pred)))

    region = Input(shape=[1], name="region")
    city = Input(shape=[1], name="city")
    pcn = Input(shape=[1], name="parent_category_name")
    cn = Input(shape=[1], name="category_name")
    para1 = Input(shape=[1], name="param_1")
    para2 = Input(shape=[1], name="param_2")
    para3 = Input(shape=[1], name="param_3")
    act = Input(shape=[1], name="activation_date")
    ut = Input(shape=[1], name="user_type")
    img_top = Input(shape=[1], name="image_top_1")

    title = Input(shape=[title_len], name="title")
    desc = Input(shape=[desc_len], name="desc")
    img = Input(shape=[img_size,img_size,3],name='image')
    conti = Input(shape=[num_conti],name='conti')

    """
    region                       28
    city                       1752
    parent_category_name          9
    category_name                47
    param_1                     372
    param_2                     278
    param_3                    1277
    title                   1022203
    description             1793973
    activation_date              30
    user_type                     3
    image                   1856666
    image_top_1                3064
    param_combined             2402
    """

    emb_region = Embedding(28,(8,))(region)
    emb_city = Embedding(1752,(16,))(city)
    emb_pcn = Embedding(9,(3,))(pcn)
    emb_cn = Embedding(47,(8,))(cn)
    emb_para1 = Embedding(372,(16,))(para1)
    emb_para2 = Embedding(278,(16,))(para2)
    emb_para3 = Embedding(1277,(16,))(para3)
    emb_act = Embedding(30,(8,))(act)
    emb_img_top = Embedding(3064,(32,))(img_top)
    emb_ut = Embedding(3,(3,),weights=[np.eye(3,3)],trainable=False)(ut)

    num_word = len(embed['title'])
    emb_title = Embedding(num_word,(title_len,),weights=[embed['title']],trainable=False)(title)
    num_word = len(embed['desc'])
    emb_desc = Embedding(num_word,(desc_len,),weights=[embed['desc']],trainable=False)(desc)

    conv = block_wrap(img,32)
    conv = MaxPool2D(padding='same')(conv)
    conv = block_wrap(conv,64)
    conv = MaxPool2D(padding='same')(conv)
    conv = block_wrap(conv,128)
    conv = MaxPool2D(padding='same')(conv)
    conv = GlobalAveragePooling2D()(conv)


    title_gru = Bidirectional(CuDNNGRU(64,return_sequences=True),merge_mode='sum')(emb_title)
    title_gru = Bidirectional(CuDNNGRU(32,return_sequences=True),merge_mode='sum')(title_gru)
    title_gru1 = GlobalMaxPooling1D()(title_gru)
    title_gru2 = GlobalAveragePooling1D()(title_gru)

    desc_gru = Bidirectional(CuDNNGRU(64, return_sequences=True), merge_mode='sum')(emb_desc)
    desc_gru = Bidirectional(CuDNNGRU(64, return_sequences=True), merge_mode='sum')(desc_gru)
    desc_gru1 = GlobalMaxPooling1D()(desc_gru)
    desc_gru2 = GlobalAveragePooling1D()(desc_gru)

    fc = concatenate([
        Flatten()(emb_region),
        Flatten()(emb_city),
        Flatten()(emb_pcn),
        Flatten()(emb_cn),
        Flatten()(emb_para1),
        Flatten()(emb_para2),
        Flatten()(emb_para3),
        Flatten()(emb_act),
        Flatten()(emb_img_top),
        Flatten()(emb_ut),
        conti,
        title_gru1,
        title_gru2,
        desc_gru1,
        desc_gru2,
        conv
    ])
    fc = Dense(256,activation='relu')(fc)
    fc = Dropout(0.5)(fc)
    fc = Dense(1,activation='sigmoid')(fc)


    model = Model([
        region,
        city,
        pcn,
        cn,
        para1,
        para2,
        para3,
        act,
        ut,
        img_top,
        title,
        desc,
        img,
        conti,
    ], output = fc)
    model.compile(optimizer=Nadam(),loss="mean_squared_error",metrics=[root_mean_squared_error])

    return model

def split_data(data,idx):
    result = {}
    for key,item in data.items():
        result[key] = item[idx]
    return result

def main(batchsize,n_folds):

    tr, Y, te, embed = get_data()

    num_conti = len(tr['conti'][0])

    model = get_model(embed,num_conti)

    del embed

    kf = KFold(len(Y), n_folds=K, shuffle=True, random_state=42)

    results = np.zeros((len(te['conti'], )))

    for i, (tr_idx, val_idx) in enumerate(kf):
        print('第{}次训练...'.format(i))

        tr_X = split_data(tr,tr_idx)
        tr_Y = Y[tr_idx]

        val_X = split_data(tr,val_idx)
        val_Y = Y[val_idx]

        model.fit(tr_X,tr_Y,batchsize,epochs=20,callbacks=EarlyStopping(patience=1),validation_data=(val_X,val_Y))

        test_pred = model.predict(te)

        results += test_pred

    results /= n_folds
    sub_df = pd.read_csv('./dataset/sample_submission.csv')
    sub_df["deal_probability"] = results
    sub_df.to_csv("baseline.csv.gz", index=False, compression='gzip')

if __name__ == '__main__':

    main(6400,5)
















