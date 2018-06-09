

import pandas as pd


# a = pd.read_csv('baseline.csv.gz')
#
# a.loc[a['deal_probability']>1,'deal_probability'] = 1
# a.loc[a['deal_probability']<0,'deal_probability'] = 0
#
#
# a.to_csv('baseline1.csv.gz',index=False,compression='gzip')



train_sub = pd.read_csv('./dataset/train.csv',usecols=['param_combined'])
# train_active = pd.read_csv('./dataset/train_active.csv')
# test_sub = pd.read_csv('./dataset/test.csv')
# test_active = pd.read_csv('./dataset/test_active.csv')

print(len(set(train_sub['param_combined'])))
# for col in ['user_id','item_id']:
#     user_train = set(train_sub[col])
#     user_train_act= set(train_active[col])
#
#     user_test = set(test_sub[col])
#     user_test_act = set(test_active[col])
#
#     print("num ",col," train ",len(user_train))
#     print("num ", col, " train act ", len(user_train_act))
#     print(len(user_train.intersection(user_train_act)))
#
#     print("num ", col, " test ", len(user_test))
#     print("num ", col, " test act ", len(user_test_act))
#     print(len(user_test.intersection(user_test_act)))
#
#
#     print(len(user_test_act.intersection(user_train_act)))
#
#     print(len(user_train.intersection(user_test)))








