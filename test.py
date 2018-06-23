import numpy as np

import pandas as pd
from tqdm import tqdm

# a = pd.read_csv('baseline.csv.gz')
#
# a.loc[a['deal_probability']>1,'deal_probability'] = 1
# a.loc[a['deal_probability']<0,'deal_probability'] = 0
#
#
# a.to_csv('baseline1.csv.gz',index=False,compression='gzip')



# tr = pd.read_csv('./dataset/train.csv',usecols=['deal_probability'])
# train_active = pd.read_csv('./dataset/train_active.csv')
# test_sub = pd.read_csv('./dataset/test.csv')
# test_active = pd.read_csv('./dataset/test_active.csv')

# value = np.sort(tr['deal_probability'].unique())
# print(value)

from math import fabs
sub1= pd.read_csv('./dataset/2209.csv')
sub2= pd.read_csv('./dataset/2204.csv.gz')
sub3 = pd.read_csv('./dataset/2212.csv')




sub1['deal_probability'] = 0.4*sub1['deal_probability'] + 0.6*sub2['deal_probability']
sub1.to_csv('baseline.csv.gz',index=False,compression='gzip')





