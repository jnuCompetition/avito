


USE_FEAT = [
    'item_seq_number_mean_param_1_user_type',
    'user_id_by_user_type',
    'co_category_name_user_id',
    'price_norm_param_1_activation_date',
    'item_seq_number_mean_category_name_user_type',
    'activation_date_user_id_by_user_type',
]


img_path = './dataset/img/'
img_size = 64
title_len = 7
desc_len = 45


"""
region                       28
city                       1752
parent_category_name          9
category_name                47
param_1                     371
param_2                     277
param_3                    1276
title                   1022203
description             1793972
activation_date              30
user_type                     3
image                   1856665
image_top_1                3063
param_combined             2402

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


num leaves      lr        bin       bag frac       feat frac
800            0.017      255       0.9            0.4
1000           0.017      255       0.8            0.4
750            0.02       50        0.9            0.4
750            0.017      255       0.9            0.75


"""

















