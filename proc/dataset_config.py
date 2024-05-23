#!/user/bin/env python3
# -*- coding: utf-8 -*-

def get_dataset():
    dataset = 'mr'
    # dataset = 'ohsumed'
    # dataset = '20ng'
    # dataset = 'R8'
    # dataset = 'R52'
    # dataset = 'AGNews'
    # dataset = 'SST2'

    return dataset


# static params
data_args = {
    '20ng':
        {'train_size': 6794,  # 11314,
         'test_size': 4442,  # 7532,
         'valid_size': 679,  # 1131,
         "num_classes": 20
         },
    'mr':
        {'train_size': 7108,
         'test_size': 3554,
         'valid_size': 711,
         "num_classes": 2,
         },
    'ohsumed':
        {'train_size': 3357,
         'test_size': 4043,
         'valid_size': 336,
         "num_classes": 23,
         },
    'R8':
        {'train_size': 5485,
         'test_size': 2189,
         'valid_size': 548,
         "num_classes": 8
         },
    'R52':
        {'train_size': 6532,
         'test_size': 2568,
         'valid_size': 653,
         "num_classes": 52
         },
    'AGNews':
        {'train_size': 6000,
         'test_size': 3000,
         'valid_size': 600,
         "num_classes": 4
         },
    'WebKB':
        {'train_size': 2777,
         'test_size': 1376,
         'valid_size': 277,
         'num_classes': 4
         },
    'dblp':
        {'train_size': 12000,
         'test_size': 3000,
         'valid_size': 1200,
         'num_classes': 6
         },
    'SST2':
        {'train_size': 7792,
         'test_size': 1821,
         'valid_size': 779,
         'num_classes': 2
         },
}
