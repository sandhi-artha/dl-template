'''Model config in json format'''

CFG = {
    'data': {
        'ds_name': 'mnist',
        'split': ['train', 'test'],
        'ds_path': 'D:\\Projects\\python\\dl-temp\\data',
        'shuffle_files': True,
        'as_supervised': True,
        'load_with_info': True,
        'image_size': (28,28,1),
    },
    'train': {
        'batch_size': 128,
        'buffer_size': 1000,
        'epochs': 20,
        'seed': 17,
        'lr': 1e-3,
    },
    'model': {
        'conv_size': [256, 64],
        'output': 1
    }
}