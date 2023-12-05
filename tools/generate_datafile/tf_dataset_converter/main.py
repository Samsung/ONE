################################################################################
# Parse arguments
################################################################################

from argparser import parse_args

# You can see arguments' information in argparser.py
args = parse_args()

if len(args.split) != len(args.length):
    print(f'length and split should have the same count')
    exit(1)

################################################################################
# Load a dataset of tensorflow
################################################################################

# Disable tensorflow cpp warning log
import os

FILTERING_WARNING = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = FILTERING_WARNING

from datasets import DatasetLoader
from pathlib import Path
import tensorflow as tf
import numpy as np
import re

ds_loader = DatasetLoader()

if args.show_datasets:
    print('Dataset list :')
    names = ',\n'.join(ds_loader.get_dataset_names())
    print(f'[{names}]')
    exit(0)

ds_loader.load(args.dataset_name, args.split, args.model)
ds_dict = ds_loader.prefetched_dataset()
ds_info = ds_loader.get_dataset_info()
print(f'class names[:10]  : {ds_loader.class_names(num=10)}')
print(f'class length      : {ds_loader.num_classes()}')

################################################################################
# Convert tensorlfow dataset to onert format
################################################################################
Path(f'{args.out_dir}').mkdir(parents=True, exist_ok=True)
prefix_name = f'{args.out_dir}/{args.prefix_name}'
if args.prefix_name != '':
    prefix_name += '.'

split_length = dict(zip(args.split, args.length))

for key in split_length.keys():
    if key not in ds_info.keys():
        print(f'Oops! The given split is not included in {args.dataset_name}')
        exit(1)

for key in split_length.keys():
    if ds_info[key] < split_length[key]:
        print(f'Oops! The number of data in the dataset is less than {v}')
        exit(1)


def _only_image(image, _):
    return image


def _only_label(_, label):
    return label


def _label_to_array(label):
    arr = np.zeros(ds_loader.num_classes(), dtype=float)
    arr[label] = 1.
    tensor = tf.convert_to_tensor(arr, tf.float32)
    return tensor


def _replace_key(key):
    for ch in ['[', ':', ']']:
        key = key.replace(ch, '_')
    return key


file_path_list = []
for k, v in split_length.items():
    _k = _replace_key(k)
    file_path_list.append(f'{prefix_name}{_k}.input.{v}.bin')
    file_path_list.append(f'{prefix_name}{_k}.output.{v}.bin')

ds_list = []
for i, v in enumerate(split_length.values()):
    ds_list.append(ds_dict[i].take(v).map(_only_image)),
    ds_list.append(
        [_label_to_array(label) for label in ds_dict[i].take(v).map(_only_label)])

if len(file_path_list) != len(ds_list):
    print(f'file_path_list and ds_list should have the same length')
    exit(1)

for i, file_path in enumerate(file_path_list):
    with open(file_path, 'wb') as f:
        ds = ds_list[i]
        for tensor in ds:
            f.write(tensor.numpy().tobytes())
        f.close()

print('The data files are created!')
