################################################################################
# Parse arguments
################################################################################

from argparser import parse_args

# You can see arguments' information in argparser.py
args = parse_args()

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

ds_loader = DatasetLoader()

if args.show_datasets:
    print('Dataset list :')
    names = ',\n'.join(ds_loader.get_dataset_names())
    print(f'[{names}]')
    exit(0)

ds_loader.load(args.dataset_name)
ds_train, ds_test = ds_loader.prefetched_datasets()
nums_train_ds = ds_loader.get_num_train_examples()
nums_test_ds = ds_loader.get_num_test_examples()
print(f'class names       : {ds_loader.class_names()}')
print(f'train dataset len : {nums_train_ds}')
print(f'test dataset len  : {nums_test_ds}')

################################################################################
# Convert tensorlfow dataset to onert format
################################################################################
Path(f'{args.out_dir}').mkdir(parents=True, exist_ok=True)
prefix_name = f'{args.out_dir}/{args.prefix_name}'
if args.prefix_name != '':
    prefix_name += '.'

nums_train = args.train_length
if (nums_train > nums_train_ds):
    print(
        f'Oops! The number of data for training in the dataset is less than {nums_train}')
    exit(1)

nums_test = args.test_length
if (nums_test > nums_test_ds):
    print(f'Oops! The number of data for test in the dataset is less than {nums_test}')
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


file_path_list = [
    f'{prefix_name}train.input.{nums_train}.bin',
    f'{prefix_name}test.input.{nums_test}.bin',
    f'{prefix_name}train.output.{nums_train}.bin',
    f'{prefix_name}test.output.{nums_test}.bin'
]

ds_list = [
    ds_train.take(nums_train).map(_only_image),
    ds_test.take(nums_test).map(_only_image),
    [_label_to_array(label) for label in ds_train.take(nums_train).map(_only_label)],
    [_label_to_array(label) for label in ds_test.take(nums_test).map(_only_label)]
]

for i in range(4):
    file_path = file_path_list[i]
    with open(file_path, 'wb') as f:
        ds = ds_list[i]
        for tensor in ds:
            f.write(tensor.numpy().tobytes())
        f.close()

print('The data files are created!')
