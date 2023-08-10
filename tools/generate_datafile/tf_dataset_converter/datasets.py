'''Deal with the Fashion-MNIST dataset.'''

import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import urllib3

dataset_root_dir = Path(__file__).parent.absolute() / 'data'


class DatasetLoader():
    '''
  Loader of the tensorflow dataset
  '''

    def __init__(self):
        # Set not to verify certificate when downloading dataset so that it works in github action
        # as well.
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def load(self, dataset_name):
        download_config = tfds.download.DownloadConfig(verify_ssl=False)
        download_and_prepare_kwargs = {'download_config': download_config}

        (ds_train, ds_test), ds_info = tfds.load(
            dataset_name,
            split=['train', 'test'],
            data_dir=dataset_root_dir,
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            download_and_prepare_kwargs=download_and_prepare_kwargs,
        )

        self.ds_info = ds_info

        def _normalize_img(image, label):
            """Normalizes images: `uint8` -> `float32`."""
            return tf.cast(image, tf.float32) / 255., label

        self.ds_train = ds_train.map(_normalize_img)
        self.ds_test = ds_test.map(_normalize_img)

        for images, labels in ds_train:
            print(f'Shape of images : {images.shape}')
            print(f'Shape of labels: {labels.shape} {labels.dtype}')
            break

    def get_dataset_names(self):
        return tfds.list_builders()

    def class_names(self):
        '''
    Get class names
    '''
        return self.ds_info.features['label'].names

    def num_classes(self):
        '''
    Get the number of classes
    '''
        return self.ds_info.features['label'].num_classes

    def get_num_train_examples(self):
        return self.ds_info.splits['train'].num_examples

    def get_num_test_examples(self):
        return self.ds_info.splits['test'].num_examples

    def prefetched_datasets(self):
        '''
    get prefetched datasets for traning.

    Return:
      Datasets for training and testing.
    '''

        train_dataset = self.ds_train.cache()
        train_dataset = train_dataset.shuffle(self.ds_info.splits['train'].num_examples)

        test_dataset = self.ds_train.cache()

        # return train_dataset, test_dataset
        return self.ds_train.cache(), self.ds_test.cache()
