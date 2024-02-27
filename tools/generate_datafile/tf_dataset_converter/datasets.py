'''Deal with the tensorflow dataset.'''

import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path

dataset_root_dir = Path(__file__).parent.absolute() / 'data'


class Mnist():
    def check(dataset_name):
        supported_dataset_list = ['fashion_mnist', 'mnist']
        if dataset_name in supported_dataset_list:
            return
        print(f'{dataset_name} does not fit Mnist')
        exit(1)

    def preprocess_input(image, label):
        """Preprocess input data for Mnist."""

        def _normalize_img(image):
            """Normalize images: `uint8` -> `float32`."""
            return tf.cast(image, tf.float32) / 255.

        return _normalize_img(image), label


class MobileNetV2():
    def check(dataset_name):
        supported_dataset_list = ['imagenet_a']
        if dataset_name in supported_dataset_list:
            return
        print(f'{dataset_name} does not fit MobileNetV2')
        exit(1)

    def preprocess_input(image, label):
        """Preprocess input data for MobileNetV2."""

        def _resize_img(image):
            _image = tf.cast(image, tf.float32) / 255.
            _image = tf.image.resize_with_crop_or_pad(_image, 224, 224)
            return _image

        return _resize_img(image), label


class DatasetLoader():
    '''
    Loader of tensorflow datasets
    '''

    def load(self, dataset_name, splits, model_name):
        ds_dict, ds_info = tfds.load(
            dataset_name,
            split=splits,
            data_dir=dataset_root_dir,
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        self.ds_info = ds_info
        self.ds_dict = []
        if model_name == 'mnist':
            Mnist.check(dataset_name)
            self.ds_dict = [data.map(Mnist.preprocess_input) for data in ds_dict]
        elif model_name == 'mobilenetv2':
            MobileNetV2.check(dataset_name)
            self.ds_dict = [data.map(MobileNetV2.preprocess_input) for data in ds_dict]

        for images, labels in self.ds_dict[0]:
            print(f'Shape of images : {images.shape}')
            print(f'Shape of labels: {labels.shape} {labels.dtype}')
            break

    def get_dataset_names(self):
        return tfds.list_builders()

    def class_names(self, num=10):
        '''
        Get class names
        '''
        return self.ds_info.features['label'].names[:num]

    def num_classes(self):
        '''
        Get the number of classes
        '''
        return self.ds_info.features['label'].num_classes

    def get_dataset_info(self):
        '''
        Get examples for each data
        '''
        dict_num = {}
        for key in self.ds_info.splits.keys():
            dict_num[key] = self.ds_info.splits[key].num_examples
        return dict_num

    def prefetched_dataset(self):
        '''
        get prefetched datasets for traning.

        Return:
           Datasets for training and testing.
        '''
        ds_dict = [d.cache() for d in self.ds_dict]
        return ds_dict
