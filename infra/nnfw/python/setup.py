from setuptools import setup, find_packages
import os
import shutil

source_directory = '../../../Product/x86_64-linux.debug/out/lib'
package_directory = 'nnfwapi'

if os.path.exists(package_directory):
    shutil.rmtree(package_directory)

os.makedirs(package_directory)

for filename in os.listdir(source_directory):
    source_file = os.path.join(source_directory, filename)
    if os.path.isfile(source_file) and source_file.endswith('.so'):
        shutil.copy(source_file, package_directory)

setup(
    name=package_directory,
    version='1.0.0',
    description='nnfw_api_pybind binding',
    long_description='you can use nnfw api by python',
    packages=[package_directory],
    package_data={package_directory: ['nnfw_api_pybind.cpython-38-x86_64-linux-gnu.so', 'libnnfw-dev.so', 'libonert_core.so', 'libbackend_cpu.so']},
)
