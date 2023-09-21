from setuptools import setup, find_packages
import os
import shutil

architecture_directory = ['x86_64', 'armv7l', 'aarch64']
package_directory = 'nnfwapi'
packaging_directory = ['build', 'dist', package_directory + '.egg-info']

try:
    # remove packaging directory
    for packaging_dir in packaging_directory:
        if os.path.exists(packaging_dir):
                print(f"Deleting existing directory '{packaging_dir}'...")
                shutil.rmtree(packaging_dir)

    # remove architectory directory
    if os.path.exists(package_directory ):
        for arch_dir in architecture_directory:
            arch_path = os.path.join(package_directory, arch_dir)
            if os.path.exists(arch_path):
                print(f"Deleting existing directory '{arch_path}'...")
                shutil.rmtree(arch_path)

        # make architecture_directory and copy .so files to the directories
        for arch_dir in architecture_directory:
            arch_path = os.path.join(package_directory, arch_dir)
            os.makedirs(arch_path)
            print(f"Created directory '{arch_path}'")

            so_dir = os.path.join( '../../../Product', arch_dir + '-linux.release/out/lib')
            so_files = [f for f in os.listdir(so_dir) if f.endswith(".so")]

            for so_file in so_files:
                src_path = os.path.join(so_dir, so_file)
                shutil.copy(src_path, arch_path)
                print(f"Copied {so_file} to {arch_path}")


    print("Operation completed successfully.")

except Exception as e:
    print(f"An error occurred: {str(e)}")


# copy .so files to architecture directories

setup(
    name=package_directory,
    version='7.0.0',
    description='nnfw_api binding',
    long_description='you can use nnfw_api by python',
    packages=[package_directory],
    package_data={package_directory: ['x86_64/libnnfw_api_pybind.so', 'x86_64/libnnfw-dev.so', 'x86_64/libonert_core.so', 'x86_64/libbackend_cpu.so','armv7l/libnnfw_api_pybind.so', 'armv7l/libnnfw-dev.so', 'armv7l/libonert_core.so', 'armv7l/libbackend_cpu.so','aarch64/libnnfw_api_pybind.so', 'aarch64/libnnfw-dev.so', 'aarch64/libonert_core.so', 'aarch64/libbackend_cpu.so']},
)
