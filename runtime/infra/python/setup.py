from setuptools import setup, find_packages
import os
import shutil
import sys

architecture_directory = ['x86_64', 'armv7l', 'aarch64']
package_name = 'onert'
package_directory = 'onert'
packaging_directory = ['build', package_directory + '.egg-info']
THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRODUCT_DIR = "../../../Product"
so_list = []
so_files = []
target_arch = 'none'

try:
    # check argument "--plat-name" includes one of architecture_directory to find target architecture
    for arg in sys.argv:
        if arg.startswith('--plat-name'):
            arg_split = arg.split('=')
            arg_arch = ''
            if len(arg_split) == 2:
                arg_arch = arg_split[1]
            else:
                arch_idx = sys.argv.index('--plat-name')
                arg_arch = sys.argv[arch_idx + 1]

            for arch in architecture_directory:
                if arch in arg_arch:
                    target_arch = arch
                    print(f"Target architecture: {target_arch}")
            if target_arch == 'none':
                print(f"Unsupported target platform: {target_arch}")
                sys.exit(1)

    if target_arch == 'none':
        print(f"Need to set target platform by '--plat-name'")
        sys.exit(1)

    # remove packaging directory
    for packaging_dir in packaging_directory:
        if os.path.exists(packaging_dir):
            print(f"Deleting existing directory '{packaging_dir}'...")
            shutil.rmtree(packaging_dir)

    # initialize package_directory
    if os.path.exists(package_directory):
        print(f"Deleting existing directory '{package_directory}'...")
        shutil.rmtree(package_directory)
    os.makedirs(package_directory)
    print(f"Created directory '{package_directory}'...")

    # copy *py files to package_directory
    PY_DIR = os.path.join(THIS_FILE_DIR, '../../../runtime/onert/api/python/package')
    for root, dirs, files in os.walk(PY_DIR):
        # Calculate the relative path from the source directory
        rel_path = os.path.relpath(root, PY_DIR)
        dest_dir = os.path.join(THIS_FILE_DIR, package_directory)
        dest_sub_dir = os.path.join(dest_dir, rel_path)
        print(f"dest_sub_dir '{dest_sub_dir}'")

        # Ensure the corresponding destination subdirectory exists
        os.makedirs(dest_sub_dir, exist_ok=True)

        # Copy only .py files
        for py_file in files:
            if py_file.endswith(".py"):
                src_path = os.path.join(root, py_file)
                # dest_path = os.path.join(THIS_FILE_DIR, package_directory)
                shutil.copy(src_path, dest_sub_dir)
                print(f"Copied '{src_path}' to '{dest_sub_dir}'")

    # remove architecture directory
    if os.path.exists(package_directory):
        arch_path = os.path.join(package_directory, target_arch)
        if os.path.exists(arch_path):
            print(f"Deleting existing directory '{arch_path}'...")
            shutil.rmtree(arch_path)

        # make architecture_directory and copy .so files to the directories
        arch_path = os.path.join(package_directory, 'native')
        os.makedirs(arch_path)
        print(f"Created directory '{arch_path}'")

        def get_directories():
            # If the environment variable is not set, get default one.
            product_dir = os.environ.get("PRODUCT_DIR", DEFAULT_PRODUCT_DIR)
            return os.path.join(THIS_FILE_DIR, product_dir), os.path.join(
                product_dir,
                "lib/" if product_dir != DEFAULT_PRODUCT_DIR else target_arch +
                '-linux.release/out/lib')

        product_dir, so_base_dir = get_directories()

        for so in os.listdir(so_base_dir):
            if so.endswith(".so"):
                so_list.append('native/' + so)
                src_path = os.path.join(so_base_dir, so)
                tgt_path = os.path.join(arch_path, so)
                shutil.copy(src_path, tgt_path)
                print(f"Copied {src_path} to {tgt_path}")

        # onert core library
        so_core_dir = os.path.join(so_base_dir, 'nnfw')
        if os.path.exists(so_core_dir):
            so_core_tgt_dir = os.path.join(arch_path, 'nnfw')
            os.makedirs(so_core_tgt_dir)
            for so in os.listdir(so_core_dir):
                if so.endswith(".so"):
                    so_list.append('native/nnfw/' + so)
                    src_path = os.path.join(so_core_dir, so)
                    tgt_path = os.path.join(so_core_tgt_dir, so)
                    shutil.copy(src_path, tgt_path)
                    print(f"Copied {src_path} to {tgt_path}")

        # onert backend library
        so_backend_dir = os.path.join(so_base_dir, 'nnfw/backend')
        if os.path.exists(so_backend_dir):
            so_backend_tgt_dir = os.path.join(arch_path, 'nnfw/backend')
            os.makedirs(so_backend_tgt_dir)
            for so in os.listdir(so_backend_dir):
                if so.endswith(".so"):
                    so_list.append('native/nnfw/backend/' + so)
                    src_path = os.path.join(so_backend_dir, so)
                    tgt_path = os.path.join(so_backend_tgt_dir, so)
                    shutil.copy(src_path, tgt_path)
                    print(f"Copied {src_path} to {tgt_path}")

        # onert odc library
        so_odc_dir = os.path.join(so_base_dir, 'nnfw/odc')
        if os.path.exists(so_odc_dir):
            so_odc_tgt_dir = os.path.join(arch_path, 'nnfw/odc')
            os.makedirs(so_odc_tgt_dir)
            for so in os.listdir(so_odc_dir):
                if so.endswith(".so"):
                    so_list.append('native/nnfw/odc/' + so)
                    src_path = os.path.join(so_odc_dir, so)
                    tgt_path = os.path.join(so_odc_tgt_dir, so)
                    shutil.copy(src_path, tgt_path)
                    print(f"Copied {src_path} to {tgt_path}")

    print("Operation completed successfully.")

except Exception as e:
    print(f"An error occurred: {str(e)}")

# copy .so files to architecture directories

setup(name=package_name,
      version='0.1.0',
      description='onert API binding',
      long_description='It provides onert Python api',
      url='https://github.com/Samsung/ONE',
      license='Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, Mozilla Public License 2.0',
      has_ext_modules=lambda: True,
      packages=find_packages(),
      package_data={package_directory: so_list},
      install_requires=['numpy >= 1.19'])
