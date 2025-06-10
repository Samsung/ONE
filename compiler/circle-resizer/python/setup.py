from setuptools import setup, find_packages
import os
import pathlib
import shutil

install_dir = os.environ.get("CMAKE_INSTALL_PREFIX")
if install_dir is None:
    print(
        'You have to set CMAKE_INSTALL_PREFIX env variable with value passed to nncc cmake as install dir'
    )
    exit(-1)

lib_dir = pathlib.Path(f'{install_dir}/lib')
resizer_dependent_so_libs = [
    'libluci_export.so',
    'libluci_import.so',
    'libluci_pass.so',
    'libluci_lang.so',
    'libloco.so',
    'libcircle_resizer_core.so',
    'circle_resizer_python_api.so',
]

package_name = 'circle_resizer'
package_data_path = (pathlib.Path(__file__).parent / package_name).resolve()
package_data_path.mkdir(parents=True, exist_ok=True)

for idx, lib_path in enumerate(resizer_dependent_so_libs):
    shutil.copy((lib_dir / lib_path).resolve(), package_data_path)
    # update path to relative to package_data_path (expected by the setup function)
    resizer_dependent_so_libs[idx] = os.path.join(package_data_path, lib_path)

setup(
    name="circle-resizer",
    version='0.0.0',
    description='circle_resizer API binding',
    long_description='It provides circle-resizer Python API',
    url='https://github.com/Samsung/ONE',
    license='Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, Mozilla Public License 2.0',
    has_ext_modules=lambda: True,
    include_package_data=True,
    packages=find_packages(),
    package_data={package_name: resizer_dependent_so_libs},
    zip_safe=False,
)
