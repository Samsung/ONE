import shutil, re
import os
import subprocess
LIB_LOCAL_DIR_NAME = './lib/'
LIB_DIR_NAME = '..'
lib_list = [
    'stdex',
    'oops',
    'pepper-str',
    'logo',
    'logo-core',
    'loco',
    'locomotiv',
    'angkor'
    # 'luci/lang',
    # 'luci/import',
    # 'luci/env', 
    ]

def copy_libs_to_local_dir():
    print("Importing libs")
    for lib_name in lib_list:
        print("----", lib_name)
        shutil.copytree(os.path.join(LIB_DIR_NAME, lib_name), 
            os.path.join(LIB_LOCAL_DIR_NAME, lib_name), dirs_exist_ok=True)

def remove_tests_from_local_dir():
    for root, dirs, files in os.walk(LIB_LOCAL_DIR_NAME):
        path = root.split(os.sep)
        for item in files:
            if item.endswith("test.cpp"):
                os.remove(os.path.join(root, item))
copy_libs_to_local_dir()
remove_tests_from_local_dir()
print("Mbed deploy")

subprocess.run(["mbed", "deploy"])
