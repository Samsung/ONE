import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

import shutil


def find_flatc():
    # 1. Check FLATC_PATH environment variable
    flatc_env = os.environ.get('FLATC_PATH')
    if flatc_env and os.path.isfile(flatc_env) and os.access(flatc_env, os.X_OK):
        return flatc_env

    # 2. Check system PATH
    flatc_path = shutil.which('flatc')
    if flatc_path:
        return flatc_path

    # 3. Check common build locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, '../../build/release/overlay/bin/flatc'),
        os.path.join(script_dir, '../../build/debug/overlay/bin/flatc'),
    ]

    for path in possible_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    return None


def generate_circle_py():
    flatc_path = find_flatc()
    if not flatc_path:
        print(
            "Error: flatc not found. Please set FLATC_PATH environment variable or ensure flatc is in your PATH or build directory."
        )
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.abspath(
        os.path.join(script_dir, '../../runtime/libs/circle-schema/circle_schema.fbs'))

    if not os.path.exists(schema_path):
        print(f"Error: Schema file not found at {schema_path}")
        sys.exit(1)

    output_dir = script_dir

    print(f"Generating circle.py using {flatc_path} from {schema_path}...")
    cmd = [
        flatc_path, '--python', '--gen-object-api', '--gen-onefile', '-o', output_dir,
        schema_path
    ]

    try:
        subprocess.run(cmd, check=True)
        generated_file = os.path.join(output_dir, 'circle_schema_generated.py')
        target_file = os.path.join(output_dir, 'circle.py')
        os.rename(generated_file, target_file)
        print("Successfully generated circle.py")
    except:
        print(f"Failed to generate circle.py")
        sys.exit(1)


def compile_ggml_lib():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ggml_src_dir = os.path.abspath(
        os.path.join(script_dir, '../../runtime/3rdparty/ggml/src'))
    lib_dir = os.path.join(script_dir, 'lib')
    lib_name = 'libggml_quant.so'
    lib_path = os.path.join(lib_dir, lib_name)

    if not os.path.exists(lib_dir):
        os.makedirs(lib_dir)

    print(f"Compiling {lib_name} from {ggml_src_dir}...")

    cmd = [
        'gcc', '-shared', '-fPIC', '-O3', '-o', lib_path,
        os.path.join(ggml_src_dir, 'ggml-quants.c'),
        os.path.join(ggml_src_dir, 'ggml-aarch64.c'),
        os.path.join(ggml_src_dir, 'ggml.c'), '-I', ggml_src_dir, '-I',
        os.path.abspath(os.path.join(ggml_src_dir, '../include')), '-lm'
    ]

    print("Running command:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
        print(f"Successfully compiled {lib_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to compile {lib_name}: {e}")
        sys.exit(1)


class CustomBuildPy(build_py):
    def run(self):
        generate_circle_py()
        compile_ggml_lib()
        super().run()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Direct execution: python setup.py
        # Compile in-place
        generate_circle_py()
        compile_ggml_lib()
    else:
        # Standard setuptools execution
        setup(
            name='oquantize',
            version='0.1.0',
            packages=['oquantize'],
            package_dir={'oquantize': '.'},
            include_package_data=True,
            cmdclass={
                'build_py': CustomBuildPy,
            },
            install_requires=[
                'numpy',
                'flatbuffers',
            ],
        )
