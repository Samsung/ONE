# The hatch build system plugin used to create the onert wheel
# This file is not a part of the Python API

import os
import shutil

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class SDistBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        super().initialize(version, build_data)

        THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DEFAULT_PRODUCT_DIR = os.path.normpath(
            os.path.join(THIS_FILE_DIR, "../../../../Product"))
        self.product_dir = os.environ.get("PRODUCT_DIR", self.DEFAULT_PRODUCT_DIR)
        self.platform = "x86_64"

        # read the environment variables that can be used to override some build settings
        self.read_env()

        # a temporary build dir used by the wheel build system
        tmp_build_dir = "_build_"
        self.recreate_dir(tmp_build_dir)

        # this is the path where the native libraries are expected to land in the wheel
        # the files copied to this location will eventually be added to the wheel by the build system
        self.whl_binaries_target_dir = os.path.join(tmp_build_dir, 'native')

        # gather the required binaries in the temporary build directory
        self.prepare_binaries()

        # include all contents of the build directory in the 'onert' subdirectory in the wheel
        # at this point the build directory should be populated with the required files
        build_data["force_include"][tmp_build_dir] = "onert"

    def read_env(self):
        self.platform = os.environ.get("PLATFORM", self.platform)
        #TODO add the platform value validation

    def prepare_binaries(self):
        # the main directory in the runtime's build tree containing the .so files
        # those files need to be copied to whl_binaries_target_dir before they are added to the wheel
        src_libs_base_dir = self.get_libs_dir()

        self.copy_libraries(src_libs_base_dir, self.whl_binaries_target_dir)
        self.copy_libraries(src_libs_base_dir, self.whl_binaries_target_dir, "nnfw")
        self.copy_libraries(src_libs_base_dir, self.whl_binaries_target_dir,
                            "nnfw/backend")
        self.copy_libraries(src_libs_base_dir, self.whl_binaries_target_dir, "nnfw/odc")

    def get_libs_dir(self):
        runtime_build_dir = self.get_runtime_build_dir()
        print(f" |> runtime_build_dir={runtime_build_dir}")

        possible_lib_dirs = ["lib64", "lib32", "lib"]
        for lib_dir in possible_lib_dirs:
            libs_dir_path = os.path.join(runtime_build_dir, lib_dir)
            if os.path.exists(libs_dir_path):
                return libs_dir_path

        raise FileNotFoundError(f"No lib directory found in {runtime_build_dir}")

    def get_runtime_build_dir(self):
        # top-level directory containing the build of the runtime
        # this can be overridden by setting the PRODUCT_DIR environment variable
        if self.product_dir != self.DEFAULT_PRODUCT_DIR:
            return self.product_dir
        else:
            # TODO - add the debug build support (via env variables)
            return os.path.join(self.product_dir, f"{self.platform}-linux.release/out")

    def copy_libraries(self, src_dir, target_dir, subdir=None):
        if subdir != None:
            src_dir = os.path.join(src_dir, subdir)
            target_dir = os.path.join(target_dir, subdir)

        os.makedirs(target_dir, exist_ok=True)

        for file in filter(lambda file: file.endswith(".so"), os.listdir(src_dir)):
            src_path = os.path.join(src_dir, file)
            tgt_path = os.path.join(target_dir, file)
            shutil.copy(src_path, tgt_path)
            print(f" |> Copied {src_path} to {tgt_path}")

    def recreate_dir(self, dir_path):
        if os.path.exists(dir_path):
            print(f" |> Deleting existing directory '{dir_path}'...")
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
