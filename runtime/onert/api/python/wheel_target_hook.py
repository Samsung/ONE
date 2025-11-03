from hatchling.builders.hooks.plugin.interface import BuildHookInterface

from packaging.tags import sys_tags
import os
import shutil

class WheelBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        super().initialize(version, build_data)

        THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DEFAULT_PRODUCT_DIR = os.path.normpath(
            os.path.join(THIS_FILE_DIR, "../../../../Product"))
        
        # default values for the variables that affect and control the build of the wheel
        self.product_dir = self.DEFAULT_PRODUCT_DIR
        self.platform = "x86_64"
        self.glibc_version = None

        # read the environment variables that can be used to override some build settings
        self.read_env()

        # a temporary build dir used by the wheel build system
        tmp_build_dir = "_build_"
        self.recreate_dir(tmp_build_dir)

        # this is the path where the native libraries are expected to land in the wheel
        # the files copied to this location will eventually be added to the wheel by the build system
        # the whole structure of subdirectories will be reflected in the final wheel
        self.whl_binaries_target_dir = os.path.join(tmp_build_dir, 'native')

        # gather the required binaries in the temporary build directory
        self.prepare_binaries()

        # include all contents of the build directory in the 'onert' subdirectory in the wheel
        # at this point the temporary build directory should be populated with the required files
        build_data["force_include"][tmp_build_dir] = "onert"

        build_data["pure_python"] = False
        build_data['infer_tag'] = False
        build_data['tag'] = self.create_build_tag()

    def read_env(self):
        '''Read the relevant environment variables or use the defaults'''

        self.product_dir = self._read_env("PRODUCT_DIR", self.product_dir)
        self.platform = self._read_env("PLATFORM", self.platform)
        self.glibc_version = self._read_env("GLIBC_VERSION", self.glibc_version)
    
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
        '''Retrieve the path of a directory where the required shared libraries are'''
        runtime_build_dir = self.get_runtime_build_dir()
        print(f" |> runtime_build_dir={runtime_build_dir}")

        possible_lib_dirs = ["lib64", "lib32", "lib"]
        for lib_dir in possible_lib_dirs:
            libs_dir_path = os.path.join(runtime_build_dir, lib_dir)
            if os.path.exists(libs_dir_path):
                return libs_dir_path

        raise FileNotFoundError(f"No lib directory found in {runtime_build_dir}")

    def get_runtime_build_dir(self):
        '''Retrieve the path of a directory where the runtime's binaries are (the build tree's root)'''
        
        if self.product_dir != self.DEFAULT_PRODUCT_DIR:
            # In case the product directory was passed as an environment variable use this path
            # as a custom root directory of the build tree
            return self.product_dir
        else:
            # TODO - add the debug build support (via env variables)
            return os.path.join(self.product_dir, f"{self.platform}-linux.release/out")

    def copy_libraries(self, src_dir, target_dir, subdir=None):
        '''
        Copy all .so files found in src_dir to the target_dir
        If subdir is provided copy from src_dir/subdir to target_dir/subdir
        '''
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
        '''Delete a directory (if it exists) and create it again but empty'''
        if os.path.exists(dir_path):
            print(f" |> Deleting existing directory '{dir_path}'...")
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    
    def create_build_tag(self):
        '''Create the most appropriate build tag that will be used to name the wheel'''

        # first get the tag using the usual way build backends do it
        tag = next(sys_tags())
        
        # now create the part of the build tag that will be overridden
        # use 'manylinux' + glibc version (if provided) + the platform string 
        tag_platform = "manylinux"
        if self.glibc_version is not None:
            tag_platform = f"{tag_platform}_{self.glibc_version}_{self.platform}"
        else:
            tag_platform = f"{tag_platform}_{self.platform}"

        # compose the final tag and override the tag.platform part with our own string
        build_tag = f"{tag.interpreter}-{tag.abi}-{tag_platform}"
        print(f" |> Created build_tag: {build_tag}")
        return build_tag
    
    def _read_env(self, env_var_name, default_value):
        validators = {
            "PLATFORM": self._validate_platform,
            "GLIBC_VERSION": self._validate_glibc_version
        }

        value = os.environ.get(env_var_name, None)
        if value is None:
            return default_value
        else:
            validate = validators.get(env_var_name, None)
            if validate is not None:
                return validate(value)
            else:
                return value

    def _validate_platform(self, value):
        return value

    def _validate_glibc_version(self, value):
        return value