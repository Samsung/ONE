from hatchling.builders.hooks.plugin.interface import BuildHookInterface

from packaging.tags import sys_tags
from hook_utils import read_env

class WheelBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        super().initialize(version, build_data)

        self.platform = read_env("PLATFORM", None)
        self.glibc_version = read_env("GLIBC_VERSION", None)
        build_tag = self.create_build_tag()

        print(f" |> Created build_tag: {build_tag}")

        build_data["pure_python"] = False
        build_data['infer_tag'] = False
        build_data['tag'] = build_tag

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
        return f"{tag.interpreter}-{tag.abi}-{tag_platform}"