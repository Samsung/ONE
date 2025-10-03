from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class WheelBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        super().initialize(version, build_data)

        # automatically create the build tag used to name the wheel
        build_data['infer_tag'] = True
