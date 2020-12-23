import os, json


class NnpkgHelper:
    def __init__(self):
        self.config_name = 'config.cfg'

    def add_config(self, src, configs):
        manifest_path = os.path.join(os.path.abspath(src), 'metadata', 'MANIFEST')
        config_path = os.path.join(os.path.abspath(src), 'metadata', self.config_name)

        try:
            # Read MANIFEST file
            with open(manifest_path, 'r') as manifest_file:
                data = json.load(manifest_file)

            # Add configs to MANIFEST file
            with open(manifest_path, 'w') as manifest_file:
                data['configs'] = [self.config_name]
                json.dump(data, manifest_file, indent=2)

            # Write config.cfg file
            with open(config_path, 'w') as config_file:
                config_file.write('\n'.join(configs))

        except IOError as e:
            print(e)
        except:
            print("Error")


if __name__ == "__main__":
    nnpkg_helper = NnpkgHelper()
    nnpkg_helper.add_config('../nnpkg_tst/jointq_sched',
                            ['BACKENDS=cpu', 'XNNPACK_THREADS=1'])
