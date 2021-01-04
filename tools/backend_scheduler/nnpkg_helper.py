#!/usr/bin/python3

import json, logging
from distutils.dir_util import copy_tree
from pathlib import Path


class NnpkgHelper:
    def __init__(self):
        self.config_name = 'config.cfg'

    def copy(self, src, dst):
        copy_tree(str(src), str(dst))

    def add_config(self, src, configs):
        manifest_path = Path(src).resolve() / 'metadata' / 'MANIFEST'
        config_path = Path(src).resolve() / 'metadata' / self.config_name

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

            logging.info(f"Scheduled nnpackage is saved at {src}")

        except IOError as e:
            logging.warn(e)
        except:
            logging.warn("Error")
