#!/usr/bin/env python

# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class ModelParser(object):
    def __init__(self, model_file):
        self.model_file = model_file

    # to be overriden
    def Parse(self):
        # Should return (subg_list, stats)
        raise NotImplementedError
