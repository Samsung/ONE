#!/usr/bin/python

# Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

from ir.subgraph import Subgraph


class TFLiteSubgraph(Subgraph):
    def __init__(self, subg_idx, tf_subgraph):
        super(TFLiteSubgraph, self).__init__()
        self.tf_subgraph = tf_subgraph

        self.index = subg_idx
        if tf_subgraph.Name() is not None:
            self.subg_name = str(tf_subgraph.Name())
        self.model_name = "#{0} {1}".format(subg_idx, self.subg_name)
        if (subg_idx == 0):  # 0th subgraph is main subgraph
            self.model_name += " (MAIN)"
