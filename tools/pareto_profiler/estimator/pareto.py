#! /usr/bin/python

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


class ParetoData:
    def __init__(self):
        self._pareto_solutions = {}
        self._configs = {}
        self._cnt = 0
        self._explore = {}

    def add_pareto_entry(self,
                         sample,
                         exec_time,
                         max_rss,
                         key,
                         explore_flag,
                         check_one_hop=True):
        self._pareto_solutions[key] = [exec_time, max_rss]
        self._configs[key] = sample
        if explore_flag == True and check_one_hop == True:
            self._explore[key] = False
        elif explore_flag == True and check_one_hop == False:
            self._explore[key] = True

    def update_pareto_solutions(self, sample, exec_time, max_rss, explore_flag=False):
        new_item = True
        if self._pareto_solutions:
            for key in list(self._pareto_solutions):
                if self._pareto_solutions[key][0] < exec_time and self._pareto_solutions[key][1] < max_rss:
                    new_item = False
                    break
                elif self._pareto_solutions[key][0] > exec_time and self._pareto_solutions[key][1] > max_rss:
                    self.add_pareto_entry(sample, exec_time, max_rss, key, explore_flag,
                                          True)
                    new_item = False

        if new_item is True:
            self.add_pareto_entry(sample, exec_time, max_rss, self._cnt, explore_flag,
                                  False)
            self._cnt += 1

    def dump_pareto_solutions(self, dumpdata):
        marked = {}
        pareto_results = []
        for i in range(self._cnt):
            if self._configs[i] not in marked:
                marked[self._configs[i]] = True
                pareto_results.append({
                    "id": self._configs[i],
                    "exec_time": self._pareto_solutions[i][0],
                    "max_rss": self._pareto_solutions[i][1]
                })
        dumpdata.update({"solutions": pareto_results})

        return dumpdata

    def get_pareto_keys(self):
        return self._configs.keys()

    def get_config(self, key):
        return self._configs[key]

    def get_exploration(self, key):
        return self._explore[key]

    def set_exploration(self, key):
        self._explore[key] = True

    def set_config(self, key, extended_value):
        self._configs[key] = extended_value
