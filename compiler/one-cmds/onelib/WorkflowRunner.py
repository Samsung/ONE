#!/usr/bin/env python

# Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

import json
import os

from onelib.OptionBuilder import OptionBuilder
from onelib.TopologicalSortHelper import TopologicalSortHelper
from onelib.CfgRunner import CfgRunner
import onelib.utils as oneutils


class WorkflowRunner:
    WORKFLOWS_K = 'workflows'
    DEPENDENCIES_K = 'run-after'
    CFG_REFERENCE_K = 'cfg-reference'
    WORKFLOW_STEPS_K = 'steps'
    ONE_CMD_TOOL_K = 'one-cmd'
    COMMANDS_K = 'commands'

    def __init__(self, path):
        try:
            with open(path) as f:
                self.json_contents = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Not found given workflow file")
        except json.decoder.JSONDecodeError:
            raise ImportError("Invalid workflow file")

        self._verify_workflow(self.json_contents)

        workflows = self.json_contents[self.WORKFLOWS_K]
        self.adj = dict.fromkeys(workflows, [])
        # decide the order according to the dependencies of each workflow.
        helper = TopologicalSortHelper(workflows)
        for workflow_k in workflows:
            workflow = self.json_contents[workflow_k]
            if self.DEPENDENCIES_K in workflow:
                for previous_workflow in workflow[self.DEPENDENCIES_K]:
                    helper.add_edge(previous_workflow, workflow_k)
                    self.adj[previous_workflow].append(workflow_k)
        self.workflow_sequence = helper.sort()

        self._check_cycle()

    def _check_cycle(self):
        pos = dict()
        index = 0
        workflow_num = len(self.workflow_sequence)
        # number the order
        for seq_idx in range(workflow_num):
            pos[self.workflow_sequence[seq_idx]] = index
            index += 1

        for seq_idx in range(workflow_num):
            first_wf = self.workflow_sequence[seq_idx]
            for adj_wf in self.adj[first_wf]:
                first_pos = 0 if first_wf not in pos else pos[first_wf]
                second_pos = 0 if adj_wf not in pos else pos[adj_wf]
                if (first_pos > second_pos):
                    raise RuntimeError("Workflows should not have a cycle")

    def _verify_workflow(self, json_contents):
        # workflow file should have WORKFLOWS_K
        if not self.WORKFLOWS_K in json_contents:
            raise ValueError("Not found \"" + self.WORKFLOWS_K +
                             "\" key in workflow file")

        workflows = json_contents[self.WORKFLOWS_K]
        # workflow file should have keys listed in WORKFLOWS_K
        for workflow_k in workflows:
            if not workflow_k in json_contents:
                raise ValueError("Not found " + workflow_k + " key listed in \"" +
                                 self.WORKFLOWS_K + "\"")

        # each workflow should have either WORKFLOW_STEPS_K or CFG_REFERENCE_K
        for workflow_k in workflows:
            if not self.WORKFLOW_STEPS_K in json_contents[
                    workflow_k] and not self.CFG_REFERENCE_K in json_contents[workflow_k]:
                raise ValueError("Each workflow should have either \"" +
                                 self.WORKFLOW_STEPS_K + "\" or \"" +
                                 self.CFG_REFERENCE_K + "\"")
        for workflow_k in workflows:
            if self.WORKFLOW_STEPS_K in json_contents[
                    workflow_k] and self.CFG_REFERENCE_K in json_contents[workflow_k]:
                raise ValueError("\"" + self.WORKFLOW_STEPS_K + "\" and \"" +
                                 self.CFG_REFERENCE_K + "\" are exclusive key")

        # each step should have ONE_CMD_TOOL_K and COMMANDS_K
        for workflow_k in workflows:
            workflow = json_contents[workflow_k]
            if self.WORKFLOW_STEPS_K in workflow:
                step_keys = workflow[self.WORKFLOW_STEPS_K]
                for step_k in step_keys:
                    step = workflow[step_k]
                    if not self.ONE_CMD_TOOL_K in step or not self.COMMANDS_K in step:
                        raise ValueError("Each step should have \"" +
                                         self.ONE_CMD_TOOL_K + "\"" + " and \"" +
                                         self.COMMANDS_K + "\"")

    def run(self, working_dir, verbose=False):
        # run workflows in sequence
        for workflow_k in self.workflow_sequence:
            workflow = self.json_contents[workflow_k]
            if self.WORKFLOW_STEPS_K in workflow:
                steps = workflow[self.WORKFLOW_STEPS_K]
                for step_k in steps:
                    step = workflow[step_k]
                    commands = step[self.COMMANDS_K]
                    driver_name = step[self.ONE_CMD_TOOL_K]
                    option_builder = OptionBuilder(driver_name)
                    options = option_builder.build(commands)
                    # get the absolute path of the caller
                    driver_path = os.path.join(working_dir, driver_name)
                    cmd = [driver_path] + options
                    oneutils.run(cmd)
            elif self.CFG_REFERENCE_K in workflow:
                cfg_path = workflow[self.CFG_REFERENCE_K]['path']
                runner = CfgRunner(cfg_path)
                runner.run(working_dir, verbose)
