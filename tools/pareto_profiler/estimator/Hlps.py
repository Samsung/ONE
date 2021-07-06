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

import numpy as np
import sys
import Queue
import utils
import signal
from pareto import ParetoData


class Hlps:
    """ 
    Initialize Runner and Pareto data structure
  """

    def __init__(self, runner, num_backends, num_samples):
        self._runner = runner
        self._num_backends = num_backends
        self._num_samples = num_samples
        self._marked = {}
        self._extended_search = False
        self._iteration = 0
        self._pareto_obj = ParetoData()

    """
    Method to generate new samples from a given sample v_vec. 
    The new samples bear a hamming distance hd from the provided sample.    
  """

    def gen_hamming(self, v_vec, hd=1, nsamples=None):
        if nsamples is None:
            nsamples = self._num_backends - 1
        ret = np.zeros((nsamples, len(v_vec)), dtype=int)
        v = v_vec
        marked = np.full(len(v), False, dtype=bool)
        cnt = 0

        for r in range(nsamples):
            ret[r] = v
        rnd_pos = np.random.permutation(range(len(v)))
        for i in range(hd):
            pos = rnd_pos[i]
            marked[pos] = True
            for r in range(nsamples):
                ret[r][pos] = (v[pos] - r - 1) % self._num_backends

        return ret

    """
      Method to generate all samples from a given sample v_vec, that
      have a hamming distance of one with respect to it.
  """

    def gen_hamming_one(self, v_vec, invert=False):
        ret = np.zeros(((self._num_backends - 1) * len(v_vec), len(v_vec)), dtype=int)
        if invert == False:
            v = v_vec
        else:
            v = [1 - x for x in v_vec]
        for nb in range(1, self._num_backends):
            c = 0
            for r in range((nb - 1) * len(v), nb * len(v)):
                ret[r] = v
                ret[r][c] = (v[c] - nb) % self._num_backends
                c += 1
        return ret

    """
      Enable profiling over extended search space
  """

    def enable_extended_search(self):
        self._extended_search = True
        for key in self._pareto_obj.get_pareto_keys():
            config = self._pareto_obj.get_config(key)
            extended_val = self._runner.get_extended_solution(config)
            self._pareto_obj.set_config(key, extended_val)
        self._iteration = 0

    """
      HLPS algorithm implementation provided here.
      Description: Starting with a random sample, fill up a sampling 
      queue with hamming neighbors. Fetch samples from queue,
      each time checking for pareto optimality. Pareto-optimal samples
      are then explored/exploited to generate new samples that are added to the queue.
      Algorithm phase terminates when the queue is empty.
      Repeat this phase in a multi-shot invokation for better results.
  """

    def hlps_routine(self, config_ids):
        # Initialize
        solution_q = Queue.Queue()
        visited = {}
        nbits = self._runner.get_nbits(self._extended_search)
        is_extended = self._runner.get_mode_extended()
        nsolutions = self._num_backends**nbits

        stop_insert = False

        cnt = 0
        q_add_cnt = 0
        round_cnt = 0

        def extended_solution(s):
            return self._runner.get_extended_solution(s)

        def mark_solution(s):
            if is_extended == True and self._extended_search == False:
                self._marked[extended_solution(s)] = True
            else:
                self._marked[s] = True

        def is_marked(s):
            if is_extended == True and self._extended_search == False:
                return (extended_solution(s) in self._marked)
            else:
                return (s in self._marked)

        def visit_solution(s):
            if is_extended == True and self._extended_search == False:
                visited[extended_solution(s)] = True
            else:
                visited[s] = True

        def is_visited(s):
            if is_extended == True and self._extended_search == False:
                return (extended_solution(s) in visited)
            else:
                return (s in visited)

        def sigint_handler(signum, frame):
            print("Round cnt = ", round_cnt)

        signal.signal(signal.SIGINT, sigint_handler)
        if len(config_ids) > 0:
            for solution in config_ids:
                if is_extended == True and self._extended_search == True and self._iteration == 0:
                    s = extended_solution(solution)
                else:
                    s = solution
                s_vec = utils.int_to_vec(s, self._num_backends, nbits)

                candidate = self.gen_hamming_one(s_vec)
                for hd in range((self._num_backends - 1) * nbits):
                    candidate_int = int(''.join(str(x) for x in reversed(candidate[hd])),
                                        self._num_backends)
                    if is_marked(candidate_int) == False:
                        solution_q.put(candidate_int)
                        mark_solution(candidate_int)
                        q_add_cnt += 1
        else:
            start_seed = int(np.random.rand() * (nsolutions))
            solution_q.put(start_seed)
            q_add_cnt += 1

        self._iteration += 1
        # Main routine
        while not solution_q.empty():
            s = solution_q.get()
            mark_solution(s)
            stop_insert = False
            if (round_cnt % 100 == 0):
                print("sample count = ", round_cnt)
            if self._extended_search == True:
                print("Queue size is ", solution_q.qsize())

            if is_extended == True and self._extended_search == False:
                time_val, memory_val = self._runner.profile_by_opname(s)
            elif is_extended == True:
                time_val, memory_val = self._runner.profile_by_opindex(s)
            else:
                time_val, memory_val = self._runner.profile_by_opname(s)
            round_cnt += 1

            utils.progressbar(round_cnt, nsolutions, prefix="% samples computed. : ")
            self._pareto_obj.update_pareto_solutions(
                s, time_val, memory_val, explore_flag=True)

            for key in self._pareto_obj.get_pareto_keys():
                pareto_sample = self._pareto_obj.get_config(key)
                explore_sample = self._pareto_obj.get_exploration(key)

                if is_visited(pareto_sample):
                    continue
                visit_solution(pareto_sample)
                s_vec = utils.int_to_vec(pareto_sample, self._num_backends, nbits)

                if explore_sample == True:
                    # Explore solutions over a larger range
                    for hd in range(1, nbits + 1):
                        if stop_insert is True:
                            break

                        candidate = self.gen_hamming(s_vec, hd=hd)
                        for i in range(self._num_backends - 1):
                            if stop_insert is True:
                                break
                            candidate_int = int(''.join(
                                str(x) for x in reversed(candidate[i])),
                                                self._num_backends)
                            try:
                                if is_marked(candidate_int) == False:
                                    solution_q.put(candidate_int)
                                    q_add_cnt += 1
                            except IndexError:
                                print("candidate[i] = ", candidate[i],
                                      ', candidate_int = ', candidate_int)
                                sys.exit(-1)
                            if (q_add_cnt >= self._num_samples):
                                print("Queue full in explore")
                                stop_insert = True
                else:
                    # Exploit solutions within immediate neighborhood
                    candidate = self.gen_hamming_one(s_vec)

                    for j in range((self._num_backends - 1) * nbits):
                        if stop_insert is True:
                            break
                        candidate_int = int(''.join(
                            str(x) for x in reversed(candidate[j])), self._num_backends)
                        if is_marked(candidate_int) == False:
                            solution_q.put(candidate_int)
                            q_add_cnt += 1
                        if (q_add_cnt >= self._num_samples):
                            print("Queue full in exploit")
                            stop_insert = True
                    self._pareto_obj.set_exploration(key)

        pfront = set([
            self._pareto_obj.get_config(key)
            for key in self._pareto_obj.get_pareto_keys()
        ])
        return pfront, q_add_cnt

    """
      Method to dump results from HLPS
  """

    def dump_results(self, dumpdata):
        dumpdata = self._pareto_obj.dump_pareto_solutions(dumpdata)
        dumpdata = self._runner.dump_config(dumpdata)
        return dumpdata
