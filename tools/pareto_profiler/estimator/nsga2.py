#! /usr/bin/python
import argparse
import json
import numpy as np
import time
from profile_args import ProfileArgs
from profile_args import Range
from runner import Runner
from utils import int_to_vec
from utils import progressbar


#Function to find index of list
def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i
    return -1


#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while (len(sorted_list) != len(list1)):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = np.inf
    return sorted_list


#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0, len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (values1[sorted1[k + 1]] - values2[sorted1[k - 1]]
                                     ) / (max(values1) - min(values1))
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (values1[sorted2[k + 1]] - values2[sorted2[k - 1]]
                                     ) / (max(values2) - min(values2))
    return distance


#Function to carry out the crossover
def crossover(a, b, nbackends, nbits=12, mut=0.5):
    def iv(x, nbits):
        return gen_utils.print_bitvector(x, maxbits=nbits)

    vi = lambda xvec: int(''.join(str(x) for x in reversed(xvec)), 2)

    avec = int_to_vec(a, nbackends, nbits)
    bvec = int_to_vec(b, nbackends, nbits)
    pos = int(nbits * np.random.rand())
    new_vec1 = np.zeros(nbits, dtype=int)
    new_vec2 = np.zeros(nbits, dtype=int)
    new_vec1[:pos] = avec[:pos]
    new_vec2[:pos] = bvec[:pos]
    new_vec1[pos:] = bvec[pos:]
    new_vec2[pos:] = avec[pos:]

    # while (marked[new1] == 1) and (marked[new2] == 1):
    for i in range(nbits):
        r = np.random.rand()
        if r <= mut_prob:
            new_vec1[i] = 1 - new_vec1[i]
            new_vec2[i] = 1 - new_vec2[i]
    new1 = vi(new_vec1.tolist())
    new2 = vi(new_vec2.tolist())

    return (new1, new2)


#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (values1[p] < values1[q] and values2[p] < values2[q]) or (
                    values1[p] <= values1[q]
                    and values2[p] < values2[q]) or (values1[p] < values1[q]
                                                     and values2[p] <= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] < values1[p] and values2[q] < values2[p]) or (
                    values1[q] <= values1[p]
                    and values2[q] < values2[p]) or (values1[q] < values1[p]
                                                     and values2[q] <= values2[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if (n[q] == 0):
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front) - 1]
    return front


class nsga2_profiler:
    def __init__(self, runner):
        self._lookup = {}
        self._runner = runner

    def profile_solutions(self, samples, lb, ub):
        exec_time_list = []
        max_rss_list = []
        for i in range(lb, ub):
            sample = samples[i]

            if sample in self._lookup:
                exec_time, max_rss = self._lookup[sample]
            elif args.mode == "name":
                exec_time, max_rss = self._runner.profile_by_opname(sample)
                self._lookup[sample] = (exec_time, max_rss)
            elif args.mode == "index":
                exec_time, max_rss = self._runner.profile_by_opindex(sample)
                self._lookup[sample] = (exec_time, max_rss)

            exec_time_list.append(exec_time)
            max_rss_list.append(max_rss)
        return exec_time_list, max_rss_list

    def dump_results(self, pfront, dumpdata):
        pareto_results = []
        for solution in pfront:
            exec_time, max_rss = self._lookup[solution]
            pareto_results.append({
                "id": solution,
                "exec_time": exec_time,
                "max_rss": max_rss
            })
        dumpdata.update({"solutions": pareto_results})
        dumpdata = self._runner.dump_config(dumpdata)
        return dumpdata


if __name__ == "__main__":
    t_start = time.time()
    parser = ProfileArgs("nsga2.py", description="NSGA-II algorithm implementation")
    parser.add_argument('--population', type=int, default=256, help='Population Size')
    parser.add_argument(
        '--generations', type=int, default=50, help='Number of generations')
    parser.add_argument(
        '--mutation', type=float, default=0.5, help="Mutation probability (0-1)")

    # Parse arguments
    args = parser.parse_args()
    runner = Runner(args.model, args.run_folder, args.backends, args.mode)
    nsga2_obj = nsga2_profiler(runner)
    ## Set NSGA-II parameters
    pop_size = args.population
    max_gen = args.generations
    N = runner.get_solution_spacelen()
    nbits = int(np.log(N) / np.log(2))
    mut_prob = args.mutation
    marked = {}
    nbackends = args.backends
    exec_time_list = []
    max_rss_list = []
    lookup = {}

    #Initialization
    solution = [int(N * np.random.rand()) for i in range(pop_size)]
    for i in range(pop_size):
        marked[solution[i]] = 1
    gen_no = 0
    # Run for `max_gen` number of generations
    while (gen_no < max_gen):
        time_vec, memory_vec = nsga2_obj.profile_solutions(solution, 0, pop_size)
        exec_time_list += time_vec
        max_rss_list += memory_vec

        non_dominated_sorted_solution = fast_non_dominated_sort(
            exec_time_list, max_rss_list)

        solution2 = solution[:]
        #Generating offsprings
        while (len(solution2) != 2 * pop_size):
            a1 = int(pop_size * np.random.rand())
            b1 = int(pop_size * np.random.rand())
            ret = crossover(
                solution[a1], solution[b1], args.backends, nbits=nbits, mut=mut_prob)
            marked[ret[0]] = 1
            marked[ret[1]] = 1
            solution2.append(ret[0])
            solution2.append(ret[1])
        time_vec, memory_vec = nsga2_obj.profile_solutions(solution2, pop_size,
                                                           2 * pop_size)
        exec_time_list += time_vec
        max_rss_list += memory_vec

        non_dominated_sorted_solution2 = fast_non_dominated_sort(
            exec_time_list, max_rss_list)
        crowding_distance_values2 = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(
                crowding_distance(exec_time_list[:], max_rss_list[:],
                                  non_dominated_sorted_solution2[i][:]))
        new_solution = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [
                index_of(non_dominated_sorted_solution2[i][j],
                         non_dominated_sorted_solution2[i])
                for j in range(0, len(non_dominated_sorted_solution2[i]))
            ]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:],
                                     crowding_distance_values2[i][:])
            front = [
                non_dominated_sorted_solution2[i][front22[j]]
                for j in range(0, len(non_dominated_sorted_solution2[i]))
            ]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if (len(new_solution) == pop_size):
                    break
            if (len(new_solution) == pop_size):
                break
        solution = [solution2[i] for i in new_solution]
        del exec_time_list[:]
        del max_rss_list[:]

        gen_no += 1
        progressbar(gen_no, max_gen, prefix="% generations computed. : ")

    exec_time_list, max_rss_list = nsga2_obj.profile_solutions(solution, 0, pop_size)
    non_dominated_sorted_solution = fast_non_dominated_sort(exec_time_list, max_rss_list)
    t_end = time.time()
    print("\n")
    print("done.., profiling time = ", (t_end - t_start), " seconds")
    pfront = list(set([solution[x] for x in non_dominated_sorted_solution[0]]))
    dumpdata = {}
    dumpdata['mode'] = args.mode
    dumpdata['profiling time'] = (t_end - t_start)
    dumpdata = nsga2_obj.dump_results(pfront, dumpdata)
    with open(args.dumpfile, "w") as ofile:
        json.dump(dumpdata, ofile)
