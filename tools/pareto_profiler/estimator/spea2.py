#! /usr/bin/python
import argparse
import json
import numpy as np
import time
from profile_args import ProfileArgs
from runner import Runner
from utils import int_to_vec
from utils import progressbar


#Function to carry out the crossover
def crossover(a, b, nbits=12, mut_prob=0.5):
    iv = lambda x: gen_utils.print_bitvector(x, maxbits=nbits)
    vi = lambda xvec: int(''.join(str(x) for x in reversed(xvec)), 2)
    avec = int_to_vec(a, 2, nbits)
    bvec = int_to_vec(b, 2, nbits)
    pos = int(nbits * np.random.rand())
    new_vec1 = np.zeros(nbits, dtype=int)
    new_vec2 = np.zeros(nbits, dtype=int)
    new_vec1[:pos] = avec[:pos]
    new_vec2[:pos] = bvec[:pos]
    new_vec1[pos:] = bvec[pos:]
    new_vec2[pos:] = avec[pos:]

    for i in range(nbits):
        r = np.random.rand()
        if r <= mut_prob:
            new_vec1[i] = 1 - new_vec1[i]
            new_vec2[i] = 1 - new_vec2[i]
    new1 = vi(new_vec1.tolist())
    new2 = vi(new_vec2.tolist())

    return (new1, new2)


class Spea2:
    def __init__(self,
                 runner,
                 population=256,
                 archive_size=256,
                 generations=50,
                 mutation=0.5):
        self._runner = runner
        self._pop_size = population
        self._archive_size = archive_size
        self._max_gen = generations
        self._mut_prob = mutation
        self._marked = {}
        self._archive = np.ones(archive_size, dtype=int) * -1
        self._n_archive = 0
        N = self._runner.get_solution_spacelen()
        self._nbits = int(np.log(N) / np.log(2))
        self._population = np.array(
            [int(N * np.random.rand()) for i in range(self._pop_size)])
        self._runtime_lookup = {}
        """
        Profile samples for run time and peak memory consumption. Also,
        maintain a lookup for quicker access to performance data.
    """

    def profile_sample(self, sample):
        if sample in self._runtime_lookup:
            return self._runtime_lookup[sample]

        if self._runner.get_mode_extended() == False:
            exec_time, max_rss = self._runner.profile_by_opname(sample)
        else:
            exec_time, max_rss = self._runner.profile_by_opindex(sample)
        self._runtime_lookup[sample] = (exec_time, max_rss)
        return (exec_time, max_rss)
        """
        Generate fitness value for each sample in the population.
        The details for the fitness computation can be found under
        Section 3.1 in the SPEA2 article.
    """

    def compute_fitness(self):
        uset = set(self._population.tolist())
        if (self._n_archive > 0):
            uset = uset.union(set(self._archive.tolist()))
        # compute R and S
        S = np.zeros(len(uset), dtype=int)
        R = np.zeros(len(uset), dtype=int)
        indx = 0
        for i in uset:
            self._marked[i] = 1
            for j in uset:
                self._marked[j] = 1
                time_i, rss_i = self.profile_sample(i)
                time_j, rss_j = self.profile_sample(j)

                if (i != j) and (time_i <= time_j) and \
                    (rss_i <= rss_j):
                    S[indx] += 1
            indx += 1

        indx_i = 0
        for i in uset:
            self._marked[i] = 1
            time_i, rss_i = self.profile_sample(i)
            indx_j = 0
            for j in uset:
                self._marked[j] = 1
                time_j, rss_j = self.profile_sample(j)
                if (i != j) and (time_j <= time_i) and \
                    (rss_j <= rss_i):
                    R[indx_i] += S[indx_j]
                indx_j += 1
            indx_i += 1

        # compute D and F
        F = np.zeros(len(uset), dtype=float)
        D = np.zeros(len(uset), dtype=float)
        k = int(np.sqrt(len(self._population) + len(self._archive)))
        indx_i = 0
        for i in uset:
            time_i, rss_i = self.profile_sample(i)
            distance = []
            for j in uset:
                if (i != j):
                    time_j, rss_j = self.profile_sample(j)
                    distance.append(np.sqrt((time_i - time_j)**2 + (rss_i - rss_j)**2))
            distance.sort()
            D[indx_i] = 1.0 / (distance[k - 1] + 1)
            F[indx_i] = D[indx_i] + R[indx_i]
            indx_i += 1

        return (uset, F)
        """
        Perform selection procedure to populate a new archive of locally pareto-optimal
        samples. The selection procedure below follows the description in Section 3.2 in the
        SPEA2 article.
    """

    def selection(self, uset, F):
        # Initialize lookup
        F_lookup = {}
        idx = 0
        for e in uset:
            F_lookup[e] = F[idx]
            idx += 1

        ret = []
        indx_i = 0
        for i in uset:
            if F[indx_i] < 1:
                ret.append(i)
            indx_i += 1
        # check if number of selections is smaller than archive size
        if len(ret) < len(self._archive):
            # Case 1
            c_uset = uset.difference(set(ret))
            lookup = {}
            for e in c_uset:
                lookup[e] = F_lookup[e]
            sorted_lookup = sorted(lookup.items(), key=lambda item: item[1])
            idx = len(ret)
            for id, F_val in sorted_lookup:
                ret.append(id)
                idx += 1
                if (idx >= len(self._archive)):
                    break
            s_ret = set(ret)
        elif len(ret) > len(self._archive):
            # Case 2
            s_ret = set(ret)
            distance = np.ones((len(s_ret), len(s_ret)), dtype=float) * np.inf
            indx_i = 0
            for i in s_ret:
                time_i, rss_i = self.profile_sample(i)
                indx_j = 0
                for j in s_ret:
                    if (i != j):
                        time_j, rss_j = self.profile_sample(j)
                        distance[indx_i][indx_j] = np.sqrt((time_i - time_j)**2 +
                                                           (rss_i - rss_j)**2)
                    indx_j += 1
                indx_i += 1
            np.sort(distance, axis=1)

            nremoved = 0
            n_sret = len(s_ret)
            nelements_to_remove = len(s_ret) - len(self._archive)
            remove_set = set()
            while nremoved < nelements_to_remove:
                indx_i = 0
                for i in s_ret:
                    if nremoved >= nelements_to_remove:
                        break
                    if i in remove_set:
                        continue
                    indx_j = 0
                    removable = True
                    for j in s_ret:
                        if (i != j):
                            cnt = 0
                            for k in range(n_sret - 1):
                                if distance[indx_i][k] == distance[indx_j][k]:
                                    cnt += 1
                            if cnt != len(s_ret) - 1:
                                tmp_diff = list(distance[indx_j][:-1] -
                                                distance[indx_i][:-1])
                                sort_diff = np.copy(tmp_diff)
                                sort_diff.sort()
                                if (tmp_diff != sort_diff):
                                    removable = False
                                    break
                    if removable is True:
                        remove_set.add(i)
                        nremoved += 1
                s_ret = s_ret.difference(remove_set)
        else:
            # Case 3
            s_ret = set(ret)
        del self._archive
        return np.array(list(s_ret))
        """
        Implementation of a Binary tournament procedure for Genetic Algorithms.
        Two samples are drawn at random from the archive. They are then added to a
        mating pool, depending on their pareto-dominance relation.
    """

    def binary_tournament(self):
        n = len(self._archive)
        mating_pool = []
        while (len(mating_pool) < self._n_archive):
            rnd_a = self._archive[int(np.random.rand() * n)]
            rnd_b = self._archive[int(np.random.rand() * n)]
            while (rnd_b == rnd_a):
                rnd_b = self._archive[int(np.random.rand() * n)]
            self._marked[rnd_a] = 1
            self._marked[rnd_b] = 1
            time_a, rss_a = self.profile_sample(rnd_a)
            time_b, rss_b = self.profile_sample(rnd_b)
            if (time_a < time_b) and (rss_a < rss_b):
                mating_pool.append(rnd_a)
            elif (time_b < time_a) and (rss_b < rss_a):
                mating_pool.append(rnd_b)
            elif np.random.rand() < 0.5:
                mating_pool.append(rnd_a)
            else:
                mating_pool.append(rnd_b)
        return mating_pool

    """
      Generate new population from the mating pool. Each time, two samples are drawn at
      random from the mating pool. They are then subjected to crossover and mutation to
      produce new samples that form a part of the new population.
  """

    def gen_population(self, mating_pool):
        new_population = []

        while len(new_population) < self._pop_size:
            a1 = int(len(mating_pool) * np.random.rand())
            b1 = int(len(mating_pool) * np.random.rand())
            ret = crossover(
                mating_pool[a1],
                mating_pool[b1],
                nbits=self._nbits,
                mut_prob=self._mut_prob)
            new_population.append(ret[0])
            new_population.append(ret[1])

        del self._population
        return np.array(new_population)
        """
        SPEA2 algorithm implementation. The code below follows the description in Section 3 of the
        SPEA2 article.
    """

    def run_estimation(self):
        for i in range(self._pop_size):
            self._marked[self._population[i]] = 1
        gen_no = 0
        while (gen_no < self._max_gen):
            uset, F = self.compute_fitness()
            # print(uset, F)
            self._archive = self.selection(uset, F)
            self._n_archive = len(self._archive)
            # print(population)
            mating_pool = self.binary_tournament()
            # print(mating_pool)
            self._population = self.gen_population(mating_pool)
            # print("generation ", gen_no)
            gen_no += 1
            progressbar(gen_no, self._max_gen, prefix="% generations computed. : ")

    def dump_results(self):
        dumpdata = {}
        pareto_results = []
        for solution in set(self._archive.tolist()):
            exec_time, max_rss = self._runtime_lookup[solution]
            pareto_results.append({
                "id": solution,
                "exec_time": exec_time,
                "max_rss": max_rss
            })
        dumpdata["solutions"] = pareto_results
        dumpdata = self._runner.dump_config(dumpdata)
        return dumpdata


if __name__ == "__main__":
    ## Import data
    t_start = time.time()
    parser = ProfileArgs("nsga2.py", description="NSGA-II algorithm implementation")
    parser.add_argument('--population', type=int, default=256, help='Population Size')
    parser.add_argument(
        '--generations', type=int, default=50, help='Number of generations')
    parser.add_argument(
        '--archivesize', type=int, default=256, help='Number of pareto items to hold')
    parser.add_argument(
        '--mutation', type=float, default=0.5, help="Mutation probability (0-1)")

    # Parse arguments
    args = parser.parse_args()
    runner = Runner(args.model, args.run_folder, args.backends, args.mode)
    ## Set SPEA2 parameters
    spea_obj = Spea2(
        runner,
        population=args.population,
        archive_size=args.archivesize,
        generations=args.generations,
        mutation=args.mutation)
    #Initialization
    spea_obj.run_estimation()
    t_end = time.time()
    dumpdata = spea_obj.dump_results()
    dumpdata['profiling time'] = (t_end - t_start)
    with open(args.dumpfile, "w") as ofile:
        json.dump(dumpdata, ofile)
    print("done.., profiling time = ", (t_end - t_start), " seconds")
