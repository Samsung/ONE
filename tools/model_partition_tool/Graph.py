#! /usr/bin/python
import graph_analysis
import json
import logging
import runtime_stats
import os
import sys
import numpy as np
from queue import LifoQueue


class ModelInfo:
    def __init__(self, modelfile, vertex_weights):
        self._model_dir = os.path.dirname(modelfile)
        self._dag = graph_analysis.generate_dag(modelfile)
        self._ops = graph_analysis.get_model_ops(modelfile)
        self._tensors = graph_analysis.get_model_tensors(modelfile)
        self._vertex_weights = vertex_weights
        """Return Directed Acyclic Graph (DAG)
    """

    def get_dag(self):
        return self._dag
        """Return list of model operations
    """

    def get_ops(self):
        return self._ops
        """Return list of model tensors
    """

    def get_tensors(self):
        return self._tensors
        """Return vertex weights representing execution times 
    of model operations
    """

    def get_vertex_weights(self):
        return self._vertex_weights
        """Return size (bytes) of tensor connecting operation indexes n1 and n2
    """

    def get_tensor_size(self, n1, n2):
        tensor_id = set(self._ops[n1]['outputs']).intersection(
            set(self._ops[n2]['inputs']))
        assert (len(tensor_id) == 1)
        idx = tensor_id.pop()
        tensor = self._tensors[idx]['shape']
        return np.prod(tensor) * tensor.itemsize

    def get_model_path(self):
        return self._model_dir


class GraphPartition:
    def __init__(self, K):
        self._K = K
        self._indx = np.zeros(K, dtype=int)
        self._session_weights = np.zeros(K, dtype=int)
        self._session_ids = []
        logging.basicConfig(level=logging.DEBUG)
        self._logger = logging.getLogger("Minmax")

    def set_dbglevel(self, dbglevel):
        logging.basicConfig(level=dbglevel)
        self._logger = logging.getLogger("Minmax")
        self._logger.setLevel(dbglevel)
        """Generates a session graph out of the provided dag (Directed Acyclic Graph)
    Each dag node is associated with a session id, stored under attribute _session_ids.
    """

    def generate_session_graph(self):
        def get_session_ids(i, j):
            cnt = 0
            idx_i = -1
            idx_j = -1
            for idx in range(self._K):
                if i in self._session_ids[idx]:
                    idx_i = idx
                    cnt += 1
                if j in self._session_ids[idx]:
                    idx_j = idx
                    cnt += 1
                if cnt == 2:
                    break
            return (idx_i, idx_j)

        dag = self._modelObj.get_dag()
        n = dag.shape[0]
        self._session_graph = np.zeros((self._K, self._K), dtype=int)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if dag[i][j] == 1:
                    idx1, idx2 = get_session_ids(i, j)
                    if idx1 == -1 or idx2 == -1:
                        self._logger.debug("Something wrong with session ids")
                        self._logger.debug(self._session_ids)
                        self._logger.debug(i, j)
                        sys.exit(-1)
                    if idx1 != idx2:
                        self._session_graph[idx1][idx2] = 1

        for i in range(self._K - 1):
            for j in range(i + 1, self._K):
                if self._session_graph[i][j] == 1 and self._session_graph[j][i] == 1:
                    self._logger.error("Session graph has cycles (%d, %d)", i, j)
                    self._logger.error("Session %d: %s", i, self._session_ids[i])
                    self._logger.error("Session %d: %s", j, self._session_ids[j])
                    sys.exit(-1)
        """Generate an initial partition of the topological ordering T, with the 
    help of provided vertex weights. This method will update _session_weights, that is,
    the cumulative sum of vertex weights within a session/partition
    """

    def initial_partition(self, modelObj, T):
        self._modelObj = modelObj
        self._logger.debug("Topological order: %s", T)
        vwgt = modelObj.get_vertex_weights()
        sorted_vwgt = np.array([vwgt[i] for i in T])
        self._logger.debug("sorted weights: %s", sorted_vwgt)
        sum1 = 0
        c_sorted_vw = []
        for s in sorted_vwgt:
            c_sorted_vw.append(sum1 + s)
            sum1 += s

        pivot = np.zeros(self._K - 1)
        self._logger.debug("Cumulative sum weights: %s", c_sorted_vw)
        for i in range(1, self._K):
            pivot[i - 1] = round(i * c_sorted_vw[-1] / self._K)

        for i in range(self._K - 1):
            self._indx[i + 1] = np.argmin(abs(c_sorted_vw - pivot[i]))

        sum_weights = []
        for i in range(self._K):
            if i == self._K - 1:
                self._session_ids.append(np.array(T[self._indx[i]:]))
                self._session_weights[i] = np.sum(sorted_vwgt[self._indx[i]:])
            else:
                self._session_ids.append(np.array(T[self._indx[i]:self._indx[i + 1]]))
                self._session_weights[i] = np.sum(
                    sorted_vwgt[self._indx[i]:self._indx[i + 1]])
        self.generate_session_graph()
        """Print a summary that includes session graph, paritition info comprising node ids and their
    cumulative vertex weights
    """

    def summarize(self, T):
        self._logger.info("Session Graph:\n%s",
                          np.array2string(
                              self._session_graph,
                              formatter={
                                  'int': lambda x: '{:>3}'.format(x)
                              }))
        for i in range(self._K):
            self._logger.info("Partition %d : %s, sum weight = %s", i,
                              self._session_ids[i].tolist(), self._session_weights[i])
        """Move nodes from session1 to session2 until the maximum of the two cumulative session weights are exceeded.
    The parameters to this method include the session ids, list of vertex weights (vwgt), and the directed adjacency matrix (dag).
    At the end of the move, the session ids per session, and the session weights are updated. As node movement may also affect the session
    graph, the session graph is updated as well.
    """

    def move_nodes(self, session1, session2):
        dag = self._modelObj.get_dag()
        vwgt = self._modelObj.get_vertex_weights()

        def session_edges(s1, s2, dag, forward_direction):
            sdict = {}
            if forward_direction == True:
                for k in s1:
                    tmp_s = set(np.where(dag[k, :] == 1)[0]).difference(set(s1))
                    if len(tmp_s) > 0:
                        sdict[k] = list(tmp_s)
            else:
                for k in s2:
                    tmp_s = set(np.where(dag[k, :] == 1)[0]).intersection(set(s1))
                    if len(tmp_s) > 0:
                        for key in tmp_s:
                            sdict[key] = k
            return sdict

        move_success = False
        if self._session_graph[session1][session2] == 1:
            forward_direction = True
        elif self._session_graph[session2][session1] == 1:
            forward_direction = False
        else:
            self._logger.warning("Cannot move nodes between non-neighboring partitions")
            return move_success

        maxval = max(self._session_weights[session1], self._session_weights[session2])
        improvement = True

        marked = {}
        while improvement == True:
            s1 = self._session_ids[session1]
            s2 = self._session_ids[session2]
            sdict = session_edges(s1, s2, dag, forward_direction)

            found_node = False
            rnd_perm = np.random.permutation(list(sdict))
            cnt = 0
            while found_node == False and cnt < len(sdict):
                rnd_key = rnd_perm[cnt]
                marked[rnd_key] = True
                found_node = True
                if forward_direction == True:
                    for k in range(session2):
                        if len(
                                set(np.where(dag[rnd_key, :] == 1)[0]).intersection(
                                    set(self._session_ids[k]))) > 0:
                            found_node = False
                            cnt += 1
                            break
                else:
                    for k in range(session2 + 1, self._K):
                        if len(
                                set(np.where(dag[:, rnd_key] == 1)[0]).intersection(
                                    set(self._session_ids[k]))) > 0:
                            found_node = False
                            cnt += 1
                            break
                if found_node == True:
                    new_maxval = max(self._session_weights[session1] - vwgt[rnd_key],
                                     self._session_weights[session2] + vwgt[rnd_key])
                    if new_maxval < maxval:
                        self._logger.info("[old maxval] %s --> %s [new maxval], id: %s",
                                          maxval, new_maxval, rnd_key)
                        self._logger.debug("edges : %s", (sdict[rnd_key]))
                        if type(sdict[rnd_key]) is list:
                            rnd_val = np.random.choice(sdict[rnd_key])
                        else:
                            rnd_val = sdict[rnd_key]
                        if forward_direction == True:
                            if np.where(s2 == rnd_val)[0].size > 0:
                                s2 = np.insert(s2, np.where(s2 == rnd_val)[0], rnd_key)
                            else:
                                s2 = np.insert(s2, 0, rnd_key)
                        else:
                            if np.where(s2 == sdict[rnd_key])[0].size > 0:
                                s2 = np.insert(s2,
                                               np.where(s2 == sdict[rnd_key])[0] + 1,
                                               rnd_key)
                            else:
                                s2 = np.insert(s2, len(s2), rnd_key)
                        s1 = np.delete(s1, np.where(s1 == rnd_key))
                        del self._session_ids[session1]
                        self._session_ids.insert(session1, s1)
                        del self._session_ids[session2]
                        self._session_ids.insert(session2, s2)
                        self._session_weights[session1] -= vwgt[rnd_key]
                        self._session_weights[session2] += vwgt[rnd_key]
                        maxval = new_maxval
                        self.generate_session_graph()
                        move_success = True
                    else:
                        self._logger.warning("Move rejected, max value is greater")
                        improvement = False
                else:
                    self._logger.warning(
                        "Candidate %d cannot be moved, as it violates acyclic constraint",
                        rnd_key)
                    improvement = False
        return move_success
        """Method to get the session with the maximum session weight, or cumulative exection time. This
    session is then searched for its neighboring sessions. The neighbors are then ranked in increasing order
    of their execution times, so that session moves can be performed in that order.
    """

    def get_bottleneck_info(self):
        maxval = 0
        ret_id = -1
        for i in range(self._K):
            if maxval < self._session_weights[i]:
                maxval = self._session_weights[i]
                ret_id = i
        neighbor_dict = {}

        for i in range(self._K):
            if self._session_graph[ret_id][i] == 1 or self._session_graph[i][ret_id] == 1:
                neighbor_dict[i] = self._session_weights[i]
        sorted_neighbor_list = sorted(neighbor_dict.items(), key=lambda item: item[1])
        self._logger.info("Bottleneck id --> %d, sorted neighbors --> %s", ret_id,
                          sorted_neighbor_list)
        return ret_id, sorted_neighbor_list
        """Get the cost and the partition id associated with the maximum value.
    """

    def get_maxPartitionCost(self):
        dag = self._modelObj.get_dag()
        maxval = 0
        indx = -1
        for i in range(self._K):
            if self._session_weights[i] > maxval:
                maxval = self._session_weights[i]
                indx = i

        def check_edges(dag, session1, session2):
            e_cnt = 0
            memory_overhead = 0
            for s1 in self._session_ids[session1]:
                for s2 in self._session_ids[session2]:
                    if dag[s1][s2] == 1:
                        e_cnt += 1
                        memory_overhead += self._modelObj.get_tensor_size(s1, s2)
                    elif dag[s2][s1] == 1:
                        self._logger.error("%d (session %d) connects to %d (session %d)",
                                           s2, session2, s1, session1)
                        self._logger.error(self._session_graph)
                        sys.exit(-1)

            assert (e_cnt > 0)
            return e_cnt, memory_overhead

        edge_cut = 0
        total_memory_overhead = 0
        for i in range(self._K - 1):
            for j in range(i + 1, self._K):
                if self._session_graph[i][j] == 1:
                    e_cnt, memory_overhead = check_edges(dag, i, j)
                    edge_cut += e_cnt
                    total_memory_overhead += memory_overhead
        return indx, maxval, edge_cut, total_memory_overhead
        """Get partition information.
    """

    def get_partitions(self):
        return self._session_ids, self._session_weights, self._session_graph


class GraphTopology:
    def __init__(self, tflite_file, trace_file):
        vertex_weights = runtime_stats.get_runtime_per_operation(trace_file)
        self._modelObj = ModelInfo(tflite_file, vertex_weights)
        self._dag = graph_analysis.generate_dag(tflite_file)
        self._T = []
        self._vwgt = np.array(vertex_weights)
        logging.basicConfig(level=logging.INFO)
        self._Graphlogger = logging.getLogger("Topology")

    def set_dbglevel(self, dbglevel):
        logging.basicConfig(level=dbglevel)
        self._Graphlogger.setLevel(dbglevel)
        """Perform Topological sort using the method outlined in https://arxiv.org/abs/1704.00705
    """

    def topological_sort(self):
        del self._T
        degree_matrix = np.copy(self._dag)
        n = self._dag.shape[0]
        S = []
        T = LifoQueue(maxsize=n)
        marked = {}

        while T.qsize() < n:
            indegree = np.sum(degree_matrix, axis=0)
            candidates, = np.where(indegree == 0)
            for i in candidates:
                if i not in marked:
                    S.append(i)
            np.random.seed()
            random_pos = int(np.random.rand() * len(S))
            random_node = S[random_pos]
            marked[random_node] = True
            T.put(random_node)
            neighbors, = np.where(self._dag[random_node, :] == 1)
            for i in neighbors:
                degree_matrix[random_node][i] = 0
            del S
            S = []

        self._T = list(T.queue)
        """Create a partition instance and perform an initial split over the cumulative sum weights
    """

    def partition_graph(self, K):
        self._partition = GraphPartition(K)
        self._partition.initial_partition(self._modelObj, self._T)
        """Move nodes between sessions id1 and id2
    """

    def partition_move(self, id1, id2):
        return self._partition.move_nodes(id1, id2)
        """Summarize partition information
    """

    def partition_summary(self):
        self._partition.summarize(self._T)
        """Optimize for minmax partition. At each iteration, find the bottlenecked partition, and shuffle nodes out of it
    to its neighbor with the smallest weight. If the neighbor session cannot accomodate any more nodes (because the minmax criterion is violated),
    then select the next neighbor with the smallest weight. Repeat iterations until no further improvement is possible.
    """

    def partition_minmax(self, oneshot=False):
        improvement = True
        while improvement == True:
            improvement = False
            bottleneck_id, neighbor_list = self._partition.get_bottleneck_info()
            for neighbor, wgt in neighbor_list:
                self._Graphlogger.debug("====Moving from session %d to session %d",
                                        bottleneck_id, neighbor)
                ret_success = self.partition_move(bottleneck_id, neighbor)
                if ret_success == True:
                    improvement = True
                    self._Graphlogger.debug(
                        "====Successful move from session %d to session %d",
                        bottleneck_id, neighbor)
                    break
                self._Graphlogger.debug("====Failed move from session %d to session %d",
                                        bottleneck_id, neighbor)
        if oneshot == True:
            self.partition_summary()

        return self._partition.get_maxPartitionCost()
        """Perform MinMax partitioning over multiple runs, and pick the best solution.
    """

    def partition_minmax_multiple(self, K=3, nruns=100):
        minval = np.inf
        session_ids = []
        session_weights = np.zeros(K, dtype=int)
        edge_cut_best = 0
        memory_overhead_best = 0
        for run in range(nruns):
            self._Graphlogger.debug("****Starting run %d", run)
            self.topological_sort()
            self.partition_graph(K)
            indx, maxval, edge_cut, memory_overhead = self.partition_minmax()
            if maxval < minval:
                minval = maxval
                edge_cut_best = edge_cut
                memory_overhead_best = memory_overhead
                session_ids, session_weights, session_graph = self._partition.get_partitions(
                )
            self._Graphlogger.debug("****Finished run %d", run)

        self._Graphlogger.info("Done.. printing results")
        self._Graphlogger.info("Session ids: ")
        for i in range(K):
            self._Graphlogger.info("Partition %d : %s, sum weight = %s", i,
                                   session_ids[i].tolist(), session_weights[i])
        self._Graphlogger.info("Session Graph:\n%s",
                               np.array2string(
                                   session_graph,
                                   formatter={
                                       'int': lambda x: '{:>3}'.format(x)
                                   }))
        self._Graphlogger.info("Edge cut: %d", edge_cut_best)
        self._Graphlogger.info("Memory overhead (bytes): %d", memory_overhead_best)
        output_data = {}
        partition_map = np.zeros(self._dag.shape[0], dtype=int)
        with open("".join([self._modelObj.get_model_path(), "/parition_map.json"]),
                  "w") as ofile:
            for i in range(K):
                for op_idx in session_ids[i]:
                    partition_map[op_idx] = i
            output_data['partition_map'] = partition_map.tolist()
            output_data['num_partitions'] = K
            json.dump(output_data, ofile)
