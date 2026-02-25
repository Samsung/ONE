/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "partition.h"
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#define MAX_DEPTH 1000
std::vector<onnx::GraphProto> Subgraphs;
/**
 * Prints the subgraph information of an ONNX model to specified files.
 *
 * @param Subgraphs A vector containing subgraph information.
 * @param subgraph_file_name The filename for the output of subgraph information.
 * @param otherSubgraphs A vector containing other subgraph information.
 * @param other_subgraph_file_name The filename for the output of other subgraph information.
 */
void print_subgraphs(std::vector<onnx::GraphProto> Subgraphs, char *subgraph_file_name,
                     std::vector<onnx::GraphProto> otherSubgraphs, char *other_subgraph_file_name)
{
  int node_sum = 0;
  std::ofstream outFile(subgraph_file_name);
  if (!outFile.is_open())
  {
    std::cerr << "Error opening file." << std::endl;
    exit(0);
  }
  int id = 0;
  for (const auto &vec : Subgraphs)
  {
    outFile << " subgraph" << id << ":";
    for (const auto &node : vec.node())
    {
      outFile << node.name() << " ";
    }
    id++;
    outFile << std::endl;
    node_sum += vec.node_size();
  }
  std::ofstream outFile_2(other_subgraph_file_name);
  if (!outFile_2.is_open())
  {
    std::cerr << "Error opening file." << std::endl;
    exit(0);
  }
  std::cout << "before:" << std::endl;
  for (const auto &vec : otherSubgraphs)
  {
    outFile_2 << " subgraph" << id << ":";
    for (const auto &node : vec.node())
    {
      outFile_2 << node.name() << " ";
    }
    id++;
    outFile_2 << std::endl;
    node_sum += vec.node_size();
  }
}
///////
/**
 * @brief     Constructs an adjacency list representation of the ONNX graph.
 *
 * @param     [in] g A const reference to an ONNX GraphProto object that contains the graph
 * structure.
 * @param     [in,out] visited A pointer to an integer array used to mark whether nodes have been
 * visited.
 * @pre       The 'visited' array should be pre-allocated with a size at least equal to the number
 * of nodes in the graph.
 * @post      The 'visited' array will be initialized to 0 for all nodes.
 * @exception None
 * @return    A vector of graph_adjacency_node objects representing the adjacency list of the graph.
 */
std::vector<graph_adjacency_node> get_adjancency_list(const onnx::GraphProto &g, int *visited)
{
  std::vector<graph_adjacency_node> adjacency_list;
  int node_index = 0;
  for (const auto &node : g.node())
  {
    visited[node_index] = 0;
    graph_adjacency_node ad_node;
    ad_node.index = node_index;
    ad_node.name = node.name();
    const auto &outputs = node.output();
    for (const auto &output : outputs)
    {
      int output_node_index = 0;
      for (const auto &output_node : g.node())
      {
        int find_flag = 0;
        const auto &inputs = output_node.input();
        for (const auto &input : inputs)
        {
          if (output == input)
          {
            find_flag = 1;
            break;
          }
        }
        if (find_flag == 1)
        {
          if (std::find(ad_node.output_node_index.begin(), ad_node.output_node_index.end(),
                        output_node_index) == ad_node.output_node_index.end())
          {
            ad_node.output_node_index.push_back(output_node_index);
          }
        }
        output_node_index++;
      }
    }
    node_index++;
    adjacency_list.push_back(ad_node);
  }
  return adjacency_list;
}
/**
 * @brief     Calculates the size of a specific node in the ONNX graph in kilobytes (KB).
 *
 * @param     [in] g A const reference to an ONNX GraphProto object that contains the graph
 * structure.
 * @param     [in] node_index The index of the node for which the size is to be calculated.
 * @pre       The node_index should be a valid index within the range of nodes in the graph.
 * @post      None
 * @exception None
 * @return    The size of the node in kilobytes (KB).
 */
float calculate_node_size(const onnx::GraphProto &g, int node_index) // unit : KB
{
  int64_t node_size = 0;
  for (int i = 0; i < g.node(node_index).input_size(); i++)
  {
    std::string input_name = g.node(node_index).input(i);
    for (int j = 0; j < g.initializer_size(); j++)
    {
      if (g.initializer(j).name() == input_name)
      {
        int64_t node_init_size = 4;
        for (int k = 0; k < g.initializer(j).dims().size(); k++)
        {
          node_init_size = g.initializer(j).dims(k) * node_init_size;
        }
        node_size += node_init_size;
        break;
      }
    }
  }
  return float(node_size * 1.0 / 1024.0);
}
/**
 * @brief     Depth-First Search (DFS) to build a NPU subgraph.
 *
 * @param     [in] onnx_graph Input ONNX graph structure.
 * @param     [out] onnx_subgraph Output subgraph.
 * @param     [in,out] subgraph_node_indices Vector storing indices of nodes in the subgraph.
 * @param     [in,out] visited Array recording whether nodes have been visited.
 * @param     [in] start_node Current starting node for the search.
 * @param     [in] current_node_index Index of the current node.
 * @param     [in] adjacency_list Adjacency list representing connections between nodes in the
 * graph.
 * @param     [in] supported_op_types List of supported operation types.
 * @param     [in] preferred_op_types List of preferred operation types (not used in the code).
 * @param     [in] current_depth Current depth of the search.
 * @param     [in,out] current_graph_size Current size of the subgraph.
 * @param     [in] max_graph_size Maximum allowed size of the subgraph.
 * @pre       `current_node_index` should be a valid node index.
 * @post      If the subgraph size exceeds `max_graph_size`, a warning message is printed.
 * @exception None
 */
void DFS(const onnx::GraphProto &g, onnx::GraphProto &subgraph,
         std::vector<int> &sugraph_node_index, int *visited, const onnx::NodeProto &start_node,
         int node_index, std::vector<graph_adjacency_node> &adjacency_list,
         const std::vector<std::string> &support_op, const std::vector<std::string> &prefer_op,
         int depth_in, float &graph_size, float max_graph_size)
{
  int depth_out = depth_in + 1;
  *subgraph.add_node() = start_node;
  visited[node_index] = 1;
  sugraph_node_index.push_back(node_index);
  float node_size = calculate_node_size(g, node_index);
  graph_size += node_size;
  if (graph_size > max_graph_size)
  {
    std::cout << "graph size exceed max size!" << graph_size << " " << max_graph_size << std::endl;
  }
  for (int i = 0; i < int(adjacency_list[node_index].output_node_index.size()); i++)
  {
    if (i > 1)
    {
      std::cout << adjacency_list[node_index].output_node_index[i] << "->";
    }
    //
    int next_node_index = adjacency_list[node_index].output_node_index[i];
    const auto &next_node = g.node(next_node_index);
    if (!visited[next_node_index] &&
        (std::find(support_op.begin(), support_op.end(), next_node.op_type()) !=
         support_op.end()) &&
        (depth_out < MAX_DEPTH) && (graph_size < max_graph_size)) // 尚未访问且op_type符合的邻接顶点
      DFS(g, subgraph, sugraph_node_index, visited, next_node, next_node_index, adjacency_list,
          support_op, prefer_op, depth_out, graph_size, max_graph_size);
  }
}
/**
 * @brief     Perform a depth-first search (DFS) to build a CPU subgraph from a given starting node.
 *
 * @param     [in] g The original ONNX graph from which the subgraph will be extracted.
 * @param     [out] subgraph The subgraph being constructed.
 * @param     [out] subgraph_node_indices A vector to store indices of nodes included in the
 * subgraph.
 * @param     [in,out] visited An array to keep track of visited nodes.
 * @param     [in] start_node The starting node for the DFS.
 * @param     [in] node_index The index of the starting node in the original graph.
 * @param     [in] adjacency_list The adjacency list representing the graph's structure.
 * @param     [in] depth_in The current depth of the DFS.
 * @param     [in,out] graph_size The cumulative size of the nodes in the subgraph.
 * @param     [in] max_graph_size The maximum allowed size for the subgraph.
 *
 * @pre       The graph `g` and `adjacency_list` should be properly initialized.
 * @pre       The `visited` array should be initialized to zero.
 * @pre       `graph_size` should be initialized to zero before the first call to this function.
 *
 * @post      The `subgraph` will contain the nodes visited during the DFS.
 * @post      The `subgraph_node_indices` will contain the indices of the nodes in the subgraph.
 * @post      The `visited` array will reflect the nodes that have been visited.
 * @post      The `graph_size` will reflect the cumulative size of the nodes in the subgraph.
 *
 * @exception None
 *
 * @return    None
 */
void DFS_other(const onnx::GraphProto &g, onnx::GraphProto &subgraph,
               std::vector<int> &sugraph_node_index, int *visited,
               const onnx::NodeProto &start_node, int node_index,
               std::vector<graph_adjacency_node> &adjacency_list, int depth_in, float &graph_size,
               float max_graph_size)
{
  int depth_out = depth_in + 1;
  *subgraph.add_node() = start_node;
  visited[node_index] = 1;
  sugraph_node_index.push_back(node_index);
  float node_size = calculate_node_size(g, node_index);
  graph_size += node_size;
  if (graph_size > max_graph_size)
  {
    std::cout << "graph size exceed max size!" << graph_size << " " << max_graph_size << std::endl;
  }
  for (int i = 0; i < int(adjacency_list[node_index].output_node_index.size()); i++)
  {
    int next_node_index = adjacency_list[node_index].output_node_index[i];
    const auto &next_node = g.node(next_node_index);
    if (!visited[next_node_index] && (depth_out < MAX_DEPTH) &&
        (graph_size < max_graph_size)) // do deep first search for each successor node
      DFS_other(g, subgraph, sugraph_node_index, visited, next_node, next_node_index,
                adjacency_list, depth_out, graph_size, max_graph_size);
  }
}

/**
 * @brief     Determine and partition subgraphs from the given ONNX graph based on DFS strategy.
 * Compared with determine_subgraphs_v2, this function is more stable but may produce more subgraphs
 *
 * @param     [in] g The original ONNX graph to be partitioned.
 * @param     [out] otherSubgraphs A vector to store the subgraphs that do not meet the preferred
 * operation criteria.
 * @param     [in] d The device object containing information about supported and preferred
 * operations.
 * @param     [in,out] visited An array to keep track of visited nodes.
 * @param     [in] adjacency_list The adjacency list representing the graph's structure.
 * @param     [in] strategy The partitioning strategy to be used (e.g., SPILTE_CPU_STRUCTURE_FIRST,
 * SPILTE_NPU_STRUCTURE_FIRST).
 *
 * @pre       The graph `g` and `adjacency_list` should be properly initialized.
 * @pre       The `visited` array should be initialized to zero.
 * @pre       The device object `d` should be properly initialized with support and preferred
 * operations.
 *
 * @post      The `otherSubgraphs` vector will contain subgraphs that do not meet the preferred
 * operation criteria.
 * @post      The `visited` array will reflect the nodes that have been visited.
 *
 * @exception None
 *
 * @return    None
 */
void determine_subgraphs(const onnx::GraphProto &g, std::vector<onnx::GraphProto> &otherSubgraphs,
                         Device &d, int *visited, std::vector<graph_adjacency_node> &adjacency_list,
                         PartitionStrategy strategy)
{
  int max_subgraph_size = d.max_subgraph_size;
  std::vector<std::string> support_op;
  std::vector<std::string> prefer_op;
  switch (strategy)
  {
    case SPILTE_CPU_STRUCTURE_FIRST:
    {
      support_op = d.getCPUSupportOp();
      break;
    }
    case SPILTE_NPU_STRUCTURE_FIRST:
    {
      support_op = d.getNPUSupportOp();
      prefer_op = d.getNPUPreferOp();
      break;
    }
    default:
      break;
  }
  for (int i = 0; i < g.node_size(); i++)
  {
    if (!visited[i] &&
        (std::find(support_op.begin(), support_op.end(), g.node(i).op_type()) != support_op.end()))
    {
      onnx::GraphProto subgraph;
      std::vector<int> sugraph_node_index;
      const auto &node = g.node(i);
      int depth = 0;
      float graph_size = 0;
      DFS(g, subgraph, sugraph_node_index, visited, node, i, adjacency_list, support_op, prefer_op,
          depth, graph_size, max_subgraph_size);
      std::cout << "graph_size: " << graph_size << std::endl;
      int find_prefer_op = 0;
      for (const auto &node : subgraph.node())
      {
        if (std::find(prefer_op.begin(), prefer_op.end(), node.op_type()) != prefer_op.end())
        {
          find_prefer_op = 1;
        }
      }
      if (find_prefer_op)
      {
        Subgraphs.push_back(subgraph);
      }
      else
      {
        for (const auto &index : sugraph_node_index)
        {
          visited[index] = 0;
        }
      }
    }
  }
  for (int i = 0; i < g.node_size(); i++)
  {
    if (!visited[i])
    {
      int depth = 0;
      float graph_size = 0;
      onnx::GraphProto subgraph;
      std::vector<int> sugraph_node_index;
      const auto &node = g.node(i);
      DFS_other(g, subgraph, sugraph_node_index, visited, node, i, adjacency_list, depth,
                graph_size, max_subgraph_size);
      std::cout << "graph_size:" << graph_size << std::endl;
      otherSubgraphs.push_back(subgraph);
    }
  }
}

/**
 * @brief     Determine and partition subgraphs from the given ONNX graph using the index of nodes,
 * compared with determine_subgraphs, this function may produce less subgraphs but some of them may
 * be not fully connected(connectivity of subgrpahs will not affect the inference procedure of
 * subgraphs) This function specifically handles the partitioning logic for NPU support and
 * preferred operations.
 *
 * @param     [in] g The original ONNX graph to be partitioned.
 * @param     [out] otherSubgraphs A vector to store the subgraphs that do not meet the preferred
 * operation criteria.
 * @param     [in] d The device object containing information about supported and preferred
 * operations.
 * @param     [in,out] visited An array to keep track of visited nodes.
 * @param     [in] adjacency_list The adjacency list representing the graph's structure.
 * @param     [in] strategy The partitioning strategy to be used (e.g., SPILTE_CPU_STRUCTURE_FIRST,
 * SPILTE_NPU_STRUCTURE_FIRST).
 *
 * @pre       The graph `g` and `adjacency_list` should be properly initialized.
 * @pre       The `visited` array should be initialized to zero.
 * @pre       The device object `d` should be properly initialized with support and preferred
 * operations.
 *
 * @post      The `otherSubgraphs` vector will contain subgraphs that do not meet the preferred
 * operation criteria.
 * @post      The `visited` array will reflect the nodes that have been visited.
 *
 * @exception None
 *
 * @return    None
 */
void determine_subgraphs_v2(const onnx::GraphProto &g,
                            std::vector<onnx::GraphProto> &otherSubgraphs, Device &d, int *visited,
                            std::vector<graph_adjacency_node> &adjacency_list,
                            PartitionStrategy strategy)
{
  float max_subgraph_size = d.max_subgraph_size;
  std::vector<std::string> support_op;
  std::vector<std::string> prefer_op;
  support_op = d.getNPUSupportOp();
  prefer_op = d.getNPUPreferOp();
  onnx::GraphProto temp_graph;
  int end_flag = 0;
  int node_count = 0;
  float temp_graph_size = 0;
  while (!end_flag)
  {
    float node_size = calculate_node_size(g, node_count);
    if (temp_graph.node_size() != 0)
    {
      if ((std::find(support_op.begin(), support_op.end(), g.node(node_count).op_type()) !=
           support_op.end()) &&
          temp_graph.node_size() <= max_subgraph_size)
      {
        *temp_graph.add_node() = g.node(node_count);
        temp_graph_size += node_size;
        if (temp_graph_size > max_subgraph_size)
        {
          std::cout << "graph size exceed max size!" << temp_graph_size << " " << max_subgraph_size
                    << std::endl;
        }
        visited[node_count] = 1;
      }
      else
      {
        int find_preferop = 0;
        for (int j = 0; j < temp_graph.node_size(); j++)
        {
          if (std::find(prefer_op.begin(), prefer_op.end(), temp_graph.node(j).op_type()) !=
              prefer_op.end())
          {
            find_preferop = 1;
            break;
          }
        }
        if (find_preferop == 1)
        {
          Subgraphs.push_back(temp_graph);
        }
        else
        {
          for (int k = 1; k <= temp_graph.node_size(); k++)
          {
            visited[node_count - k] = 0;
          }
        }
        temp_graph.Clear();
        temp_graph_size = 0;
        if (std::find(support_op.begin(), support_op.end(), g.node(node_count).op_type()) !=
            support_op.end())
        {
          *temp_graph.add_node() = g.node(node_count);
          temp_graph_size += node_size;
          visited[node_count] = 1;
          continue;
        }
      }
    }
    else
    {
      if (std::find(support_op.begin(), support_op.end(), g.node(node_count).op_type()) !=
          support_op.end())
      {
        *temp_graph.add_node() = g.node(node_count);
        temp_graph_size += node_size;
        if (temp_graph_size > max_subgraph_size)
        {
          std::cout << "graph size exceed max size!" << temp_graph_size << " " << max_subgraph_size
                    << std::endl;
        }
        visited[node_count] = 1;
      }
    }
    node_count++;
    if (node_count == g.node_size())
    {
      end_flag = 1;
      if (temp_graph.node_size() != 0)
      {
        Subgraphs.push_back(temp_graph);
      }
    }
  }
  onnx::GraphProto temp_graph2;
  float temp_graph_size2 = 0;
  for (int i = 0; i < g.node_size(); i++)
  {
    float node_size = calculate_node_size(g, i);
    if (visited[i] == 0 && temp_graph_size2 < max_subgraph_size)
    {
      *temp_graph2.add_node() = g.node(i);
      temp_graph_size2 += node_size;
    }
    else
    {
      std::cout << "i = " << i << " temp_graph_size2: " << temp_graph_size2 << std::endl;
      if (temp_graph2.node_size() != 0)
      {
        otherSubgraphs.push_back(temp_graph2);
        temp_graph_size2 = 0;
        temp_graph2.Clear();
      }
      if (visited[i] == 0)
      {
        *temp_graph2.add_node() = g.node(i);
        temp_graph_size2 += node_size;
        continue;
      }
    }
    if (i == g.node_size() - 1)
    {
      if (temp_graph2.node_size() != 0)
      {
        otherSubgraphs.push_back(temp_graph2);
        temp_graph2.Clear();
      }
    }
  }
}
/**
 * @brief     Perform Tarjan's algorithm to find all strongly connected components in a directed
 * graph. This function uses depth-first search (DFS) to identify and group nodes into strongly
 * connected components.
 *
 * @param     [in] index The current node index being visited.
 * @param     [in] depth The current depth in the DFS traversal.
 * @param     [out] strongly_connected_subgraphs A vector to store the identified strongly connected
 * components.
 * @param     [in,out] DFN An array to store the discovery time of each node.
 * @param     [in,out] LOW An array to store the lowest discovery time reachable from each node.
 * @param     [in,out] stack_subgraphs A stack to keep track of nodes in the current DFS path.
 * @param     [in] successors_Subgraphs A vector of vectors representing the adjacency list of the
 * graph.
 *
 * @pre       The `DFN` and `LOW` arrays should be initialized to zero.
 * @pre       The `stack_subgraphs` should be empty before the first call to this function.
 * @pre       The `successors_Subgraphs` should be properly initialized with the graph's adjacency
 * list.
 *
 * @post      The `strongly_connected_subgraphs` vector will contain all the strongly connected
 * components found in the graph.
 * @post      The `DFN` and `LOW` arrays will reflect the discovery times and lowest reachable
 * discovery times for each node.
 * @post      The `stack_subgraphs` will be empty after the function completes.
 *
 * @exception None
 *
 * @return    None
 */
void Tarjan(int index, int depth, std::vector<std::vector<int>> &strongly_connected_subgraphs,
            int *DFN, int *LOW, std::vector<int> &stack_subgraphs,
            std::vector<std::vector<int>> &successors_Subgraphs)
{
  int rank = depth + 1;
  DFN[index] = LOW[index] = rank; // initialize DFN and LOW to 0
  stack_subgraphs.push_back(index);
  for (const auto &successor : successors_Subgraphs[index])
  {
    if (DFN[successor] == 0) // the successor is not visited
    {
      Tarjan(successor, rank, strongly_connected_subgraphs, DFN, LOW, stack_subgraphs,
             successors_Subgraphs); // visit successor
      LOW[index] = std::min(LOW[index], LOW[successor]);
    }
    else if (std::find(stack_subgraphs.begin(), stack_subgraphs.end(), successor) !=
             stack_subgraphs.end())
    {
      LOW[index] = std::min(LOW[index], DFN[successor]);
    }
  }
  if (LOW[index] == DFN[index]) // if this node is the smallest root of the strongly connected
                                // component subtree, then subsequent nodes are popped out of the
                                // stack and the obtained strongly connected components are saved.
  {
    auto it = stack_subgraphs.end() - 1;
    std::vector<int> strongly_connected;
    while (*it != index)
    {
      strongly_connected.insert(strongly_connected.begin(), *it);
      stack_subgraphs.pop_back();
      it = stack_subgraphs.end() - 1;
    }
    strongly_connected.insert(strongly_connected.begin(), *it);

    if (strongly_connected.size() > 1)
    {
      strongly_connected_subgraphs.push_back(strongly_connected);
    }
    stack_subgraphs.pop_back(); // pop
  }
}
/**
 * @brief     Calculate the rank of each node in the merged graph formed by the given strongly
 * connected components. The rank is determined based on the topological order of the nodes.
 *
 * @param     [in] strongly_connected A vector containing indices of strongly connected components.
 * @param     [in] Subgraphs A vector of ONNX GraphProtos representing the main subgraphs.
 * @param     [in] otherSubgraphs A vector of ONNX GraphProtos representing additional subgraphs.
 *
 * @pre       The `strongly_connected` vector should contain valid indices for `Subgraphs` and
 * `otherSubgraphs`.
 * @pre       The `Subgraphs` and `otherSubgraphs` vectors should be properly initialized with ONNX
 * GraphProtos.
 *
 * @post      The `node_rank_list` vector will contain the nodes of the merged graph with their
 * respective ranks.
 *
 * @exception None
 *
 * @return    A vector of `graph_adjacency_node` structures containing the nodes and their ranks.
 */
std::vector<graph_adjacency_node> calculate_node_rank(std::vector<int> &strongly_connected,
                                                      std::vector<onnx::GraphProto> &Subgraphs,
                                                      std::vector<onnx::GraphProto> &otherSubgraphs)
{
  onnx::GraphProto merged_graph;
  std::vector<graph_adjacency_node> node_rank_list;
  for (const auto &index : strongly_connected)
  {
    if (index < int(Subgraphs.size()))
    {
      mergeGraphs(merged_graph, Subgraphs[index]);
    }
    else
    {
      mergeGraphs(merged_graph, otherSubgraphs[index - Subgraphs.size()]);
    }
  }
  int index = 0;
  for (const auto &node : merged_graph.node())
  {
    graph_adjacency_node node_rank;
    node_rank.name = node.name();
    node_rank.index = index;
    node_rank.rank = -1;
    node_rank_list.push_back(node_rank);
    index++;
  }
  int sort_count = 0;
  int finished_flag = 0;
  while (!finished_flag)
  {
    finished_flag = 1;
    if (sort_count == 0)
    {
      for (int i = 0; i < merged_graph.node_size(); i++) // Traverse all nodes
      {
        int find_flag = 0;
        for (const auto &input : merged_graph.node(i).input())
        {
          for (int j = 0; j < merged_graph.node_size(); j++)
          {
            for (const auto &output : merged_graph.node(j).output())
            {
              if (input == output)
              {
                find_flag = 1;
                break;
              }
            }
            if (find_flag)
            {
              break;
            }
          }
          if (find_flag)
          {
            break;
          }
        }
        if (!find_flag)
        {
          node_rank_list[i].rank = sort_count;
        }
      }
      finished_flag = 0;
    }
    else
    {
      for (int i = 0; i < merged_graph.node_size(); i++)
      {
        int find_flag = 0;
        if (node_rank_list[i].rank >= 0 && node_rank_list[i].rank < sort_count)
        {
          continue;
        } ////If it has already been sorted, skip this subgraph
        for (const auto &input :
             merged_graph.node(i).input()) ////traveres all inputs of this subgraph
        {
          for (int j = 0; j < merged_graph.node_size();
               j++) ////examint if the input is the output of j th subgraph
          {
            for (const auto &output : merged_graph.node(j).output())
            {
              if (output == input)
              {
                if ((node_rank_list[j].rank < 0 ||
                     node_rank_list[j].rank >= sort_count)) // the j th subgraph has not been sorted
                {
                  find_flag = 1;
                  break;
                }
              }
            }
            if (find_flag)
            {
              break;
            }
          }
          if (find_flag)
          {
            break;
          }
        }
        if (!find_flag)
        {
          node_rank_list[i].rank = sort_count;
        }
        else
        {
          node_rank_list[i].rank = sort_count + 1;
          finished_flag = 0;
        }
      }
    }
    sort_count++;
  }
  return node_rank_list;
}
/**
 * @brief     Calculate the rank of each node in the merged graph formed by the given strongly
 * connected components. The rank is determined based on the topological order of the nodes.
 * Compared with calculate_node_rank, this function has different input parameters.
 *
 * @param     [in] strongly_connected A vector containing indices of strongly connected components.
 * @param     [in] Subgraphs A vector of ONNX GraphProtos representing the main subgraphs.
 * @param     [in] otherSubgraphs A vector of ONNX GraphProtos representing additional subgraphs.
 * @param     [in] subgraph_size The size of the Subgraphs vector.
 * @param     [in] other_subgraph_size The size of the otherSubgraphs vector.
 *
 * @pre       The `strongly_connected` vector should contain valid indices for `Subgraphs` and
 * `otherSubgraphs`.
 * @pre       The `Subgraphs` and `otherSubgraphs` vectors should be properly initialized with ONNX
 * GraphProtos.
 * @pre       `subgraph_size` should be equal to the size of the `Subgraphs` vector.
 * @pre       `other_subgraph_size` should be equal to the size of the `otherSubgraphs` vector.
 *
 * @post      The `node_rank_list` vector will contain the nodes of the merged graph with their
 * respective ranks.
 *
 * @exception None
 *
 * @return    A vector of `graph_adjacency_node` structures containing the nodes and their ranks.
 */
std::vector<graph_adjacency_node> calculate_node_rank_v2(
  std::vector<int> &strongly_connected, std::vector<onnx::GraphProto> &Subgraphs,
  std::vector<onnx::GraphProto> &otherSubgraphs, int subgraph_size, int other_subgraph_size)
{
  onnx::GraphProto merged_graph;
  std::vector<graph_adjacency_node> node_rank_list;
  for (const auto &index : strongly_connected)
  {
    if (index < subgraph_size)
    {
      mergeGraphs(merged_graph, Subgraphs[index]);
    }
    else
    {
      mergeGraphs(merged_graph, otherSubgraphs[index - subgraph_size]);
    }
  }
  int index = 0;
  for (const auto &node : merged_graph.node())
  {
    graph_adjacency_node node_rank;
    node_rank.name = node.name();
    node_rank.index = index;
    node_rank.rank = -1;
    node_rank_list.push_back(node_rank);
    index++;
  }
  int sort_count = 0;
  int finished_flag = 0;
  while (!finished_flag)
  {
    finished_flag = 1;
    if (sort_count == 0)
    {
      for (int i = 0; i < merged_graph.node_size(); i++) // traverse all nodes
      {
        int find_flag = 0;
        for (const auto &input : merged_graph.node(i).input())
        {
          for (int j = 0; j < merged_graph.node_size(); j++)
          {
            for (const auto &output : merged_graph.node(j).output())
            {
              if (input == output)
              {
                find_flag = 1;
                break;
              }
            }
            if (find_flag)
            {
              break;
            }
          }
          if (find_flag)
          {
            break;
          }
        }
        if (!find_flag)
        {
          node_rank_list[i].rank = sort_count;
        }
      }
      finished_flag = 0;
    }
    else
    {
      for (int i = 0; i < merged_graph.node_size(); i++)
      {
        int find_flag = 0;
        if (node_rank_list[i].rank >= 0 && node_rank_list[i].rank < sort_count)
        {
          continue;
        }
        for (const auto &input :
             merged_graph.node(i).input()) ////traverses all inputs of this subgraph
        {
          for (int j = 0; j < merged_graph.node_size();
               j++) /// examint if the input is the output of j th subgraph
          {
            for (const auto &output : merged_graph.node(j).output())
            {
              if (output == input)
              {
                if ((node_rank_list[j].rank < 0 ||
                     node_rank_list[j].rank >= sort_count)) // the j th subgraph has not been sorted
                {
                  find_flag = 1;
                  break;
                }
              }
            }
            if (find_flag)
            {
              break;
            }
          }
          if (find_flag)
          {
            break;
          }
        }
        if (!find_flag)
        {
          node_rank_list[i].rank = sort_count;
        }
        else
        {
          node_rank_list[i].rank = sort_count + 1;
          finished_flag = 0;
        }
      }
    }
    sort_count++;
  }
  return node_rank_list;
}
/**
 * @brief     Calculate the rank of each node in the given merged ONNX graph.
 *            The rank is determined based on the topological order of the nodes.
 *            This function is only used to calculate the rank of the nodes in a single graph,
 * especially the original graph
 *
 * @param     [in] merged_graph The ONNX GraphProto representing the merged graph.
 * @param     [out] node_rank_list A vector of `graph_adjacency_node` structures to store the nodes
 * and their ranks.
 *
 * @pre       The `merged_graph` should be a valid ONNX GraphProto.
 * @pre       The `node_rank_list` should be an empty vector or properly initialized.
 *
 * @post      The `node_rank_list` vector will contain the nodes of the merged graph with their
 * respective ranks.
 *
 * @exception None
 *
 * @return    None
 */
void calculate_node_rank_v3(const onnx::GraphProto &merged_graph,
                            std::vector<graph_adjacency_node> &node_rank_list)
{
  int index = 0;
  for (const auto &node : merged_graph.node())
  {
    graph_adjacency_node node_rank;
    node_rank.name = node.name();
    node_rank.index = index;
    node_rank.rank = -1;
    node_rank_list.push_back(node_rank);
    index++;
  }
  int sort_count = 0;
  int finished_flag = 0;
  while (!finished_flag)
  {
    finished_flag = 1;
    if (sort_count == 0)
    {
      for (int i = 0; i < merged_graph.node_size(); i++) // traverse all nodes
      {
        int find_flag = 0;
        for (const auto &input : merged_graph.node(i).input())
        {
          for (int j = 0; j < merged_graph.node_size(); j++)
          {
            for (const auto &output : merged_graph.node(j).output())
            {
              if (input == output)
              {
                find_flag = 1;
                break;
              }
            }
            if (find_flag)
            {
              break;
            }
          }
          if (find_flag)
          {
            break;
          }
        }
        if (!find_flag)
        {
          node_rank_list[i].rank = sort_count;
        }
      }
      finished_flag = 0;
    }
    else
    {
      for (int i = 0; i < merged_graph.node_size(); i++)
      {
        int find_flag = 0;
        if (node_rank_list[i].rank >= 0 && node_rank_list[i].rank < sort_count)
        {
          continue;
        }
        for (const auto &input :
             merged_graph.node(i).input()) ////traverses all inputs of this subgraph
        {
          for (int j = 0; j < merged_graph.node_size();
               j++) /// examint if the input is the output of j th subgraph
          {
            for (const auto &output : merged_graph.node(j).output())
            {
              if (output == input)
              {
                if ((node_rank_list[j].rank < 0 ||
                     node_rank_list[j].rank >= sort_count)) // the j th subgraph has not been sorted
                {
                  find_flag = 1;
                  break;
                }
              }
            }
            if (find_flag)
            {
              break;
            }
          }
          if (find_flag)
          {
            break;
          }
        }
        if (!find_flag)
        {
          node_rank_list[i].rank = sort_count;
        }
        else
        {
          node_rank_list[i].rank = sort_count + 1;
          finished_flag = 0;
        }
      }
    }
    sort_count++;
  }
}
/**
 * @brief     Determine the cut ranks in the given list of SCC (Strongly Connected Component) node
 * ranks. A cut rank is defined as a rank where no node exists, but there is at least one node at
 * the next rank.
 *
 * @param     [in] scc_node_rank A vector of `graph_adjacency_node` structures representing the
 * nodes and their ranks.
 *
 * @pre       The `scc_node_rank` vector should be properly initialized and contain valid node
 * ranks.
 *
 * @post      The function does not modify the `scc_node_rank` vector.
 *
 * @exception None
 *
 * @return    A vector of integers representing the cut ranks.
 */
std::vector<int> get_cut_rank_v2(std::vector<graph_adjacency_node> &scc_node_rank)
{
  std::vector<int> cut_rank_list;
  int min_cut_rank = -1;
  int max_rank = 0;
  // get min
  for (int i = 0; i < int(scc_node_rank.size()); i++)
  {
    if (scc_node_rank[i].rank < min_cut_rank || min_cut_rank < 0)
    {
      min_cut_rank = scc_node_rank[i].rank;
    }
    if (scc_node_rank[i].rank > max_rank)
    {
      max_rank = scc_node_rank[i].rank;
    }
  }
  int find_flag = 1;
  while (find_flag)
  {
    min_cut_rank++;
    int temp_find_flag = 0;
    for (int i = 0; i < int(scc_node_rank.size()); i++)
    {
      if (scc_node_rank[i].rank == min_cut_rank)
      {
        temp_find_flag = 1;
        break;
      }
    }
    find_flag = temp_find_flag;
  }
  cut_rank_list.push_back(min_cut_rank);
  int cut_rank = min_cut_rank;
  while (cut_rank < max_rank)
  {
    cut_rank = cut_rank + 1;
    int rank_flag = 0;
    int rank_plus_flag = 0;
    for (int i = 0; i < int(scc_node_rank.size()); i++)
    {
      if (scc_node_rank[i].rank == cut_rank)
      {
        rank_flag = 1;
      }
      else if (scc_node_rank[i].rank == cut_rank + 1)
      {
        rank_plus_flag = 1;
      }
    }
    if (rank_flag == 0 && rank_plus_flag == 1)
    {
      cut_rank_list.push_back(cut_rank + 1);
    }
  }

  return cut_rank_list;
}
/**
 * @brief     Eliminate strongly connected components in the graph and partition them into subgraphs
 * based on node ranks.
 *
 * @param     [in] strongly_connected_subgraphs List of indices representing strongly connected
 * components.
 * @param     [in,out] Subgraphs List of subgraphs that will be updated.
 * @param     [in,out] otherSubgraphs List of other subgraphs that will be updated.
 * @param     [in] g The original graph from which strongly connected components are derived.
 * @pre       The input graph `g` should be properly initialized and contain nodes.
 * @post      The `Subgraphs` and `otherSubgraphs` lists may be modified with new partitions based
 * on node ranks.
 * @exception None
 * @return    None
 */
void eliminate_scc_v2(std::vector<std::vector<int>> &strongly_connected_subgraphs,
                      std::vector<onnx::GraphProto> &Subgraphs,
                      std::vector<onnx::GraphProto> &otherSubgraphs, const onnx::GraphProto &g)
{
  int subgraph_size = Subgraphs.size();
  std::vector<graph_adjacency_node> node_rank_list;
  calculate_node_rank_v3(g, node_rank_list);
  for (auto &strongly_connected : strongly_connected_subgraphs)
    for (const auto scc_index : strongly_connected)
    {
      onnx::GraphProto scc_graph;
      if (scc_index < subgraph_size)
      {
        scc_graph = Subgraphs[scc_index];
      }
      else
      {
        scc_graph = otherSubgraphs[scc_index - subgraph_size];
      }
      std::vector<graph_adjacency_node> scc_node_rank;
      for (int i = 0; i < scc_graph.node_size(); i++)
      {
        for (int j = 0; j < int(node_rank_list.size()); j++)
        {
          if (scc_graph.node(i).name() == node_rank_list[j].name)
          {
            scc_node_rank.push_back(node_rank_list[j]);
            break;
          }
        }
      }
      std::vector<int> cut_rank = get_cut_rank_v2(scc_node_rank);
      onnx::GraphProto temp_graph_upper;
      int node_in_upper = 0;
      for (int i = 0; i < scc_graph.node_size(); i++)
      {
        if (scc_node_rank[i].rank < cut_rank[0])
        {
          node_in_upper++;
        }
      }
      int node_in_upper_added = 0;
      std::vector<onnx::GraphProto> temp_graph_upper_adder_list;
      int record_i = 0;
      std::cout << "node size: " << scc_graph.node_size() << std::endl;
      std::cout << "node in upper: " << node_in_upper << std::endl;
      while (node_in_upper_added < node_in_upper)
      {
        onnx::GraphProto temp_graph_upper_adder;
        for (int i = record_i; i < scc_graph.node_size(); i++)
        {
          int i_minus_1 = 0;
          if (i == 0)
          {
            i_minus_1 = 0;
          }
          else
          {
            i_minus_1 = i - 1;
          }
          if (scc_node_rank[i].rank < cut_rank[0] &&
              (i == record_i || (scc_node_rank[i].rank == scc_node_rank[i_minus_1].rank + 1)))
          {
            *temp_graph_upper_adder.add_node() = scc_graph.node(i);
            node_in_upper_added++;
          }
          else
          {
            if (scc_node_rank[i].rank >= cut_rank[0])
            {
              record_i = i + 1;
            }
            else
            {
              record_i = i;
            }
            if (temp_graph_upper_adder.node_size() > 0)
            {
              temp_graph_upper_adder_list.push_back(temp_graph_upper_adder);
              temp_graph_upper_adder.clear_node();
            }
            break;
          }
          if (i == scc_graph.node_size() - 1 && temp_graph_upper_adder.node_size() > 0)
          {
            temp_graph_upper_adder_list.push_back(temp_graph_upper_adder);
            temp_graph_upper_adder.clear_node();
          }
        }
        std::cout << "loop ended:temp graph upper adder size: "
                  << temp_graph_upper_adder.node_size() << " " << record_i << "/"
                  << scc_graph.node_size() << " node_in_upper_added:" << node_in_upper_added
                  << std::endl;
      }
      if (scc_index < subgraph_size)
      {
        Subgraphs[scc_index] = temp_graph_upper_adder_list[0];
      }
      else
      {
        otherSubgraphs[scc_index - subgraph_size] = temp_graph_upper_adder_list[0];
      }

      if (temp_graph_upper_adder_list.size() > 1)
      {
        for (int i = 1; i < int(temp_graph_upper_adder_list.size()); i++)
        {
          if (scc_index < subgraph_size)
          {
            Subgraphs.push_back(temp_graph_upper_adder_list[i]);
          }
          else
          {
            otherSubgraphs.push_back(temp_graph_upper_adder_list[i]);
          }
        }
      }
      std::cout << "scc index" << scc_index << " scc size: " << scc_graph.node_size() << std::endl;
      std::cout << "scc node rank: ";
      for (int i = 0; i < scc_graph.node_size(); i++)
      {
        std::cout << scc_node_rank[i].name << " " << scc_node_rank[i].rank << " ";
      }
      std::cout << std::endl;
      for (int i = 0; i < int(cut_rank.size()) - 1; i++)
      {
        onnx::GraphProto temp_graph_lower;
        for (int j = 0; j < scc_graph.node_size(); j++)
        {
          if (scc_node_rank[j].rank >= cut_rank[i] && scc_node_rank[j].rank < cut_rank[i + 1])
          {
            *temp_graph_lower.add_node() = scc_graph.node(j);
          }
        }
        if (scc_index < subgraph_size)
        {
          if (temp_graph_lower.node_size() > 0)
          {
            Subgraphs.push_back(temp_graph_lower);
          }
        }
        else
        {
          if (temp_graph_lower.node_size() > 0)
          {
            otherSubgraphs.push_back(temp_graph_lower);
          }
        }
      }
      onnx::GraphProto temp_graph_lower;
      for (int j = 0; j < scc_graph.node_size(); j++)
      {
        if (scc_node_rank[j].rank >= cut_rank[cut_rank.size() - 1])
        {
          *temp_graph_lower.add_node() = scc_graph.node(j);
        }
      }
      if (scc_index < subgraph_size)
      {
        if (temp_graph_lower.node_size() > 0)
        {
          Subgraphs.push_back(temp_graph_lower);
        }
      }
      else
      {
        if (temp_graph_lower.node_size() > 0)
        {
          otherSubgraphs.push_back(temp_graph_lower);
        }
      }
    }
  for (int i = Subgraphs.size() - 1; i >= 0; i--)
  {
    if (Subgraphs[i].node_size() == 0)
    {
      Subgraphs.erase(Subgraphs.begin() + i);
    }
  }
  for (int i = otherSubgraphs.size() - 1; i >= 0; i--)
  {
    if (otherSubgraphs[i].node_size() == 0)
    {
      otherSubgraphs.erase(otherSubgraphs.begin() + i);
    }
  }
}
/**
 * @brief     Eliminate strongly connected components in the graph and partition them into
 * individual subgraphs.
 *
 * @param     [in] strongly_connected_subgraphs List of indices representing strongly connected
 * components.
 * @param     [in,out] Subgraphs List of subgraphs that will be updated.
 * @param     [in,out] otherSubgraphs List of other subgraphs that will be updated.
 * @param     [in] g The original graph from which strongly connected components are derived.
 * @pre       The input graph `g` should be properly initialized and contain nodes.
 * @post      The `Subgraphs` and `otherSubgraphs` lists will be updated with individual nodes from
 * each strongly connected component.
 * @exception None
 * @return    None
 */
void eliminate_scc_v3(std::vector<std::vector<int>> &strongly_connected_subgraphs,
                      std::vector<onnx::GraphProto> &Subgraphs,
                      std::vector<onnx::GraphProto> &otherSubgraphs, const onnx::GraphProto &g)
{
  int subgraph_size = Subgraphs.size();
  for (int i = 0; i < int(strongly_connected_subgraphs.size()); i++)
  {
    for (const auto scc_index : strongly_connected_subgraphs[i])
    {
      std::cout << "scc index: " << scc_index << std::endl;
      onnx::GraphProto scc_graph;
      if (scc_index < subgraph_size)
      {
        scc_graph = Subgraphs[scc_index];
      }
      else
      {
        scc_graph = otherSubgraphs[scc_index - subgraph_size];
      }
      for (int j = 0; j < scc_graph.node_size(); j++)
      {
        onnx::GraphProto graph_temp;
        *graph_temp.add_node() = scc_graph.node(j);
        if (j == 0)
        {
          if (scc_index < subgraph_size)
          {
            Subgraphs[scc_index] = graph_temp;
          }
          else
          {
            otherSubgraphs[scc_index - subgraph_size] = graph_temp;
          }
        }
        else
        {
          if (scc_index < subgraph_size)
          {
            Subgraphs.push_back(graph_temp);
          }
          else
          {
            otherSubgraphs.push_back(graph_temp);
          }
        }
      }
    }
  }
  for (int i = Subgraphs.size() - 1; i >= 0; i--)
  {
    if (Subgraphs[i].node_size() == 0)
    {
      Subgraphs.erase(Subgraphs.begin() + i);
    }
  }
  for (int i = otherSubgraphs.size() - 1; i >= 0; i--)
  {
    if (otherSubgraphs[i].node_size() == 0)
    {
      otherSubgraphs.erase(otherSubgraphs.begin() + i);
    }
  }
}
/**
 * @brief     Determine the graph type based on the given index and return the corresponding graph.
 *
 * @param     [in] index The index of the graph to determine.
 * @param     [in] Subgraphs List of subgraphs.
 * @param     [in] otherSubgraphs List of other subgraphs.
 * @param     [in] subgraph_size The size of the Subgraphs list.
 * @pre       The `index` should be a valid index within the combined range of `Subgraphs` and
 * `otherSubgraphs`.
 * @post      None
 * @exception None
 * @return    The graph corresponding to the given index.
 */
onnx::GraphProto determinegraphtype_v2(int index, std::vector<onnx::GraphProto> &Subgraphs,
                                       std::vector<onnx::GraphProto> &otherSubgraphs,
                                       int subgraph_size)
{
  if (index < subgraph_size)
  {
    return Subgraphs[index];
  }
  else
  {
    return otherSubgraphs[index - subgraph_size];
  }
}
/**
 * @brief     Find pairs of strongly connected subgraphs based on input and output tensors.
 *
 * @param     [in] strongly_connected_subgraphs List of strongly connected subgraphs.
 * @param     [in] Subgraphs List of subgraphs.
 * @param     [in] otherSubgraphs List of other subgraphs.
 * @param     [in] graphs_inputs List of input tensors for each graph.
 * @param     [in] graphs_outputs List of output tensors for each graph.
 * @param     [out] sccs_pairs List of pairs of strongly connected subgraphs.
 * @pre       The input lists should be properly initialized and contain valid data.
 * @post      The `sccs_pairs` list will contain pairs of indices representing connected subgraphs.
 * @exception None
 * @return    None
 */
void find_subgraph_pair_v2(std::vector<std::vector<int>> &strongly_connected_subgraphs,
                           std::vector<onnx::GraphProto> &Subgraphs,
                           std::vector<onnx::GraphProto> &otherSubgraphs,
                           std::vector<std::unordered_set<NodeTensor>> &graphs_inputs,
                           std::vector<std::unordered_set<NodeTensor>> &graphs_outputs,
                           std::vector<std::vector<std::vector<int>>> &sccs_pairs)
{
  int count = 0;
  for (const auto &strongly_connected : strongly_connected_subgraphs)
  {
    std::vector<onnx::GraphProto> scc_graphs;
    std::vector<std::unordered_set<NodeTensor>> scc_graphs_inputs;
    std::vector<std::unordered_set<NodeTensor>> scc_graphs_outputs;
    for (const auto &index : strongly_connected)
    {
      std::unordered_set<NodeTensor> graph_inputs = graphs_inputs[index];
      std::unordered_set<NodeTensor> graph_outputs = graphs_outputs[index];
      scc_graphs_inputs.push_back(graph_inputs);
      scc_graphs_outputs.push_back(graph_outputs);
    }
    std::vector<std::vector<int>> scc_pairs;
    std::vector<int> is_pushed;
    for (int j = 0; j < int(strongly_connected.size()); j++)
    {
      is_pushed.push_back(0);
    }
    for (int i = 0; i < int(strongly_connected.size()); i++)
    {
      for (const auto &graph_input : scc_graphs_inputs[i])
      {
        for (int j = i + 1; j < int(strongly_connected.size()); j++)
        {
          std::vector<int> scc_pair;
          if (scc_graphs_outputs[j].find(graph_input) != scc_graphs_outputs[j].end() &&
              is_pushed[j] == 0)
          {
            for (const auto &graph_output : scc_graphs_outputs[i])
            {
              if (scc_graphs_inputs[j].find(graph_output) != scc_graphs_inputs[j].end())
              {
                scc_pair.push_back(strongly_connected[i]);
                scc_pair.push_back(strongly_connected[j]);
                scc_pairs.push_back(scc_pair);
                is_pushed[j] = 1;
                is_pushed[i] = 1;
                break;
              }
            }
          }
          if (is_pushed[i] == 1)
          {
            break;
          }
        }
        if (is_pushed[i] == 1)
        {
          break;
        }
      }
    }
    if (scc_pairs.size() != 0)
    {
      sccs_pairs.push_back(scc_pairs);
    }
    count++;
  }
  for (const auto &scc_pairs : sccs_pairs)
  {
    std::cout << "scc pair:";
    for (const auto &scc_pair : scc_pairs)
    {

      for (const auto &scc_id : scc_pair)
      {
        std::cout << scc_id << " ";
      }
      std::cout << ";";
    }
    std::cout << std::endl;
  }
}
/**
 * @brief     Cut a pair of subgraphs into upper and lower parts based on node rank.
 *
 * @param     [in] Subgraphs List of subgraphs.
 * @param     [in] otherSubgraphs List of other subgraphs.
 * @param     [in] graphs_inputs List of input tensors for each graph.
 * @param     [in] graphs_outputs List of output tensors for each graph.
 * @param     [in] scc_pair Pair of subgraph indices to be cut.
 * @param     [out] scc_pair_cut List of cut subgraphs (upper and lower parts of master graph and
 * slave graph).
 * @param     [in] subgraph_size Size of subgraph.
 * @pre       The input lists should be properly initialized and contain valid data.
 * @post      The `scc_pair_cut` list will contain the cut subgraphs.
 * @exception None
 * @return    A vector containing the index of the master graph and the cut rank.
 */
std::vector<int> cut_pair(std::vector<onnx::GraphProto> &Subgraphs,
                          std::vector<onnx::GraphProto> &otherSubgraphs,
                          std::vector<std::unordered_set<NodeTensor>> &graphs_inputs,
                          std::vector<std::unordered_set<NodeTensor>> &graphs_outputs,
                          std::vector<int> &scc_pair, std::vector<onnx::GraphProto> &scc_pair_cut,
                          int subgraph_size)
{
  std::vector<graph_adjacency_node> pair_node_list =
    calculate_node_rank(scc_pair, Subgraphs, otherSubgraphs);
  int master_graph = 0;
  for (const auto &node : pair_node_list)
  {
    if (node.rank == 0)
    {
      int find_flag = -1;
      onnx::GraphProto graph_temp =
        determinegraphtype_v2(scc_pair[0], Subgraphs, otherSubgraphs, subgraph_size);
      for (const auto &graph_node : graph_temp.node())
      {
        if (graph_node.name() == node.name)
        {
          find_flag = 1;
          break;
        }
      }
      if (find_flag == 1)
      {
        master_graph = 0;
        break;
      }
      else
      {
        master_graph = 1;
        break;
      }
    }
  }
  int slave_graph = 1 - master_graph;
  // find the position where master and slave graph connect
  int cut_rank = -1;
  for (const auto &output : graphs_outputs[scc_pair[slave_graph]])
  {
    for (const auto &input : graphs_inputs[scc_pair[master_graph]])
    {

      if (input.name == output.name)
      {
        int node_index = 0;
        onnx::GraphProto graph_temp =
          determinegraphtype_v2(scc_pair[slave_graph], Subgraphs, otherSubgraphs, subgraph_size);
        for (const auto &graph_node : graph_temp.node())
        {
          int update_node_rank = 0;
          for (const auto &output_node : graph_node.output())
          {
            if (output_node == output.name)
            {
              if (slave_graph == 0)
              {
                if (cut_rank == -1 || cut_rank > pair_node_list[node_index].rank)
                {
                  cut_rank = pair_node_list[node_index].rank;
                }
              }
              else
              {
                onnx::GraphProto graph_temp_1 = determinegraphtype_v2(
                  scc_pair[master_graph], Subgraphs, otherSubgraphs, subgraph_size);
                if (cut_rank == -1 ||
                    cut_rank > pair_node_list[node_index + graph_temp_1.node_size()].rank)
                {
                  cut_rank = pair_node_list[node_index + graph_temp_1.node_size()].rank;
                }
              }
              update_node_rank = 1;
              break;
            }
          }
          if (update_node_rank == 1)
          {
            break;
          }
          node_index++;
        }
        break;
      }
    }
  }
  // cut master graph according to the rank
  onnx::GraphProto master_upper;
  onnx::GraphProto master_lower;
  int node_index = 0;
  onnx::GraphProto graph_temp =
    determinegraphtype_v2(scc_pair[master_graph], Subgraphs, otherSubgraphs, subgraph_size);
  for (const auto &node : graph_temp.node())
  {
    int node_rank;
    if (master_graph == 0)
    {
      node_rank = pair_node_list[node_index].rank;
    }
    else
    {
      onnx::GraphProto graph_temp_2 =
        determinegraphtype_v2(scc_pair[slave_graph], Subgraphs, otherSubgraphs, subgraph_size);
      node_rank = pair_node_list[node_index + graph_temp_2.node_size()].rank;
    }
    if (node_rank < cut_rank)
    {
      *master_upper.add_node() = node;
    }
    else
    {
      *master_lower.add_node() = node;
    }
    node_index++;
  }
  scc_pair_cut.push_back(master_upper);
  scc_pair_cut.push_back(master_lower);
  scc_pair_cut.push_back(
    determinegraphtype_v2(scc_pair[slave_graph], Subgraphs, otherSubgraphs, subgraph_size));
  if (master_graph == 1)
  {
    int temp = scc_pair[0];
    scc_pair[0] = scc_pair[1];
    scc_pair[1] = temp;
    master_graph = 0;
  } // assure the first graph is master
  std::vector<int> return_value;
  return_value.push_back(master_graph);
  return_value.push_back(cut_rank);
  return return_value;
}
/**
 * @brief     Eliminate pairs of subgraphs by cutting them and updating the subgraph lists.
 *
 * @param     [in,out] Subgraphs List of subgraphs to be processed and updated.
 * @param     [in,out] otherSubgraphs List of other subgraphs to be processed and updated.
 * @param     [in] graphs_inputs List of input tensors for each graph.
 * @param     [in] graphs_outputs List of output tensors for each graph.
 * @param     [in] strongly_connected_subgraphs List of strongly connected subgraphs.
 * @param     [in] subgraph_size Size of subgraph.
 * @pre       The input lists should be properly initialized and contain valid data.
 * @post      The `Subgraphs` and `otherSubgraphs` lists will be updated with cut subgraphs.
 * @exception None
 * @return    None
 */
void eliminate_pair_v2(std::vector<onnx::GraphProto> &Subgraphs,
                       std::vector<onnx::GraphProto> &otherSubgraphs,
                       std::vector<std::unordered_set<NodeTensor>> &graphs_inputs,
                       std::vector<std::unordered_set<NodeTensor>> &graphs_outputs,
                       std::vector<std::vector<int>> &strongly_connected_subgraphs,
                       int subgraph_size)
{
  int original_node_size = 0;
  for (auto &subgraph : Subgraphs)
  {
    original_node_size += subgraph.node_size();
  }
  for (auto &subgraph : otherSubgraphs)
  {
    original_node_size += subgraph.node_size();
  }
  std::vector<std::vector<std::vector<int>>> sccs_pairs;
  find_subgraph_pair_v2(strongly_connected_subgraphs, Subgraphs, otherSubgraphs, graphs_inputs,
                        graphs_outputs, sccs_pairs);
  for (auto &scc_pairs : sccs_pairs)
  {
    for (auto &scc_pair : scc_pairs)
    {
      std::vector<onnx::GraphProto> scc_pair_cut;
      cut_pair(Subgraphs, otherSubgraphs, graphs_inputs, graphs_outputs, scc_pair, scc_pair_cut,
               subgraph_size);
      if (scc_pair[0] < subgraph_size)
      {
        Subgraphs[scc_pair[0]] = scc_pair_cut[0];
        Subgraphs.push_back(scc_pair_cut[1]);
      }
      else
      {
        otherSubgraphs[scc_pair[0] - subgraph_size] = scc_pair_cut[0];
        otherSubgraphs.push_back(scc_pair_cut[1]);
      }

      if (scc_pair[1] < subgraph_size)
      {
        Subgraphs[scc_pair[1]] = scc_pair_cut[2];
      }
      else
      {
        otherSubgraphs[scc_pair[1] - subgraph_size] = scc_pair_cut[2];
      }
    }
  }
  for (int i = Subgraphs.size() - 1; i >= 0; i--)
  {
    if (Subgraphs[i].node_size() == 0)
    {
      Subgraphs.erase(Subgraphs.begin() + i);
    }
  }
  for (int i = otherSubgraphs.size() - 1; i >= 0; i--)
  {
    if (otherSubgraphs[i].node_size() == 0)
    {
      otherSubgraphs.erase(otherSubgraphs.begin() + i);
    }
  }
}
/**
 * @brief     Find the successor or predecessor subgraph with the least number of nodes.
 *
 * @param     [in] index Index of the current subgraph.
 * @param     [in] successor List of successor indices.
 * @param     [in] predecessor List of predecessor indices.
 * @param     [in] Subgraphs List of subgraphs.
 * @param     [in] otherSubgraphs List of other subgraphs.
 * @pre       The input lists should be properly initialized and contain valid data.
 * @post      None
 * @exception None
 * @return    Index of the successor or predecessor subgraph with the least number of nodes, or -1
 * if no such subgraph exists.
 */
int find_min_size(int index, std::vector<int> &successor, std::vector<int> &predecessor,
                  std::vector<onnx::GraphProto> &Subgraphs,
                  std::vector<onnx::GraphProto>
                    &otherSubgraphs) // find the successor or predecessor with the least nodes
{
  std::vector<int> size_list;
  int min_index = -1;
  int min_size = 10000;
  for (int i = 0; i < int(successor.size()); i++)
  {
    std::cout << "successor: " << successor[i];
    onnx::GraphProto tempgraph;
    if ((successor[i] < int(Subgraphs.size()) && index < int(Subgraphs.size())) ||
        (successor[i] >= int(Subgraphs.size()) && index >= int(Subgraphs.size())))
    {
      if (successor[i] < int(Subgraphs.size()))
      {
        tempgraph = Subgraphs[successor[i]];
      }
      else
      {
        tempgraph = otherSubgraphs[successor[i] - int(Subgraphs.size())];
      }
    }
    else
    {
      continue;
    }
    int size = int(tempgraph.node_size());
    std::cout << " size:" << size << " min:" << min_size;
    if (size < min_size && size != 1)
    {
      min_size = size;
      min_index = successor[i];
      std::cout << " update min index:" << min_index;
    }
    std::cout << std::endl;
  }
  for (int i = 0; i < int(predecessor.size()); i++)
  {
    std::cout << "predecessor: " << predecessor[i];
    onnx::GraphProto tempgraph;
    if ((predecessor[i] < int(Subgraphs.size()) && index < int(Subgraphs.size())) ||
        (predecessor[i] >= int(Subgraphs.size()) && index >= int(Subgraphs.size())))
    {
      if (predecessor[i] < int(Subgraphs.size()))
      {
        tempgraph = Subgraphs[predecessor[i]];
      }
      else
      {
        tempgraph = otherSubgraphs[predecessor[i] - int(Subgraphs.size())];
      }
    }
    else
    {
      continue;
    }
    int size = int(tempgraph.node_size());
    std::cout << " size:" << size << " min:" << min_size;
    if (size < min_size && size != 1)
    {
      min_size = size;
      min_index = predecessor[i];
      std::cout << " update min index:" << min_index;
    }
    std::cout << std::endl;
  }
  return min_index;
}
void Partition::PartitionGraph(const onnx::GraphProto &g, Device &d, PartitionStrategy strategy,
                               const std::unordered_map<std::string, NodeIOSize> &node_io_size)
{
  std::unordered_set<NodeTensor> IOvalueNames = getIOvalue(g);
  int *visited = (int *)malloc(g.node_size() * sizeof(int));
  std::vector<graph_adjacency_node> adjacency_list = get_adjancency_list(g, visited);
  std::vector<onnx::GraphProto> otherSubgraphs;
  determine_subgraphs_v2(g, otherSubgraphs, d, visited, adjacency_list, strategy);
  std::cout << "Partition Done" << std::endl;
  free(visited);
  std::vector<graph_adjacency_node>().swap(adjacency_list);
  int node_sum = 0;
  // traverse the structures and print each element
  std::ofstream outFile("./subgraphs_1.txt");
  if (!outFile.is_open())
  {
    std::cerr << "Error opening file." << std::endl;
    exit(0);
  }
  int id = 0;
  for (const auto &vec : Subgraphs)
  {
    outFile << " subgraph" << id << ":";
    for (const auto &node : vec.node())
    {
      outFile << node.name() << " ";
    }
    id++;
    outFile << std::endl;
    node_sum += vec.node_size();
  }
  int id_record = id;
  std::ofstream outFile_2("./subgraphs_2.txt");
  if (!outFile_2.is_open())
  {
    std::cerr << "Error opening file." << std::endl;
    exit(0);
  }
  std::cout << "before:" << std::endl;
  for (const auto &vec : otherSubgraphs)
  {
    outFile_2 << " subgraph" << id << ":";
    for (const auto &node : vec.node())
    {
      outFile_2 << node.name() << " ";
    }
    id++;
    outFile_2 << std::endl;
    node_sum += vec.node_size();
  }
  std::vector<std::unordered_set<std::string>> subgraphs_2_input_nodes_;
  std::vector<std::unordered_set<std::string>> subgraphs_2_nodes_;
  for (const auto &sg : otherSubgraphs)
  {
    std::unordered_set<NodeTensor> graphInputs;
    determineGraphInput(sg, IOvalueNames, graphInputs);
    std::unordered_set<std::string> graphInputsNodes;
    for (const auto &input : graphInputs)
    {
      auto nodename = findInputNode(g, input.name);
      if (nodename != "")
      {
        graphInputsNodes.insert(nodename);
      }
    }
    subgraphs_2_input_nodes_.push_back(graphInputsNodes);
    subgraphs_2_nodes_.push_back(collectNodeNames(sg));
  }
  int *is_merged = (int *)malloc(otherSubgraphs.size() * sizeof(int));
  for (int i = 0; i < int(otherSubgraphs.size()); i++)
  {
    is_merged[i] = 0;
  }
  std::cout << "graph size after merging:" << otherSubgraphs.size() << std::endl;
  free(is_merged);
  std::ofstream outFile_3("./subgraphs_3.txt");
  if (!outFile_3.is_open())
  {
    std::cerr << "Error opening file." << std::endl;
    exit(0);
  }
  ////othersubgraphs after merged
  for (const auto &vec : otherSubgraphs)
  {
    outFile_3 << " subgraph" << id_record << ":";
    for (const auto &node : vec.node())
    {
      outFile_3 << node.name() << " ";
    }
    id_record++;
    outFile_3 << std::endl;
  }
  std::cout << "sub node size:" << node_sum << std::endl;

  std::vector<std::unordered_set<NodeTensor>> subgraphs_1_inputs;
  std::vector<std::unordered_set<std::string>> subgraphs_1_input_nodes;
  std::vector<std::unordered_set<std::string>> subgraphs_1_nodes;
  for (const auto &sg : Subgraphs)
  {
    std::unordered_set<NodeTensor> graphInputs;
    determineGraphInput(sg, IOvalueNames, graphInputs);
    subgraphs_1_inputs.push_back(graphInputs);
    std::unordered_set<std::string> graphInputsNodes;
    for (const auto &input : graphInputs)
    {
      auto nodename = findInputNode(g, input.name);
      if (nodename != "")
      {
        graphInputsNodes.insert(nodename);
      }
    }
    subgraphs_1_input_nodes.push_back(graphInputsNodes);
    subgraphs_1_nodes.push_back(collectNodeNames(sg));
  }

  std::vector<std::unordered_set<NodeTensor>> subgraphs_2_inputs;
  std::vector<std::unordered_set<std::string>> subgraphs_2_input_nodes;
  std::vector<std::unordered_set<std::string>> subgraphs_2_nodes;
  for (const auto &sg : otherSubgraphs)
  {
    std::unordered_set<NodeTensor> graphInputs;
    determineGraphInput(sg, IOvalueNames, graphInputs);
    subgraphs_2_inputs.push_back(graphInputs);
    std::unordered_set<std::string> graphInputsNodes;
    for (const auto &input : graphInputs)
    {
      auto nodename = findInputNode(g, input.name);
      if (nodename != "")
      {
        graphInputsNodes.insert(nodename);
      }
    }
    subgraphs_2_input_nodes.push_back(graphInputsNodes);
    subgraphs_2_nodes.push_back(collectNodeNames(sg));
  }
  std::vector<std::unordered_set<NodeTensor>> subgraphs_1_outputs;

  int node_number = 0;

  for (const auto &sg : Subgraphs)
  {
    std::unordered_set<NodeTensor> graphOutputs;
    node_number += sg.node_size();
    determineGraphOutput(g, sg, subgraphs_1_inputs, subgraphs_2_inputs, graphOutputs);
    subgraphs_1_outputs.push_back(graphOutputs);
  }
  std::vector<std::unordered_set<NodeTensor>> subgraphs_2_outputs;
  for (const auto &sg : otherSubgraphs)
  {
    std::unordered_set<NodeTensor> graphOutputs;
    node_number += sg.node_size();
    determineGraphOutput(g, sg, subgraphs_1_inputs, subgraphs_2_inputs, graphOutputs);
    subgraphs_2_outputs.push_back(graphOutputs);
  }
  int graph_node_size_minus_constant = g.node_size();
  for (const auto &node : g.node())
  {
    if (node.op_type() == "Constant")
    {
      graph_node_size_minus_constant--;
    }
  }
  std::cout << "total number of nodes in subgraphs:" << node_number << std::endl;
  std::cout << "total number of nodes in origional graph:" << graph_node_size_minus_constant
            << std::endl;
  std::vector<std::unordered_set<NodeTensor>> graphs_inputs;
  graphs_inputs.insert(graphs_inputs.end(), subgraphs_1_inputs.begin(), subgraphs_1_inputs.end());
  graphs_inputs.insert(graphs_inputs.end(), subgraphs_2_inputs.begin(), subgraphs_2_inputs.end());
  std::vector<std::unordered_set<NodeTensor>> graphs_outputs;
  graphs_outputs.insert(graphs_outputs.end(), subgraphs_1_outputs.begin(),
                        subgraphs_1_outputs.end());
  graphs_outputs.insert(graphs_outputs.end(), subgraphs_2_outputs.begin(),
                        subgraphs_2_outputs.end());

  std::vector<std::vector<int>> predecessors_Subgraphs(graphs_inputs.size());
  std::vector<std::vector<int>> successors_Subgraphs(graphs_inputs.size());
  for (int i = 0; i < int(graphs_inputs.size()); i++) // traversal all subgraphs
  {
    std::vector<int> predecessors;
    for (const auto &g_input : graphs_inputs[i])
    {
      for (int j = 0; j < int(graphs_outputs.size()); j++)
      {
        if ((graphs_outputs[j].find(g_input) != graphs_outputs[j].end()))
        {
          predecessors.push_back(j);
        }
      }
    }
    if (predecessors.size() == 0)
    {
      std::cout << "subgraph " << i << " has no predecessors" << std::endl;
    }
    predecessors_Subgraphs[i].insert(predecessors_Subgraphs[i].end(), predecessors.begin(),
                                     predecessors.end());
  }
  for (int i = 0; i < int(graphs_inputs.size()); i++)
  {
    for (int j = 0; j < int(graphs_inputs.size()); j++)
    {
      if (find(predecessors_Subgraphs[j].begin(), predecessors_Subgraphs[j].end(), i) !=
          predecessors_Subgraphs[j].end())
      {
        successors_Subgraphs[i].push_back(j);
      }
    }
  }
  std::vector<std::vector<int>> strongly_connected_subgraphs;
  int *DFN = (int *)malloc(graphs_inputs.size() * sizeof(int));
  int *LOW = (int *)malloc(graphs_inputs.size() * sizeof(int));
  for (int i = 0; i < int(graphs_inputs.size()); i++)
  {
    DFN[i] = 0;
    LOW[i] = 0;
  }
  for (int temp_count = 0; temp_count < int(predecessors_Subgraphs.size()); temp_count++)
  {
    if (DFN[temp_count] == 0)
    {
      std::vector<int> stack_subgraphs;
      int depth = 0;
      Tarjan(temp_count, depth, strongly_connected_subgraphs, DFN, LOW, stack_subgraphs,
             successors_Subgraphs);
    }
  }

  std::string file_name_scc = "scc.txt";
  std::ofstream outfile_scc(file_name_scc);
  outfile_scc << strongly_connected_subgraphs.size() << std::endl;
  for (const auto &scc : strongly_connected_subgraphs)
  {
    std::cout << "scc:";
    outfile_scc << "scc: ";
    for (const auto &scc_id : scc)
    {
      outfile_scc << scc_id << " ";
    }
    outfile_scc << std::endl;
    for (const auto &scc_id : scc)
    {
      std::cout << scc_id << " ";
      outfile_scc << "subgraph" << scc_id << " input:";
      for (const auto &scc_input : graphs_inputs[scc_id])
      {
        outfile_scc << scc_input.name << ";";
      }
      outfile_scc << " output:";
      for (const auto &scc_output : graphs_outputs[scc_id])
      {
        outfile_scc << scc_output.name << ";";
      }
      outfile_scc << std::endl;
    }

    std::cout << std::endl;
  }
  outfile_scc.close();
  free(DFN);
  free(LOW);
  int node_num_all = 0;
  for (const auto &sg : Subgraphs)
  {
    node_num_all += sg.node_size();
  }
  for (const auto &sg : otherSubgraphs)
  {
    node_num_all += sg.node_size();
  }
  std::cout << "node num in original graph: " << g.node_size() << std::endl;
  std::cout << "node_num after cut " << node_num_all << std::endl;
  ///////////////////////+++
  int *DFN_ = (int *)malloc(graphs_inputs.size() * sizeof(int));
  int *LOW_ = (int *)malloc(graphs_inputs.size() * sizeof(int));
  for (int i = 0; i < int(graphs_inputs.size()); i++)
  {
    DFN_[i] = 0;
    LOW_[i] = 0;
  }
  for (int temp_count = 0; temp_count < int(predecessors_Subgraphs.size()); temp_count++)
  {
    if (DFN_[temp_count] == 0)
    {
      std::vector<int> stack_subgraphs;
      int depth = 0;
      Tarjan(temp_count, depth, strongly_connected_subgraphs, DFN_, LOW_, stack_subgraphs,
             successors_Subgraphs);
    }
  }
  free(DFN_);
  free(LOW_);
  eliminate_scc_v2(strongly_connected_subgraphs, Subgraphs, otherSubgraphs, g);
  /////////////////////
  strongly_connected_subgraphs.clear();
  predecessors_Subgraphs.clear();
  successors_Subgraphs.clear();
  std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_2_inputs);
  std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_1_inputs);
  std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_2_outputs);
  std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_1_outputs);
  std::vector<std::unordered_set<NodeTensor>>().swap(graphs_inputs);
  std::vector<std::unordered_set<NodeTensor>>().swap(graphs_outputs);
  for (const auto &sg : Subgraphs)
  {
    std::unordered_set<NodeTensor> graphInputs;
    determineGraphInput(sg, IOvalueNames, graphInputs);
    subgraphs_1_inputs.push_back(graphInputs);
  }
  for (const auto &sg : otherSubgraphs)
  {
    std::unordered_set<NodeTensor> graphInputs;
    determineGraphInput(sg, IOvalueNames, graphInputs);
    subgraphs_2_inputs.push_back(graphInputs);
  }
  node_number = 0;
  for (const auto &sg : Subgraphs)
  {
    std::unordered_set<NodeTensor> graphOutputs;
    node_number += sg.node_size();
    determineGraphOutput(g, sg, subgraphs_1_inputs, subgraphs_2_inputs, graphOutputs);
    subgraphs_1_outputs.push_back(graphOutputs);
  }
  for (const auto &sg : otherSubgraphs)
  {
    std::unordered_set<NodeTensor> graphOutputs;
    node_number += sg.node_size();
    determineGraphOutput(g, sg, subgraphs_1_inputs, subgraphs_2_inputs, graphOutputs);
    subgraphs_2_outputs.push_back(graphOutputs);
  }
  graphs_inputs.insert(graphs_inputs.end(), subgraphs_1_inputs.begin(), subgraphs_1_inputs.end());
  graphs_inputs.insert(graphs_inputs.end(), subgraphs_2_inputs.begin(), subgraphs_2_inputs.end());
  graphs_outputs.insert(graphs_outputs.end(), subgraphs_1_outputs.begin(),
                        subgraphs_1_outputs.end());
  graphs_outputs.insert(graphs_outputs.end(), subgraphs_2_outputs.begin(),
                        subgraphs_2_outputs.end());
  for (int i = 0; i < int(graphs_inputs.size()); i++)
  {
    std::vector<int> predecessors;
    for (const auto &g_input : graphs_inputs[i])
    {
      for (int j = 0; j < int(graphs_outputs.size()); j++)
      {
        if ((graphs_outputs[j].find(g_input) != graphs_outputs[j].end()))
        {
          predecessors.push_back(j);
        }
      }
    }
    predecessors_Subgraphs.push_back(predecessors);
  }
  for (int i = 0; i < int(graphs_inputs.size()); i++)
  {
    std::vector<int> temp;
    for (int j = 0; j < int(graphs_inputs.size()); j++)
    {
      if (find(predecessors_Subgraphs[j].begin(), predecessors_Subgraphs[j].end(), i) !=
          predecessors_Subgraphs[j].end())
      {
        temp.push_back(j);
      }
    }
    successors_Subgraphs.push_back(temp);
  }
  std::string file_name_predecessor_2 = "predecessor_final_2.txt";
  std::string file_name_successor_2 = "successor_final_2.txt";
  std::ofstream outfile_predecessor_2(file_name_predecessor_2);
  std::ofstream outfile_successor_2(file_name_successor_2);
  if (!(outfile_predecessor_2.is_open() && outfile_successor_2.is_open()))
  {
    std::cerr << "Error opening file." << std::endl;
    exit(0);
  }
  for (int i = 0; i < int(graphs_inputs.size()); i++)
  {
    outfile_predecessor_2 << "predecessor of subgraph " << i << ":";
    for (const auto &predecessor : predecessors_Subgraphs[i])
    {
      outfile_predecessor_2 << predecessor << ";";
    }
    outfile_predecessor_2 << std::endl;
    outfile_successor_2 << "successor of subgraph " << i << ":";
    for (const auto &successor : successors_Subgraphs[i])
    {
      outfile_successor_2 << successor << ";";
    }
    outfile_successor_2 << std::endl;
  }
  outfile_predecessor_2.close();
  outfile_successor_2.close();
  print_subgraphs(Subgraphs, (char *)"./subgraphs_final_2.txt", otherSubgraphs,
                  (char *)"./other_subgraphs_final_2.txt");
  int *DFN_2 = (int *)malloc(graphs_inputs.size() * sizeof(int));
  int *LOW_2 = (int *)malloc(graphs_inputs.size() * sizeof(int));
  for (int i = 0; i < int(graphs_inputs.size()); i++)
  {
    DFN_2[i] = 0;
    LOW_2[i] = 0;
  }
  for (int temp_count = 0; temp_count < int(predecessors_Subgraphs.size()); temp_count++)
  {
    if (DFN_[temp_count] == 0)
    {
      std::vector<int> stack_subgraphs;
      int depth = 0;
      Tarjan(temp_count, depth, strongly_connected_subgraphs, DFN_2, LOW_2, stack_subgraphs,
             successors_Subgraphs);
    }
  }
  std::string file_name_scc2 = "scc2.txt";
  std::ofstream outfile_scc2(file_name_scc2);
  for (const auto &scc : strongly_connected_subgraphs)
  {
    std::cout << "scc:";
    outfile_scc2 << "scc: ";
    for (const auto &scc_id : scc)
    {
      outfile_scc2 << scc_id << " ";
    }
    outfile_scc2 << std::endl;
    for (const auto &scc_id : scc)
    {
      std::cout << scc_id << " ";
      outfile_scc2 << "subgraph" << scc_id << " input:";
      for (const auto &scc_input : graphs_inputs[scc_id])
      {
        outfile_scc2 << scc_input.name << ";";
      }
      outfile_scc2 << " output:";
      for (const auto &scc_output : graphs_outputs[scc_id])
      {
        outfile_scc2 << scc_output.name << ";";
      }
      outfile_scc2 << std::endl;
    }

    std::cout << std::endl;
  }
  outfile_scc.close();
  free(DFN_2);
  free(LOW_2);
  // eliminate_scc_v2(strongly_connected_subgraphs,  Subgraphs, otherSubgraphs, g);
  int subgraph_size_2 = Subgraphs.size();
  int other_subgraph_size_2 = otherSubgraphs.size();
  std::vector<int> eliminated_small_graph_id;
  std::vector<int> eliminated_small_graph_size;
  std::vector<int> eliminated_small_graph_size_2;
  std::vector<int> unmerged_graph_id;
  for (int i = 0; i < subgraph_size_2 + other_subgraph_size_2; i++)
  {
    std::cout << "i:" << i << std::endl;
    if (i < subgraph_size_2)
    {
      if (Subgraphs[i].node_size() < 2)
      {
        int merge_id = find_min_size(i, successors_Subgraphs[i], predecessors_Subgraphs[i],
                                     Subgraphs, otherSubgraphs);
        if (merge_id < subgraph_size_2 && merge_id >= 0)
        {
          mergeGraphs(Subgraphs[merge_id], Subgraphs[i]);
          eliminated_small_graph_id.push_back(i);
          eliminated_small_graph_size.push_back(Subgraphs[i].node_size());
          std::cout << "eliminating small graph " << i << "and merged to " << merge_id << std::endl;
        }
        else if (merge_id >= 0)
        {
          mergeGraphs(otherSubgraphs[merge_id - subgraph_size_2], Subgraphs[i]);
          eliminated_small_graph_id.push_back(i);
          eliminated_small_graph_size.push_back(Subgraphs[i].node_size());
          std::cout << "eliminating small graph " << i << "and merged to " << merge_id << std::endl;
        }
        else
        {
          unmerged_graph_id.push_back(i);
        }
      }
    }
    else
    {
      if (otherSubgraphs[i - subgraph_size_2].node_size() < 2)
      {
        int merge_id = find_min_size(i, successors_Subgraphs[i], predecessors_Subgraphs[i],
                                     Subgraphs, otherSubgraphs);
        if (merge_id < subgraph_size_2 && merge_id >= 0)
        {
          mergeGraphs(Subgraphs[merge_id], otherSubgraphs[i - subgraph_size_2]);
          eliminated_small_graph_id.push_back(i);
          eliminated_small_graph_size.push_back(otherSubgraphs[i - subgraph_size_2].node_size());
          std::cout << "eliminating small graph " << i << "and merged to " << merge_id << std::endl;
        }
        else if (merge_id >= 0)
        {
          mergeGraphs(otherSubgraphs[merge_id - subgraph_size_2],
                      otherSubgraphs[i - subgraph_size_2]);
          eliminated_small_graph_id.push_back(i);
          eliminated_small_graph_size.push_back(otherSubgraphs[i - subgraph_size_2].node_size());
          std::cout << "eliminating small graph " << i << "and merged to " << merge_id << std::endl;
        }
        else
        {
          unmerged_graph_id.push_back(i);
        }
      }
    }
  }
  std::cout << "succeed in reaching here" << std::endl;
  for (int i = eliminated_small_graph_id.size() - 1; i >= 0; i--)
  {
    if (std::find(unmerged_graph_id.begin(), unmerged_graph_id.end(),
                  eliminated_small_graph_id[i]) != unmerged_graph_id.end())
    {
      continue;
    }
    std::cout << eliminated_small_graph_id[i] << " ";
    int index = eliminated_small_graph_id[i];
    if (index < subgraph_size_2)
    {
      if (Subgraphs[index].node_size() > 1)
      {
        std::cout << "eliminate Subgraphs" << index << " ";
        for (auto node : Subgraphs[index].node())
        {
          std::cout << node.name() << " ";
        }
      }
      eliminated_small_graph_size_2.push_back(Subgraphs[index].node_size());
      Subgraphs.erase(Subgraphs.begin() + index);
    }
    else
    {
      if (otherSubgraphs[index - subgraph_size_2].node_size() > 1)
      {
        std::cout << "eliminate otherSubgraphs" << index - subgraph_size_2 << " ";
        for (auto node : otherSubgraphs[index - subgraph_size_2].node())
        {
          std::cout << node.name() << " ";
        }
      }
      eliminated_small_graph_size_2.push_back(otherSubgraphs[index - subgraph_size_2].node_size());
      otherSubgraphs.erase(otherSubgraphs.begin() + index - subgraph_size_2);
    }
  }
  std::cout << std::endl;
  std::cout << "eliminated_small_graph_size_1: ";
  for (const auto &size : eliminated_small_graph_size)
  {
    std::cout << size << " ";
  }
  std::cout << std::endl;
  std::cout << "eliminated_small_graph_size_2: ";
  for (const auto &size : eliminated_small_graph_size_2)
  {
    std::cout << size << " ";
  }
  std::cout << std::endl;
  /////////clear
  strongly_connected_subgraphs.clear();
  predecessors_Subgraphs.clear();
  successors_Subgraphs.clear();
  std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_2_inputs);
  std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_1_inputs);
  std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_2_outputs);
  std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_1_outputs);
  std::vector<std::unordered_set<NodeTensor>>().swap(graphs_inputs);
  std::vector<std::unordered_set<NodeTensor>>().swap(graphs_outputs);
  for (const auto &sg : Subgraphs)
  {
    std::unordered_set<NodeTensor> graphInputs;
    determineGraphInput(sg, IOvalueNames, graphInputs);
    subgraphs_1_inputs.push_back(graphInputs);
  }
  for (const auto &sg : otherSubgraphs)
  {
    std::unordered_set<NodeTensor> graphInputs;
    determineGraphInput(sg, IOvalueNames, graphInputs);
    subgraphs_2_inputs.push_back(graphInputs);
  }
  node_number = 0;
  for (const auto &sg : Subgraphs)
  {
    std::unordered_set<NodeTensor> graphOutputs;
    node_number += sg.node_size();
    determineGraphOutput(g, sg, subgraphs_1_inputs, subgraphs_2_inputs, graphOutputs);
    subgraphs_1_outputs.push_back(graphOutputs);
  }
  for (const auto &sg : otherSubgraphs)
  {
    std::unordered_set<NodeTensor> graphOutputs;
    node_number += sg.node_size();
    determineGraphOutput(g, sg, subgraphs_1_inputs, subgraphs_2_inputs, graphOutputs);
    subgraphs_2_outputs.push_back(graphOutputs);
  }
  graphs_inputs.insert(graphs_inputs.end(), subgraphs_1_inputs.begin(), subgraphs_1_inputs.end());
  graphs_inputs.insert(graphs_inputs.end(), subgraphs_2_inputs.begin(), subgraphs_2_inputs.end());
  graphs_outputs.insert(graphs_outputs.end(), subgraphs_1_outputs.begin(),
                        subgraphs_1_outputs.end());
  graphs_outputs.insert(graphs_outputs.end(), subgraphs_2_outputs.begin(),
                        subgraphs_2_outputs.end());
  for (int i = 0; i < int(graphs_inputs.size()); i++)
  {
    std::vector<int> predecessors;
    for (const auto &g_input : graphs_inputs[i])
    {
      for (int j = 0; j < int(graphs_outputs.size()); j++)
      {
        if ((graphs_outputs[j].find(g_input) != graphs_outputs[j].end()))
        {
          predecessors.push_back(j);
        }
      }
    }
    predecessors_Subgraphs.push_back(predecessors);
  }
  for (int i = 0; i < int(graphs_inputs.size()); i++)
  {
    std::vector<int> temp;
    for (int j = 0; j < int(graphs_inputs.size()); j++)
    {
      if (find(predecessors_Subgraphs[j].begin(), predecessors_Subgraphs[j].end(), i) !=
          predecessors_Subgraphs[j].end())
      {
        temp.push_back(j);
      }
    }
    successors_Subgraphs.push_back(temp);
  }
  std::string file_name_predecessor_3 = "predecessor_final_3.txt";
  std::string file_name_successor_3 = "successor_final_3.txt";
  std::ofstream outfile_predecessor_3(file_name_predecessor_3);
  std::ofstream outfile_successor_3(file_name_successor_3);
  if (!(outfile_predecessor_3.is_open() && outfile_successor_3.is_open()))
  {
    std::cerr << "Error opening file." << std::endl;
    exit(0);
  }
  for (int i = 0; i < int(graphs_inputs.size()); i++)
  {
    outfile_predecessor_3 << "predecessor of subgraph " << i << ":";
    for (const auto &predecessor : predecessors_Subgraphs[i])
    {
      outfile_predecessor_3 << predecessor << ";";
    }
    outfile_predecessor_3 << std::endl;
    outfile_successor_3 << "successor of subgraph " << i << ":";
    for (const auto &successor : successors_Subgraphs[i])
    {
      outfile_successor_3 << successor << ";";
    }
    outfile_successor_3 << std::endl;
  }
  outfile_predecessor_3.close();
  outfile_successor_3.close();
  print_subgraphs(Subgraphs, (char *)"./subgraphs_final_3.txt", otherSubgraphs,
                  (char *)"./other_subgraphs_final_3.txt");
  node_num_all = 0;
  for (const auto &sg : Subgraphs)
  {
    node_num_all += sg.node_size();
  }
  for (const auto &sg : otherSubgraphs)
  {
    node_num_all += sg.node_size();
  }
  int *DFN_3 = (int *)malloc(graphs_inputs.size() * sizeof(int));
  int *LOW_3 = (int *)malloc(graphs_inputs.size() * sizeof(int));
  for (int i = 0; i < int(graphs_inputs.size()); i++)
  {
    DFN_3[i] = 0;
    LOW_3[i] = 0;
  }
  for (int temp_count = 0; temp_count < int(predecessors_Subgraphs.size()); temp_count++)
  {
    if (DFN_[temp_count] == 0)
    {
      std::vector<int> stack_subgraphs;
      int depth = 0;
      Tarjan(temp_count, depth, strongly_connected_subgraphs, DFN_3, LOW_3, stack_subgraphs,
             successors_Subgraphs);
    }
  }
  std::string file_name_scc3 = "scc3.txt";
  std::ofstream outfile_scc3(file_name_scc3);
  for (const auto &scc : strongly_connected_subgraphs)
  {
    std::cout << "scc:";
    outfile_scc3 << "scc: ";
    for (const auto &scc_id : scc)
    {
      outfile_scc3 << scc_id << " ";
    }
    outfile_scc3 << std::endl;
    for (const auto &scc_id : scc)
    {
      std::cout << scc_id << " ";
      outfile_scc3 << "subgraph" << scc_id << " input:";
      for (const auto &scc_input : graphs_inputs[scc_id])
      {
        outfile_scc3 << scc_input.name << ";";
      }
      outfile_scc3 << " output:";
      for (const auto &scc_output : graphs_outputs[scc_id])
      {
        outfile_scc3 << scc_output.name << ";";
      }
      outfile_scc3 << std::endl;
    }

    std::cout << std::endl;
  }
  outfile_scc.close();
  free(DFN_3);
  free(LOW_3);
  std::cout << "node_num after cut " << node_num_all << std::endl;
  if (node_num_all != g.node_size())
  {
    std::cout << "num error!" << std::endl;
    exit(0);
  }
  int count_cut_pair = 0;
  while (1)
  {
    count_cut_pair++;
    if (count_cut_pair > 15)
    {
      std::cout << "cut pair error! So many times!" << std::endl;
      exit(0);
      break;
    }
    int subgraph_size = Subgraphs.size();
    std::vector<std::vector<int>> strongly_connected_subgraphs_all;
    std::vector<int> scc_all;
    for (int i = 0; i < int(Subgraphs.size()) + int(otherSubgraphs.size()); i++)
    {
      scc_all.push_back(i);
    }
    strongly_connected_subgraphs_all.push_back(scc_all);
    if (((count_cut_pair > 1 && count_cut_pair < 5) ||
         (count_cut_pair > 10 && count_cut_pair < 13)) &&
        strongly_connected_subgraphs.size() != 0)
    {
      std::cout << count_cut_pair << " eliminate scc v2 executed" << std::endl;
      eliminate_scc_v2(strongly_connected_subgraphs, Subgraphs, otherSubgraphs, g);
      // eliminate_pair_v2(Subgraphs, otherSubgraphs, graphs_inputs, graphs_outputs,
      // strongly_connected_subgraphs_all, subgraph_size);
    }
    else if (((count_cut_pair == 15)) && strongly_connected_subgraphs.size() != 0)
    {
      std::cout << count_cut_pair << " eliminate scc v3 executed" << std::endl;
      eliminate_scc_v3(strongly_connected_subgraphs, Subgraphs, otherSubgraphs, g);
    }
    else
    {
      std::cout << count_cut_pair << " eliminate pair v2 executed" << std::endl;
      eliminate_pair_v2(Subgraphs, otherSubgraphs, graphs_inputs, graphs_outputs,
                        strongly_connected_subgraphs_all, subgraph_size);
    }
    strongly_connected_subgraphs.clear();
    predecessors_Subgraphs.clear();
    successors_Subgraphs.clear();
    std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_2_inputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_1_inputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_2_outputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(subgraphs_1_outputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(graphs_inputs);
    std::vector<std::unordered_set<NodeTensor>>().swap(graphs_outputs);
    for (const auto &sg : Subgraphs)
    {
      std::unordered_set<NodeTensor> graphInputs;
      determineGraphInput(sg, IOvalueNames, graphInputs);
      subgraphs_1_inputs.push_back(graphInputs);
    }
    for (const auto &sg : otherSubgraphs)
    {
      std::unordered_set<NodeTensor> graphInputs;
      determineGraphInput(sg, IOvalueNames, graphInputs);
      subgraphs_2_inputs.push_back(graphInputs);
    }
    node_number = 0;
    for (const auto &sg : Subgraphs)
    {
      std::unordered_set<NodeTensor> graphOutputs;
      node_number += sg.node_size();
      determineGraphOutput(g, sg, subgraphs_1_inputs, subgraphs_2_inputs, graphOutputs);
      subgraphs_1_outputs.push_back(graphOutputs);
    }
    for (const auto &sg : otherSubgraphs)
    {
      std::unordered_set<NodeTensor> graphOutputs;
      node_number += sg.node_size();
      determineGraphOutput(g, sg, subgraphs_1_inputs, subgraphs_2_inputs, graphOutputs);
      subgraphs_2_outputs.push_back(graphOutputs);
    }
    graphs_inputs.insert(graphs_inputs.end(), subgraphs_1_inputs.begin(), subgraphs_1_inputs.end());
    graphs_inputs.insert(graphs_inputs.end(), subgraphs_2_inputs.begin(), subgraphs_2_inputs.end());
    graphs_outputs.insert(graphs_outputs.end(), subgraphs_1_outputs.begin(),
                          subgraphs_1_outputs.end());
    graphs_outputs.insert(graphs_outputs.end(), subgraphs_2_outputs.begin(),
                          subgraphs_2_outputs.end());
    for (int i = 0; i < int(graphs_inputs.size()); i++)
    {
      std::vector<int> predecessors;
      for (const auto &g_input : graphs_inputs[i])
      {
        for (int j = 0; j < int(graphs_outputs.size()); j++)
        {
          if ((graphs_outputs[j].find(g_input) != graphs_outputs[j].end()))
          {
            predecessors.push_back(j);
          }
        }
      }
      predecessors_Subgraphs.push_back(predecessors);
    }
    for (int i = 0; i < int(graphs_inputs.size()); i++)
    {
      std::vector<int> temp;
      for (int j = 0; j < int(graphs_inputs.size()); j++)
      {
        if (find(predecessors_Subgraphs[j].begin(), predecessors_Subgraphs[j].end(), i) !=
            predecessors_Subgraphs[j].end())
        {
          temp.push_back(j);
        }
      }
      successors_Subgraphs.push_back(temp);
    }
    node_num_all = 0;
    for (const auto &sg : Subgraphs)
    {
      node_num_all += sg.node_size();
    }
    for (const auto &sg : otherSubgraphs)
    {
      node_num_all += sg.node_size();
    }
    int *DFN_4 = (int *)malloc(graphs_inputs.size() * sizeof(int));
    int *LOW_4 = (int *)malloc(graphs_inputs.size() * sizeof(int));
    for (int i = 0; i < int(graphs_inputs.size()); i++)
    {
      DFN_4[i] = 0;
      LOW_4[i] = 0;
    }
    for (int temp_count = 0; temp_count < int(predecessors_Subgraphs.size()); temp_count++)
    {
      if (DFN_[temp_count] == 0)
      {
        std::vector<int> stack_subgraphs;
        int depth = 0;
        Tarjan(temp_count, depth, strongly_connected_subgraphs, DFN_4, LOW_4, stack_subgraphs,
               successors_Subgraphs);
      }
    }
    std::string file_name_scc4 = "scc4.txt";
    std::ofstream outfile_scc4(file_name_scc4);
    for (const auto &scc : strongly_connected_subgraphs)
    {
      std::cout << "scc4:";
      for (const auto &scc_id : scc)
      {
        std::cout << scc_id << " ";
        outfile_scc4 << "subgraph" << scc_id << " input:";
        for (const auto &scc_input : graphs_inputs[scc_id])
        {
          outfile_scc4 << scc_input.name << ";";
        }
        outfile_scc4 << " output:";
        for (const auto &scc_output : graphs_outputs[scc_id])
        {
          outfile_scc4 << scc_output.name << ";";
        }
        outfile_scc4 << std::endl;
      }

      std::cout << std::endl;
    }
    outfile_scc.close();
    free(DFN_4);
    free(LOW_4);
    std::cout << "node num in original graph: " << g.node_size() << std::endl;
    std::cout << "node_num after cut " << node_num_all << std::endl;
    if (node_num_all != g.node_size())
    {
      std::cout << "num error!, time" << count_cut_pair << std::endl;
      exit(0);
    }
    if (count_cut_pair == 15)
    {
      if (strongly_connected_subgraphs.size() == 0)
      {
        break;
      }
      else
      {
        std::cout << "error!" << std::endl;
        exit(0);
      }
    }
    std::cout << "graph number after " << count_cut_pair
              << "loops: " << Subgraphs.size() + otherSubgraphs.size() << std::endl;
  } // end of while
  std::string file_name_predecessor_4 = "predecessor_final_4.txt";
  std::string file_name_successor_4 = "successor_final_4.txt";
  std::ofstream outfile_predecessor_4(file_name_predecessor_4);
  std::ofstream outfile_successor_4(file_name_successor_4);
  if (!(outfile_predecessor_4.is_open() && outfile_successor_4.is_open()))
  {
    std::cerr << "Error opening file." << std::endl;
    exit(0);
  }
  for (int i = 0; i < int(graphs_inputs.size()); i++)
  {
    outfile_predecessor_4 << "predecessor of subgraph " << i << ":";
    for (const auto &predecessor : predecessors_Subgraphs[i])
    {
      outfile_predecessor_4 << predecessor << ";";
    }
    outfile_predecessor_4 << std::endl;
    outfile_successor_4 << "successor of subgraph " << i << ":";
    for (const auto &successor : successors_Subgraphs[i])
    {
      outfile_successor_4 << successor << ";";
    }
    outfile_successor_4 << std::endl;
  }
  outfile_predecessor_4.close();
  outfile_successor_4.close();
  print_subgraphs(Subgraphs, (char *)"./subgraphs_final_4.txt", otherSubgraphs,
                  (char *)"./other_subgraphs_final_4.txt");
  ////*
  int temp_count_subgraph = 0;

  std::ofstream outfile_conv_flag("end_with_conv.txt");
  for (const auto &graph_outputs : subgraphs_1_outputs)
  {
    int find_flag = 0;
    for (const auto &graph_output : graph_outputs)
    {
      for (const auto &node : Subgraphs[temp_count_subgraph].node())
      {
        for (const auto &output : node.output())
        {
          if (graph_output.name == output && node.op_type() == "Conv")
          {
            outfile_conv_flag << temp_count_subgraph << " ";
            find_flag = 1;
            break;
          }
        }
        if (find_flag)
        {
          break;
        }
      }
      if (find_flag)
      {
        break;
      }
    }
    temp_count_subgraph++;
  }
  outfile_conv_flag.close();
  std::cout << "succeeded in reaching sorting" << std::endl;
  int finished_flag = 0;
  int sort_count = 0;
  std::vector<int> order_Subgraphs(graphs_inputs.size());
  std::vector<int> issort_Subgraphs(graphs_inputs.size());
  while (!finished_flag)
  {
    finished_flag = 1;
    int changed_sort_flag = 0;
    if (sort_count == 0)
    {
      changed_sort_flag = 1;
      for (int i = 0; i < int(graphs_inputs.size()); i++)
      {
        int find_flag = 0;
        for (const auto &g_input : graphs_inputs[i])
        {
          for (int j = 0; j < int(graphs_outputs.size()); j++)
          {
            if (graphs_outputs[j].find(g_input) != graphs_outputs[j].end())
            {
              find_flag = 1;
              break;
            }
          }
          if (find_flag)
          {
            break;
          }
        }
        if (!find_flag)
        {
          order_Subgraphs[i] = 0;
          issort_Subgraphs[i] = 1;
        }
        else
        {
          order_Subgraphs[i] = 1;
          issort_Subgraphs[i] = 0;
          finished_flag = 0;
        }
      }
    }
    else
    {
      std::cout << "sort count:" << sort_count << std::endl;
      for (int i = 0; i < int(graphs_inputs.size()); i++)
      {
        int find_flag = 0;
        if (issort_Subgraphs[i] == 1 && i != int(graphs_inputs.size()) - 1)
        {
          continue;
        }
        for (const auto &g_input : graphs_inputs[i])
        {
          for (int j = 0; j < int(graphs_outputs.size()); j++)
          {
            if ((graphs_outputs[j].find(g_input) != graphs_outputs[j].end()))
            {
              if ((issort_Subgraphs[j] == 0))
              {
                std::cout << "graph " << i << "is after graph " << j << std::endl;
                find_flag = 1;
                break;
              }
            }
          }
          if (find_flag)
          {
            break;
          }
        }
        if (!find_flag)
        {
          if (!(issort_Subgraphs[i] == 1))
          {
            order_Subgraphs[i] = sort_count;
          }
        }
        else
        {
          order_Subgraphs[i] = sort_count + 1;
          issort_Subgraphs[i] = 0;
          finished_flag = 0;
        }
        if (i == int(graphs_inputs.size()) -
                   1) // add the subgraph to the queue only when cycle is completed to prevent the
                      // newly added subgraph in this cycle from being the predecessor of the
                      // subsequent sub-graph.
        {
          for (int j = 0; j < int(graphs_inputs.size()); j++)
          {
            if (order_Subgraphs[j] == sort_count)
            {
              issort_Subgraphs[j] = 1;
              changed_sort_flag = 1;
              std::cout << "graph " << j << " is in the " << sort_count << "th sort" << std::endl;
            }
          }
        }
      }
      if (changed_sort_flag == 0)
      {
        std::cout << "error: endless loop!" << std::endl;
        std::cout << "sort count:" << sort_count << std::endl;
        std::cout << "count_cut_pair: " << count_cut_pair << std::endl;
        for (int i = 0; i < int(graphs_inputs.size()); i++)
        {
          std::cout << "order_Subgraphs[" << i << "]:" << order_Subgraphs[i] << " ";
        }
        std::cout << std::endl;
        std::exit(1);
        break;
      }
    }
    sort_count++;
  }
  char *sub1_type, *sub2_type;
  if (strategy == SPILTE_CPU_STRUCTURE_FIRST)
  {
    sub1_type = (char *)"CPU";
    sub2_type = (char *)"NPU";
  }
  else
  {
    sub1_type = (char *)"NPU";
    sub2_type = (char *)"CPU";
  }
  std::cout << " order" << std::endl;
  for (auto element : order_Subgraphs)
  {
    std::cout << element << " ";
  }
  std::cout << std::endl;

  std::string file_name = "subgraphs_ios.txt";
  std::ofstream outfile1(file_name);
  if (!outfile1.is_open())
  {
    std::cerr << "Error opening file." << std::endl;
    exit(0);
  }
  int sub1_size = subgraphs_1_inputs.size();
  for (int i = 0; i < int(graphs_inputs.size()); i++)
  {
    outfile1 << (i >= sub1_size ? sub2_type : sub1_type) << "subgraph"
             << (i >= sub1_size ? (i - sub1_size) : i) << ": order" << order_Subgraphs[i];
    outfile1 << "--input-name ";
    std::cout << (i >= sub1_size ? sub2_type : sub1_type) << "subgraph"
              << (i >= sub1_size ? (i - sub1_size) : i) << ": order" << order_Subgraphs[i]
              << std::endl;
    std::cout << "Inputs:";
    for (auto element : graphs_inputs[i])
    {
      std::cout << element.name << "; size:";
      for (auto Size : element.shape)
      {
        std::cout << Size << " ";
      }
      outfile1 << element.name << ";";
    }
    std::cout << std::endl;
    std::cout << "Outputs:";
    outfile1 << "--output-name ";
    for (auto element : graphs_outputs[i])
    {
      std::cout << element.name << "; size:";
      for (auto Size : element.shape)
      {
        std::cout << Size << " ";
      }
      outfile1 << element.name << ";";
    }
    outfile1 << std::endl;
    std::cout << std::endl;
    std::cout << " The predecessors of " << (i >= sub1_size ? sub2_type : sub1_type) << "subgraph"
              << (i >= sub1_size ? (i - sub1_size) : i) << ": ";
    for (auto element : predecessors_Subgraphs[i])
    {
      std::cout << (element >= sub1_size ? sub2_type : sub1_type) << "subgraph"
                << (element >= sub1_size ? (element - sub1_size) : element) << "; ";
    }
    std::cout << std::endl;
    std::cout << " The successors of " << (i >= sub1_size ? sub2_type : sub1_type) << "subgraph"
              << (i >= sub1_size ? (i - sub1_size) : i) << ": ";
    for (auto element : successors_Subgraphs[i])
    {
      std::cout << (element >= sub1_size ? sub2_type : sub1_type) << "subgraph"
                << (element >= sub1_size ? (element - sub1_size) : element) << "; ";
    }
    std::cout << std::endl;
  }
  outfile1.close();
  for (const auto &tensor : IOvalueNames)
  {
    std::cout << "Name: " << tensor.name << ", Shape: [";
    for (size_t i = 0; i < tensor.shape.size(); ++i)
    {
      std::cout << tensor.shape[i];
      if (i < tensor.shape.size() - 1)
      {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

  switch (d.getType())
  {
    case DeviceType::Target_NPU:
    {
      if (strategy == SPILTE_CPU_STRUCTURE_FIRST)
      {
        d.GenerateCutInstruction(Subgraphs, "cpu", subgraphs_1_inputs, subgraphs_1_outputs);
        d.GenerateCutInstruction(otherSubgraphs, "npu", subgraphs_2_inputs, subgraphs_2_outputs);
      }
      else if (strategy == SPILTE_NPU_STRUCTURE_FIRST)
      {
        d.GenerateCutInstruction(Subgraphs, "npu", subgraphs_1_inputs, subgraphs_1_outputs);
        d.GenerateCutInstruction(otherSubgraphs, "cpu", subgraphs_2_inputs, subgraphs_2_outputs);
      }
      break;
    }
    default:
      std::cout << "Unknown device type" << std::endl;
      exit(0);
  }
  std::cout << "node num in original graph: " << g.node_size() << std::endl;
  std::cout << "node_num after cut " << node_num_all << std::endl;
}
