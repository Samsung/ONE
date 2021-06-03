/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

/**
 * @file  pareto_lookup.h
 * @brief This file describes API for pareto front lookup and optimization
 */

#ifndef _PARETO_LOOKUP_H
#define _PARETO_LOOKUP_H
#include <unordered_map>

class ParetoOptimizer
{
private:
  std::string _json_file; // Pareto config obtained from estimation
  // Mapping from backend configuration (encoded as integer string) to performance metrics
  // (execution time, RSS memory)
  std::unordered_map<std::string, std::tuple<float, int32_t>> _solution_map;
  // Mapping from backend configuration (encoded as integer string) to a list of <operation index,
  // backend> tuples encoded as string
  std::unordered_map<std::string, std::string> _backend_allocations;
  // Currently used Pareto reference points
  float _current_reference_time;
  int32_t _current_reference_memory;
  int32_t _largest_rss_value; // ceiling for memory consumption
  // Precomputed placeholders for performance monitoring
  float _threshold_time_check;
  int32_t _threshold_memory_check;
  int _threshold_margin_latency;
  int _threshold_margin_memory;

public:
  /**
   * @brief     Construct a new ParetoOptimizer object
   * @param[in] json_file Config file containing pareto front information
   */
  ParetoOptimizer(std::string json_file);
  /**
   * @brief     Initialize maps for indexing performance metrics and backend configurations
   */
  void initialize_maps(void);
  /**
   * @brief     print performance metrics and backend configurations
   */
  void print_maps(void);
  /**
   * @brief     return smallest execution time from pareto front
   *
   * @return    float smallest execution time
   */
  float fetch_smallest_exec(void);
  /**
   * @brief     fetch backend configuration corresponding to smallest execution time
   *
   * @return    std::string mapping model operation indexes to their assigned backends (cpu, acl_cl)
   */
  std::string fetch_config_with_smallest_exec(void);
  /**
   * @brief     fetch backend configuration corresponding to smallest RSS memory
   *
   * @return    std::string mapping model operation indexes to their assigned backends (cpu, acl_cl)
   */
  std::string fetch_config_with_smallest_memory(void);
  /**
   * @brief     fetch backend configuration close to and within execution time bounds
   * @param[in] float exec_time_limit execution time upper bound in milliseconds
   *
   * @return    std::string mapping model operation indexes to their assigned backends (cpu, acl_cl)
   */
  std::string fetch_config_within_exectime(float exec_time_limit);
  /**
   * @brief     fetch backend configuration close to and within bounded memory increase
   * @param[in] int32_t memory_diff allowable memory increase in MB
   *
   * @return    std::string mapping model operation indexes to their assigned backends (cpu, acl_cl)
   */
  std::string fetch_config_within_memory(int32_t memory_diff);
  /**
   * @brief     fetch backend configuration close to and within bounded RSS memory
   * @param[in] uint32_t currently_available_memory upper bound on memory in MB
   *
   * @return    std::string mapping model operation indexes to their assigned backends (cpu, acl_cl)
   */
  std::string fetch_config_within_memory_bound(uint32_t currently_available_memory);
  /**
   * @brief     check whether latency exceeds a threshold margin
   * @param[in] float exec_time execution time of most recent inference run
   *
   * @return    bool true if latency exceeds margin, false otherwise
   */
  bool exec_time_increased(float exec_time);
  /**
   * @brief     check whether increase in system memory can be exploited to revert back to GPU-based
   * setting
   * @param[in] int321_t memory_diff increase in available system memory, in MB
   *
   * @return    bool true if the session can be configured to a low-latency/ high RSS memory
   * setting, false otherwise
   */
  bool feasible_memory_increase(int32_t memory_diff);
  /**
   * @brief     get currently set reference performance metrics
   *
   * @return    std::string tuple in the format (reference execution time, reference RSS memory)
   */
  std::string get_current_setting(void);
  int32_t get_pareto_rss(void);
  /**
   * @brief     set threshold margins (in percentage) for latency and memory. The margins are
   * checked against at runtime by the controller
   * @param[in] int margin_latency, latency margin as relative percentage [0-100]
   * @param[in] int margin_memory, memory margin as relative percentage [0-100]
   */
  void set_thresholds(int margin_latency, int margin_memory);
};
#endif
