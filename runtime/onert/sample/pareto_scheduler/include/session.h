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
 * @file  session.h
 * @brief This file describes API for session run and reconfiguration
 */
#ifndef _PARETO_SESSION_H
#define _PARETO_SESSION_H

#include <string>
#include <unistd.h>
#include <vector>
#include "common.h"
#include "nnfw.h"
#include "pareto_lookup.h"
#include "tracer.h"

// Use globals declared in main application
extern MessageQueue_t mq;
extern JsonWriter *json;
extern ParetoOptimizer *opt;

class RunSession
{
private:
  std::string _model;          //  Model file
  nnfw_session *_session;      // Session
  std::vector<void *> _inputs; // input tensors
  bool _inputs_initialized;    // avoid re-allocation between runs

  uint64_t num_elems(const nnfw_tensorinfo *ti);
  template <typename T> void random_input_float(float *vec, T nelements);
  template <typename T1, typename T2> void random_input_int(T1 *vec, T2 nelements);

public:
  /**
   * @brief     Construct a new RunSession object
   * @param[in] std::string model. nnpackage path for model
   */
  RunSession(std::string model);
  /**
   * @brief     Load session, initialize tensors, and assign all backends to acl_cl
   */
  void load_session(void);
  /**
   * @brief     wrapper to check whether latency exceeds a threshold margin
   * @param[in] float exec_time execution time of most recent inference run
   *
   * @return    bool true if latency exceeds margin, false otherwise
   */
  bool latency_increased(float exec_time);
  /**
   * @brief     wrapper to check whether increase in system memory can be exploited to revert back
   * to GPU-based setting
   * @param[in] int321_t memory_diff increase in available system memory, in MB
   *
   * @return    bool true if the session can be configured to a low-latency/ high RSS memory
   * setting, false otherwise
   */
  bool memory_improved(int32_t memory_diff);
  /**
   * @brief     reconfigure backends such that resulting latency is close to and within execution
   * time bounds
   * @param[in] float exec_time_limit execution time upper bound in milliseconds
   */
  void reconfigure_within_exec_time(float exec_time);
  /**
   * @brief     reconfigure backends configuration such that resulting memory usage is within
   * allowable limit
   * @param[in] int32_t memory_diff allowable memory increase in MB
   *
   * @return    std::string mapping model operation indexes to their assigned backends (cpu, acl_cl)
   */
  void reconfigure_within_memory(int32_t memory_val);
  /**
   * @brief     run inference
   *
   * @return    int64_t elapsed time in microseconds
   */
  int64_t run_inference(void);
  /**
   * @brief     close session
   */
  void close(void);
  /**
   * @brief     initialize input tensors
   */
  void initialize_inputs(void);
  /**
   * @brief     initialize output tensors
   */
  void prepare_output(void);
  /**
   * @brief     wrapper to get currently set reference performance metrics
   *
   * @return    std::string tuple in the format (reference execution time, reference RSS memory)
   */
  std::string get_pareto_setting(void);
};
#endif
