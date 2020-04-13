/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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
 * @file execution.h
 * @brief This file contains ANeuralNetworksExecution class for handling Execution NNAPI such as
 * ANeuralNetworksExecution_create, ANeuralNetworksExecution_setInput
 * @ingroup COM_AI_RUNTIME
 */

#ifndef __EXECUTION_H__
#define __EXECUTION_H__

#include "internal/arm_compute.h"
#include "internal/Sink.h"
#include "internal/Source.h"

/**
 * @brief struct to express Execution of NNAPI
 */
struct ANeuralNetworksExecution
{
public:
  /**
   * @brief Construct with params
   * @param [in] plan Pointer to get internal::arm_compute::Plan
   */
  ANeuralNetworksExecution(const std::shared_ptr<const internal::arm_compute::Plan> &plan)
      : _plan{plan}
  {
    _sources.resize(_plan->model().inputs.size());
    _sinks.resize(_plan->model().outputs.size());
  }

public:
  /**
   * @brief Get reference of internal::arm_compute::Plan
   * @return Const reference of internal::arm_compute::Plan
   */
  const internal::arm_compute::Plan &plan(void) const { return *_plan; }

private:
  std::shared_ptr<const internal::arm_compute::Plan> _plan;

public:
  /**
   * @brief Set the nth source with param
   * @param [in] n Index of the nth source
   * @param [in] source Pointer to set the nth source from
   * @return N/A
   */
  // TODO Use InputIndex instead of int
  void source(int n, std::unique_ptr<Source> &&source) { _sources.at(n) = std::move(source); }
  /**
   * @brief Set the nth source with param
   * @param [in] n Index of the nth source
   * @param [in] args Arguments to set the nth source from
   * @return N/A
   */
  template <typename T, typename... Args> void source(int n, Args &&... args)
  {
    source(n, std::unique_ptr<T>{new T{std::forward<Args>(args)...}});
  }

public:
  /**
   * @brief Get the nth source
   * @param [in] n Index of the nth source
   * @return Const reference of Source
   */
  const Source &source(int n) const { return *(_sources.at(n)); }

public:
  /**
   * @brief Set the nth sink with param
   * @param [in] n Index of the nth sink
   * @param [in] sink Pointer to set the nth sink from
   * @return N/A
   */
  // TODO Use OutputIndex instead of int
  void sink(int n, std::unique_ptr<Sink> &&sink) { _sinks.at(n) = std::move(sink); }
  /**
   * @brief Set the nth sink with param
   * @param [in] n Index of the nth sink
   * @param [in] args Arguments to set the nth sink from
   * @return N/A
   */
  template <typename T, typename... Args> void sink(int n, Args &&... args)
  {
    sink(n, std::unique_ptr<T>{new T{std::forward<Args>(args)...}});
  }

public:
  /**
   * @brief Get the nth sink
   * @param [in] n Index of the nth sink
   * @return Const reference of Sink
   */
  const Sink &sink(int n) const { return *(_sinks.at(n)); }

private:
  std::vector<std::unique_ptr<Source>> _sources;
  std::vector<std::unique_ptr<Sink>> _sinks;
};

#endif
