/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "core/OMTrainingRuntimeModule.h"
#include "import/OMExecutionPlanCreator.h"
#include "train/OMBackpropExecute.h"

using namespace onert_micro::core;
using namespace onert_micro;

/*
 * Import train model, validate it, prepare necessary structures and data for future calculations
 */
OMStatus OMTrainingRuntimeModule::importTrainModel(const char *model_ptr, const OMConfig &config)
{
  // 1 - Import main model
  // 2 - Import and create training entities

  // 1 - Import main model
  OMStatus status = importModel(model_ptr, config);
  if (status != Ok)
    return status;

  // 2 - Import and create training entities
  _backward_graphs.resize(_graphs.size());
  for (uint32_t i = 0; i < _backward_graphs.size(); ++i)
  {
    // load default backward graph
    OMRuntimeGraph &graph = _backward_graphs.at(i);

    OMRuntimeContext &runtime_context = graph.getRuntimeContext();
    OMRuntimeStorage &runtime_storage = graph.getRuntimeStorage();
    memory::OMRuntimeAllocator &runtime_allocator = graph.getRuntimeAllocator();

    runtime_context.setModel(model_ptr, i);

    // Parse and validate WOF file if it is exist
    // WARNING: setWofFile method of RuntimeContext should follow after setModel.
    if (config.wof_ptr != nullptr)
      runtime_context.setWofFile(config.wof_ptr);

    // AllocDeallocPlan backward graph creation
    status = import::OMExecutionPlanCreator::createBackwardExecutionPlan(
      runtime_storage, runtime_context, runtime_allocator, config);
    if (status != Ok)
      return status;
  }

  // Set current optimizer
  _training_handler.setOptimizer(config);

  return Ok;
}

/*
 *  Train single step: run forward graph (with data which was set in SetInput) ->
 *  -> calculate error (with target data which was set in SetTarget) ->
 *  -> run backward graph -> update optimizer state -> after batch_size steps update weights
 *  Warning: before using trainSingleStep call: 1) importTrainModel; 2) setInput; 3) setTarget
 */
OMStatus OMTrainingRuntimeModule::trainSingleStep(const OMConfig &config)
{
  OMStatus status = Ok;
  uint32_t batch_size = config.training_context.batch_size;

  // A - Run and update optimizers state batch_size times
  //  a. Allocate forward inputs and memcopy current sample (in current batch) into forward inputs
  //  b. Run forward graph
  //  c. Handle (calculate) current error backpropagation function result
  //  d. Run backward graph (calculate backpropagation values)
  //  e. Update optimization state
  //  f. Reset runtime graphs (to deallocate forward outputs)
  // B - Update weights according chosen optimization strategy (after batch_size times)

  // A
  for (uint32_t b = 0; b < batch_size; ++b)
  {
    //  a. Allocate forward inputs and memcopy current sample (in current batch) into forward inputs
    {
      OMRuntimeGraph &forward_main_graph = _graphs.at(0);
      forward_main_graph.allocateGraphInputs();
      uint32_t input_nums = forward_main_graph.getNumberOfInputs();
      for (uint32_t i = 0; i < input_nums; ++i)
      {
        uint8_t *allocated_input_data =
          reinterpret_cast<uint8_t *>(forward_main_graph.getInputDataAt(i));
        size_t type_size = forward_main_graph.getInputDataTypeSize(i);
        size_t tensor_size = forward_main_graph.getInputSizeAt(i);
        size_t offset = b * type_size * tensor_size;

        uint8_t *input_data = _training_handler.getInputData(i);
        std::memcpy(allocated_input_data, input_data + offset, tensor_size * type_size);
      }
    }

    //  b. Run forward graph
    {
      status = run();
      assert(status == Ok);
      if (status != Ok)
        return status;
    }

    if (_backward_graphs.empty())
      return ModelNotImport;

    for (uint32_t i = 0; i < _graphs.size(); ++i)
    {
      // Get forward entities
      OMRuntimeGraph &forward_graph = _graphs.at(i);
      OMRuntimeContext &forward_context = forward_graph.getRuntimeContext();
      OMRuntimeStorage &forward_storage = forward_graph.getRuntimeStorage();

      // Get backward entities
      OMRuntimeGraph &backward_graph = _backward_graphs.at(i);
      OMRuntimeContext &backward_context = backward_graph.getRuntimeContext();
      OMRuntimeStorage &backward_storage = backward_graph.getRuntimeStorage();

      //  c. Handle (calculate) current error backpropagation function result
      {
        status = _training_handler.handleError(config, forward_storage, backward_storage,
                                               backward_context, b);
        assert(status == Ok);
        if (status != Ok)
          return status;
      }

      //  d. Run backward graph
      {
        onert_micro::train::OMBackpropExecuteArgs backprop_execute_args = {
          forward_storage, backward_storage, backward_context, false, 0};

        status = onert_micro::train::OMBackpropExecute::runBackward(
          config, backprop_execute_args, backward_graph.getRuntimeAllocator());
        assert(status == Ok);
        if (status != Ok)
          return status;
      }

      //  e. Update optimization state
      {
        status = _training_handler.updateOptimizerState(config, backward_storage, backward_context);
        assert(status == Ok);
        if (status != Ok)
          return status;
      }

      //  f. Reset runtime graphs
      backward_graph.reset();
      forward_graph.reset();
    }
  }

  // B - Update weights according chosen optimization strategy
  for (uint32_t i = 0; i < _graphs.size(); ++i)
  {
    // Get backward context
    OMRuntimeGraph &backward_graph = _backward_graphs.at(i);
    OMRuntimeContext &backward_context = backward_graph.getRuntimeContext();
    status = _training_handler.updateWeights(config, backward_context);
  }

  return status;
}

/*
 * Evaluate metrics
 *
 *  Warning: 1) assume that all metric_val for all OMMetrics types actually are float values.
 *           2) metric_val should be initialized with some value before calling this method due to
 *              after calculation for current batch_num (the sequence number of the current sample)
 *              this value is added to metric_val
 */
OMStatus OMTrainingRuntimeModule::evaluateMetric(OMMetrics metric, void *metric_val,
                                                 uint32_t test_size)
{
  OMStatus status = Ok;
  OMRuntimeGraph &forward_main_graph = _graphs.at(0);
  uint32_t input_nums = forward_main_graph.getNumberOfInputs();
  for (uint32_t b = 0; b < test_size; ++b)
  {
    //  a. Allocate forward inputs and memcopy current sample (in current batch) into forward inputs
    {
      forward_main_graph.allocateGraphInputs();
      for (uint32_t i = 0; i < input_nums; ++i)
      {
        uint8_t *allocated_input_data =
          reinterpret_cast<uint8_t *>(forward_main_graph.getInputDataAt(i));
        size_t type_size = forward_main_graph.getInputDataTypeSize(i);
        size_t tensor_size = forward_main_graph.getInputSizeAt(i);
        size_t offset = b * type_size * tensor_size;

        uint8_t *input_data = _training_handler.getInputData(i);
        std::memcpy(allocated_input_data, input_data + offset, tensor_size * type_size);
      }
    }

    //  b. Run forward graph
    {
      status = run();
      assert(status == Ok);
      if (status != Ok)
        return status;
    }

    // c. Calculate loss on test_size data
    _training_handler.evaluateMetric(metric, metric_val, forward_main_graph.getRuntimeStorage(),
                                     forward_main_graph.getRuntimeContext(), b);

    // Reset values
    forward_main_graph.reset();
  }

  // Averaged result
  {
    float *f_metric_val = reinterpret_cast<float *>(metric_val);
    *f_metric_val /= test_size;
  }

  return Ok;
}

/*
 * Reset all forward and backward graphs
 */
OMStatus OMTrainingRuntimeModule::reset()
{
  OMStatus status = Ok;

  if (_graphs.empty())
    return ModelNotImport;

  for (auto &graph : _graphs)
  {
    graph.reset();
  }

  for (auto &graph : _backward_graphs)
  {
    graph.reset();
  }

  _training_handler.reset();

  return status;
}
