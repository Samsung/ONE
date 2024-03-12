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

#include "core/OMDataType.h"
#include "core/train/OMTrainingRuntimeModule.h"

#include "import/OMExecutionPlanCreator.h"
#include "import/OMKernelConfiguration.h"
#include "import/OMConfigureArgs.h"

#include "optimize/OMOptimizer.h"

#include "execute/OMKernelExecute.h"

#include <unordered_set>
#include <cmath>

// TODO: remove it
#include <iostream>

using namespace onert_micro;
using namespace onert_micro::core;
using namespace onert_micro::core::train;

namespace
{

uint32_t getGraphsCount(const char *model_ptr)
{
  reader::OMCircleReader reader;
  reader.parse(model_ptr);

  uint32_t num_subgraph = reader.num_subgraph();

  assert(num_subgraph >= 1 && "Should exist at list one graph");

  return num_subgraph;
}

} // namespace

uint32_t OMTrainingRuntimeModule::getNumberOfInputs()
{
  assert(_main_runtime_graphs.size() > 0);
  if (_main_runtime_graphs.size() == 0)
    return 0;

  return _main_runtime_graphs.at(0).getNumberOfInputs();
}

uint32_t OMTrainingRuntimeModule::getNumberOfOutputs()
{
  assert(_main_runtime_graphs.size() > 0);
  if (_main_runtime_graphs.size() == 0)
    return 0;

  return _main_runtime_graphs.at(0).getNumberOfOutputs();
}

uint32_t OMTrainingRuntimeModule::getNumberOfTargets()
{
  return _training_storage.getTargetsIndexes().size();
}

uint32_t OMTrainingRuntimeModule::getInputSizeAt(uint32_t position)
{
  assert(_main_runtime_graphs.size() > 0);
  if (_main_runtime_graphs.size() == 0)
    return 0;

  return _main_runtime_graphs.at(0).getInputSizeAt(position);
}

uint32_t OMTrainingRuntimeModule::getOutputSizeAt(uint32_t position)
{
  assert(_main_runtime_graphs.size() > 0);
  if (_main_runtime_graphs.size() == 0)
    return 0;

  return _main_runtime_graphs.at(0).getOutputSizeAt(position);
}

uint32_t OMTrainingRuntimeModule::getTargetSizeAt(uint32_t position)
{
  auto &targets_indexes = _training_storage.getTargetsIndexes();
  assert(targets_indexes.size() >= position);
  if (targets_indexes.size() < position)
    return 0;

  assert(_backpropagation_runtime_graphs.size() > 0);
  if (_backpropagation_runtime_graphs.size() == 0)
    return 0;

  const auto target_index = targets_indexes.at(position);
  const circle::Tensor *target_tensor = _backpropagation_runtime_graphs.at(0).getRuntimeContext().getTensorByIndex(target_index);

  OMRuntimeShape shape(target_tensor);
  return shape.flatSize();
}

void *OMTrainingRuntimeModule::getInputDataAt(uint32_t position)
{
  assert(_main_runtime_graphs.size() > 0);
  if (_main_runtime_graphs.size() == 0)
    return 0;

  return _main_runtime_graphs.at(0).getInputDataAt(position);
}

void *OMTrainingRuntimeModule::getOutputDataAt(uint32_t position)
{
  assert(_main_runtime_graphs.size() > 0);
  if (_main_runtime_graphs.size() == 0)
    return 0;

  return _main_runtime_graphs.at(0).getOutputDataAt(position);
}

void *OMTrainingRuntimeModule::getTargetDataAt(uint32_t position)
{
  auto &targets_indexes = _training_storage.getTargetsIndexes();
  assert(targets_indexes.size() > position);
  if (targets_indexes.size() <= position)
    return 0;

  assert(_main_runtime_graphs.size() > 0);
  if (_main_runtime_graphs.size() == 0)
    return 0;

  auto &storage = _backpropagation_runtime_graphs.at(0).getRuntimeStorage();
  auto &context = _backpropagation_runtime_graphs.at(0).getRuntimeContext();
  auto &allocator = _backpropagation_runtime_graphs.at(0).getRuntimeAllocator();

  const auto target_index = targets_indexes.at(position);
  uint8_t *data;
  storage.getDataByTensorIndex(&data, target_index);

  if (data == nullptr)
  {
    allocator.allocateGraphInputs(&context, &storage, targets_indexes);
    storage.getDataByTensorIndex(&data, target_index);
  }

  return reinterpret_cast<void *>(data);
}


OMStatus OMTrainingRuntimeModule::import(const char *model_ptr, const char *backpropagation_model_ptr, const OMConfig &config)
{
  assert(model_ptr != nullptr and backpropagation_model_ptr != nullptr && "Model ptr shouldn't be nullptr");
  if (model_ptr == nullptr or backpropagation_model_ptr == nullptr)
    return UnknownError;

  // Initialize training storage
  {
    reader::OMCircleReader reader;
    reader.parse(backpropagation_model_ptr);
    reader.select_subgraph(0);
    _training_storage.initTrainingStorage(&reader, config);
  }

  OMStatus status = importMainModel(model_ptr, config);
  assert(status == Ok);
  if (status != Ok)
    return status;

  status = importBackpropagationModel(backpropagation_model_ptr, config);
  assert(status == Ok);
  if (status != Ok)
    return status;

  return Ok;
}

// TODO: remove Code Duplication
OMStatus OMTrainingRuntimeModule::importMainModel(const char *model_ptr, const OMConfig &config)
{
  // 1 - Parse reader for model
  // 2 - Load default graph for model
  // 3 - Optimize graphs
  // 4 - AllocDeallocPlan creation for model
  // 5 - KernelConfigure for model

  OMStatus status;
  // 1 - Parse reader for main model and backprop model
  // Main graph
  uint32_t num_subgraph = getGraphsCount(model_ptr);

  if (num_subgraph == 0)
    return UnknownError;

  // Resize graphs
  _main_runtime_graphs.resize(num_subgraph);

  // Get tensors indexes for main graph, that should be saved during training
  std::unordered_set<uint16_t> saved_tensors_indexes;
  {
    auto &mapping_table = _training_storage.getBackpropIndexesToMainIndexesTable();
    for (const auto &map_pair : mapping_table)
    {
      saved_tensors_indexes.insert(map_pair.second);
    }
  }

  for (uint32_t i = 0; i < num_subgraph; ++i)
  {
    // 2 - load default graph
    OMRuntimeGraph &main_graph = _main_runtime_graphs.at(i);

    OMRuntimeContext &runtime_context = main_graph.getRuntimeContext();
    OMRuntimeStorage &runtime_storage = main_graph.getRuntimeStorage();
    memory::OMRuntimeAllocator &runtime_allocator = main_graph.getRuntimeAllocator();

    runtime_context.setModel(model_ptr, i);

    // Parse and validate WOF file if it is exist
    // WARNING: setWofFile method of RuntimeContext should follow after setModel.
    if (config.wof_ptr != nullptr)
      runtime_context.setWofFile(config.wof_ptr);

    // 3 - Optimize it until can
    status = optimize::OMOptimizer::optimize(runtime_storage, runtime_context, config);
    if (status != Ok)
      return status;

    // 4 - AllocDeallocPlan creation
    status = import::OMExecutionPlanCreator::createExecutionPlan(runtime_storage, runtime_context,
                                                                 runtime_allocator, config, saved_tensors_indexes);
    if (status != Ok)
      return status;

    // 5 - KernelConfigure
    import::OMConfigureArgs configure_args = {runtime_storage, runtime_context, 0, config, nullptr};

    status = import::OMKernelConfiguration::configureKernels(configure_args);
    if (status != Ok)
      return status;

    // Done!
  }
  return Ok;
}

// TODO: remove Code Duplication
OMStatus OMTrainingRuntimeModule::importBackpropagationModel(const char *backpropagation_model_ptr, const OMConfig &config)
{
  // 1 - Parse reader for model
  // 2 - Load default graph for model
  // 3 - Optimize graphs
  // 4 - AllocDeallocPlan creation for model
  // 5 - KernelConfigure for model

  OMStatus status;
  // 1 - Parse reader for main model and backprop model
  // Main graph
  uint32_t num_subgraph = getGraphsCount(backpropagation_model_ptr);

  if (num_subgraph == 0)
    return UnknownError;

  // Resize graphs
  _backpropagation_runtime_graphs.resize(num_subgraph);

  for (uint32_t i = 0; i < num_subgraph; ++i)
  {
    // 2 - load default graph
    OMRuntimeGraph &main_graph = _backpropagation_runtime_graphs.at(i);

    OMRuntimeContext &runtime_context = main_graph.getRuntimeContext();
    OMRuntimeStorage &runtime_storage = main_graph.getRuntimeStorage();
    memory::OMRuntimeAllocator &runtime_allocator = main_graph.getRuntimeAllocator();

    runtime_context.setModel(backpropagation_model_ptr, i);

//    // Parse and validate WOF file if it is exist
//    // WARNING: setWofFile method of RuntimeContext should follow after setModel.
//    if (config.wof_ptr != nullptr)
//      runtime_context.setWofFile(config.wof_ptr);

    // 3 - Optimize it until can
    status = optimize::OMOptimizer::optimize(runtime_storage, runtime_context, config);
    if (status != Ok)
      return status;

    // 4 - AllocDeallocPlan creation
    status = import::OMExecutionPlanCreator::createExecutionPlan(runtime_storage, runtime_context,
                                                                 runtime_allocator, config);
    if (status != Ok)
      return status;

    // 5 - KernelConfigure
    import::OMConfigureArgs configure_args = {runtime_storage, runtime_context, 0, config, nullptr};

    status = import::OMKernelConfiguration::configureKernels(configure_args);
    if (status != Ok)
      return status;

    // Done!
  }
  return Ok;
}

OMStatus OMTrainingRuntimeModule::allocateInputs()
{
  assert(_main_runtime_graphs.size() > 0);
  if (_main_runtime_graphs.size() == 0)
    return ModelNotImport;
  return _main_runtime_graphs.at(0).allocateGraphInputs();
}

OMStatus OMTrainingRuntimeModule::allocateTargets()
{
  auto &targets_indexes = _training_storage.getTargetsIndexes();

  assert(_backpropagation_runtime_graphs.size() > 0);
  if (_main_runtime_graphs.size() == 0)
    return ModelNotImport;

  auto &storage = _backpropagation_runtime_graphs.at(0).getRuntimeStorage();
  auto &context = _backpropagation_runtime_graphs.at(0).getRuntimeContext();
  auto &allocator = _backpropagation_runtime_graphs.at(0).getRuntimeAllocator();

  allocator.allocateGraphInputs(&context, &storage, targets_indexes);

  return Ok;
}

OMStatus OMTrainingRuntimeModule::forward()
{
  OMStatus status = Ok;

  if (_main_runtime_graphs.empty())
    return ModelNotImport;

  core::OMRuntimeGraph &main_graph = _main_runtime_graphs.at(0);

  execute::OMExecuteArgs execute_args = {main_graph.getRuntimeStorage(),
                                         main_graph.getRuntimeContext(), 0, nullptr};

  status = execute::OMKernelExecute::executeKernel(execute_args, main_graph.getRuntimeAllocator());
  if (status != Ok)
    return status;

  return status;
}

OMStatus OMTrainingRuntimeModule::backward()
{
  OMStatus status = Ok;

  if (_backpropagation_runtime_graphs.empty() or _main_runtime_graphs.empty())
    return ModelNotImport;

  // Move tensors data from main to backprop graph
  {
    auto &main_storage = _main_runtime_graphs.at(0).getRuntimeStorage();
    auto &backprop_storage = _backpropagation_runtime_graphs.at(0).getRuntimeStorage();

    auto &main_tensor_index_to_data = main_storage.getTensorIndexToData();
    auto &backprop_tensor_index_to_data = backprop_storage.getTensorIndexToData();

    auto tensors_indexes = _training_storage.getBackpropIndexesToMainIndexesTable();
    for (const auto &map_pair : tensors_indexes)
    {
      //assert(main_tensor_index_to_data.find(map_pair.second) != main_tensor_index_to_data.end());
      if (main_tensor_index_to_data.find(map_pair.second) == main_tensor_index_to_data.end())
        continue;
      auto *data = main_tensor_index_to_data.at(map_pair.second);

      assert(data != nullptr);

      assert(backprop_tensor_index_to_data.find(map_pair.first) ==
             backprop_tensor_index_to_data.end());
      backprop_tensor_index_to_data[map_pair.first] = data;

      main_storage.removeTensorFromTensorIndexToData(map_pair.second);
    }
  }

  core::OMRuntimeGraph &backprop_graph = _backpropagation_runtime_graphs.at(0);

  execute::OMExecuteArgs execute_args = {backprop_graph.getRuntimeStorage(),
                                         backprop_graph.getRuntimeContext(), 0, nullptr};

  status =
    execute::OMKernelExecute::executeKernel(execute_args, backprop_graph.getRuntimeAllocator());
  if (status != Ok)
    return status;

  return status;
}

template <typename T>
void OMTrainingRuntimeModule::updateSGDWeights(uint8_t *dest, uint8_t *src, size_t size)
{
  assert(dest != nullptr); // Check caller
  assert(src != nullptr);  // Check caller
  assert(size > 0); // Check caller

  T *dest_f = reinterpret_cast<T *>(dest);
  T *src_f = reinterpret_cast<T *>(src);

  auto lamda = _training_storage.getLambda();

  for (size_t s = 0; s < size; s++)
  {
    dest_f[s] -= lamda * src_f[s];
  }
}

template <typename T>
void OMTrainingRuntimeModule::updateRMSPropWeights(uint8_t *dest, uint8_t *src, size_t size, uint16_t tensor_index)
{
  assert(dest != nullptr); // Check caller
  assert(src != nullptr);  // Check caller
  assert(size > 0); // Check caller

  T *dest_f = reinterpret_cast<T *>(dest);
  T *src_f = reinterpret_cast<T *>(src);

  auto lamda = _training_storage.getLambda();
  auto beta_squares = _training_storage.getBetaSquares();
  auto epsilon = _training_storage.getEpsilon();

  T *exp_avg_squares = reinterpret_cast<T *>(_training_storage.getExponentAvgSquaresData(tensor_index));
  assert(exp_avg_squares != nullptr);

  for (size_t s = 0; s < size; s++)
  {
    exp_avg_squares[s] = beta_squares * exp_avg_squares[s] + (1 - beta_squares) * std::pow(src_f[s], 2);
    dest_f[s] -= lamda * (src_f[s] / (std::sqrt(exp_avg_squares[s] + epsilon)));
  }
}

template <typename T>
void OMTrainingRuntimeModule::updateADAMWeights(uint8_t *dest, uint8_t *src, size_t size, uint16_t tensor_index)
{
  assert(dest != nullptr); // Check caller
  assert(src != nullptr);  // Check caller
  assert(size > 0); // Check caller

  T *dest_f = reinterpret_cast<T *>(dest);
  T *src_f = reinterpret_cast<T *>(src);

  auto lamda = _training_storage.getLambda();
  auto beta = _training_storage.getBeta();
  auto beta_squares = _training_storage.getBetaSquares();
  auto epsilon = _training_storage.getEpsilon();
  int32_t adam_step = _training_storage.getAdamStep();

  // Add 1 step
  adam_step++;

  T *exp_avg_squares = reinterpret_cast<T *>(_training_storage.getExponentAvgSquaresData(tensor_index));
  assert(exp_avg_squares != nullptr);

  T *exp_avg = reinterpret_cast<T *>(_training_storage.getExponentAvgData(tensor_index));
  assert(exp_avg != nullptr);

  for (size_t s = 0; s < size; s++)
  {
    exp_avg[s] = beta * exp_avg[s] + (1 - beta) * src_f[s];
    exp_avg_squares[s] = beta_squares * exp_avg_squares[s] + (1 - beta_squares) * std::pow(src_f[s], 2);

    auto exp_avg_corrected = exp_avg[s] / (1.f - std::pow(beta, adam_step));
    auto exp_avg_squares_corrected = exp_avg_squares[s] / (1.f - std::pow(beta_squares, adam_step));

    dest_f[s] -= lamda * (exp_avg_corrected / (std::sqrt(exp_avg_squares_corrected) + epsilon));
  }
}

OMStatus OMTrainingRuntimeModule::reset()
{
  OMStatus status = Ok;

  if (_main_runtime_graphs.empty())
    return ModelNotImport;

  if (_backpropagation_runtime_graphs.empty())
    return ModelNotImport;

  for (auto &graph : _main_runtime_graphs)
  {
    graph.reset();
  }

  for (auto &graph : _backpropagation_runtime_graphs)
  {
    graph.reset();
  }

  return status;
}

OMStatus OMTrainingRuntimeModule::updateWeights()
{
  auto &tensors_indexes = _training_storage.getBackpropIndexesToMainIndexesTable();
  auto outputs_tensor_indexes = _backpropagation_runtime_graphs.at(0).getRuntimeContext().getCircleOutputs();

  auto &backpop_storage = _backpropagation_runtime_graphs.at(0).getRuntimeStorage();
  auto &main_graph_context = _main_runtime_graphs.at(0).getRuntimeContext();

  auto optimization_strategy = _training_storage.getOptimizationStrategy();
  for (uint16_t i = 0; i < outputs_tensor_indexes->size(); ++i)
  {
    auto output_index = outputs_tensor_indexes->operator[](i);
    auto origin_index = tensors_indexes.at(output_index);

    uint8_t *gradients_data = nullptr;
    OMStatus status = backpop_storage.getDataByTensorIndex(&gradients_data, output_index);

    assert(gradients_data != nullptr); // Check data is calculated

    uint8_t *weight_data;
    status = main_graph_context.getConstDataByTensorIndex(&weight_data, origin_index);

    const auto output_size = _backpropagation_runtime_graphs.at(0).getOutputSizeAt(i);

    switch (optimization_strategy)
    {
      case SGD:
      {
        updateSGDWeights<float>(weight_data, gradients_data, output_size);
        break;
      }
      case RMSProp:
      {
        updateRMSPropWeights<float>(weight_data, gradients_data, output_size, output_index);
        break;
      }
      case ADAM:
      {
        updateADAMWeights<float>(weight_data, gradients_data, output_size, output_index);
        break;
      }
      default:
      {
        assert(false && "Unsuppoprted optimization strategy");
        return UnsupportedType;
      }
    }
  }

  return Ok;
}
