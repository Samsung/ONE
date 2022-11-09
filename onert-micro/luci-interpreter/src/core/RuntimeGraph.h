/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_CORE_RUNTIMEGRAPH_H
#define LUCI_INTERPRETER_CORE_RUNTIMEGRAPH_H

#include "luci_interpreter/core/Tensor.h"
#include "luci_interpreter/memory_managers/MemoryManager.h"
#include "core/Kernel.h"

#include <memory>
#include <vector>

namespace luci_interpreter
{

class RuntimeModule;

class BaseRuntimeGraph
{
public:
  explicit BaseRuntimeGraph(IMemoryManager *memory_manager);
  virtual ~BaseRuntimeGraph() = default;

  Tensor *addTensor(std::unique_ptr<Tensor> &&tensor);
  AffineQuantization *addAffineQuantization(std::unique_ptr<AffineQuantization> &&quantization);

  void addInputTensor(Tensor *input_tensor);
  void addOutputTensor(Tensor *output_tensor);

  void configureAllocations(Tensor *tensor);

  const std::vector<Tensor *> &getInputTensors() const { return _input_tensors; }
  const std::vector<Tensor *> &getOutputTensors() const { return _output_tensors; }

  void addKernel(std::unique_ptr<Kernel> &&kernel);

  virtual void execute() = 0;
  virtual void configure() = 0;

  void invalidate() { _is_valid = false; }
  bool isValid() const { return _is_valid; }

protected:
  IMemoryManager *_memory_manager;
  std::vector<std::unique_ptr<Tensor>> _tensors;
  std::vector<std::unique_ptr<AffineQuantization>> _affine_quantizations;
  std::vector<Tensor *> _input_tensors;
  std::vector<Tensor *> _output_tensors;

  bool _is_valid = false;

  // Kernels in execution order.
  std::vector<std::unique_ptr<Kernel>> _kernels;
};

class RuntimeGraph final : public BaseRuntimeGraph
{
public:
  explicit RuntimeGraph(IMemoryManager *memory_manager);
  ~RuntimeGraph() final;

  void execute() final;
  void configure() final;

private:
  void buildAllocDeallocPlan();
  void allocate(size_t kernel_index) const;
  void deallocate(size_t kernel_index) const;

private:
  // Tensors that are not used anymore after given op
  std::vector<std::vector<Tensor *>> _alloc_plan;
  std::vector<std::vector<Tensor *>> _dealloc_plan;
};

class StaticRuntimeGraph final : public BaseRuntimeGraph
{
public:
  explicit StaticRuntimeGraph(IMemoryManager *memory_manager);
  ~StaticRuntimeGraph() final;

  void execute() final;
  void configure() final;

  void configure_kernels();
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_CORE_RUNTIMEGRAPH_H
