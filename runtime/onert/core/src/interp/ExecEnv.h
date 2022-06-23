/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file  ExecEnv.h
 * @brief This file contains ExecEnv to access interpreter tensor and execution status
 */
#ifndef __ONERT_INTERP_EXEC_ENV_H_
#define __ONERT_INTERP_EXEC_ENV_H_

#include <unordered_set>

#include "ir/Graph.h"
#include "Tensor.h"

namespace onert
{
namespace interp
{

/**
 * @brief Class to gather interpreter execution environment
 *        Each interpreter instance own execution environment
 */
class ExecEnv
{
public:
  /**
   * @brief Construct a new Exec Env object (deleted)
   */
  ExecEnv(void) = delete;
  /**
   * @brief Construct a new ExecEnv object
   * @param[in] graph Graph to execute by interpreter
   */
  explicit ExecEnv(const ir::Graph &graph) : _graph(graph)
  {
    // DO NOTHING
  }

public:
  /**
   * @brief   Return graph to execute
   * @return  Graph
   */
  const ir::Graph &graph(void) const { return _graph; }
  /**
   * @brief     Assign tensor to environment which have allocated or assigned buffer
   * @param[in] index   Tensor index
   * @param[in] tensor  Tensor
   */
  void assignTensor(const ir::OperandIndex index, std::shared_ptr<ITensor> tensor)
  {
    assert(tensor->bufferRO() != nullptr);
    _tensors.emplace(index, tensor);
  }

  /**
   * @brief     Return tensor pointer in environment
   * @param[in] index         Tensor index
   *            can_optional  @c True if tensor can be optional input, otherwise @c false
   * @return    Tensor pointer
   */
  const ITensor *tensorAt(const ir::OperandIndex index, bool can_optional = false) const
  {
    if (_tensors.find(index) == _tensors.end())
    {
      // It may optional input,
      // otherwise input is not set by runtime user
      if (can_optional)
      {
        return nullptr;
      }

      throw std::runtime_error{"ExecEnv: Input is not set"};
    }

    return _tensors.at(index).get();
  }

  /**
   * @brief     Check environment contains tensor
   * @param[in] index Tensor index
   * @return    @c true if environment contain tensor, otherwise @c false
   */
  bool contains(const ir::OperandIndex index) const
  {
    return (_tensors.find(index) != _tensors.end());
  }

  /**
   * @brief     Allocate tensor using operand info
   * @param[in] index     Tensor index
   * @param[in] info      Operand info
   * @note      If already allocated, just return
   * @TODO      More smart allocation policy
   */
  void allocateIfNeeded(const ir::OperandIndex index, const ir::OperandInfo &info)
  {
    // already allocated, or constant
    if (contains(index))
    {
      return;
    }

    // Buffer from external (ex. model output)
    auto tensor = std::make_shared<Tensor>(info);
    if (isExtBuffer(index))
    {
      tensor->setBuffer(_external_buffers.at(index));
      assignTensor(index, tensor);

      return;
    }

    tensor->setBuffer(std::make_shared<InternalBuffer>(tensor->total_size()));
    assignTensor(index, tensor);

    _buffers.insert(index);
  }

  /**
   * @brief     Allocate read-only tensor and share data with other tensor
   * @param[in] index           Tensor index
   * @param[in] info            Operand info
   * @param[in] index_to_share  Tensor index that have data to share
   */
  void allocateAndShareIfNeeded(const ir::OperandIndex index, const ir::OperandInfo &info,
                                const ir::OperandIndex index_to_share)
  {
    if (!contains(index_to_share))
    {
      throw std::runtime_error{"Cannot find tensor to share data"};
    }

    // already allocated
    if (contains(index))
    {
      return;
    }

    if (isExtBuffer(index))
    {
      auto tensor = std::make_shared<Tensor>(info);
      tensor->setBuffer(_external_buffers.at(index));
      assignTensor(index, tensor);
    }
    else
    {
      auto tensor = std::make_shared<ROTensor>(info);
      tensor->setData(tensorAt(index_to_share)->shareData());
      assignTensor(index, tensor);
      _buffers.insert(index);
    }
  }

  /**
   * @brief     Free buffer if allocated by allocateIfNeed
   * @param[in] index Tensor index
   * @note      If allocated by outside, just return
   */
  void freeIfAllocated(const ir::OperandIndex index)
  {
    if (_buffers.find(index) != _buffers.end())
    {
      _tensors.at(index)->releaseData();
    }
  }

  /**
   * @brief     Assign ExternalBuffer into external buffer map
   * @param[in] index   Tensor index
   * @param[in] buffer  External buffer
   */
  void assignExternalBuffer(const ir::OperandIndex index, std::shared_ptr<ExternalBuffer> buffer)
  {
    _external_buffers.emplace(index, buffer);
  }

private:
  bool isExtBuffer(const ir::OperandIndex index)
  {
    return (_external_buffers.find(index) != _external_buffers.end());
  }

private:
  const ir::Graph &_graph;
  // Tensor map to use in interpreter
  // It should map tensors that have allocated or assigned buffer pointer
  std::unordered_map<ir::OperandIndex, std::shared_ptr<ITensor>> _tensors;
  // Tensors allocated by allocateIfNeed (buffer)
  std::unordered_set<ir::OperandIndex> _buffers;
  // Tensor buffer from external
  std::unordered_map<ir::OperandIndex, std::shared_ptr<ExternalBuffer>> _external_buffers;
};

} // namespace interp
} // namespace onert

#endif // __ONERT_INTERP_EXEC_ENV_H_
