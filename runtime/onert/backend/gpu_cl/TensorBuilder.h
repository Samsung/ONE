/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_GPU_CL_TENSOR_BUILDER_H__
#define __ONERT_BACKEND_GPU_CL_TENSOR_BUILDER_H__

#include "TensorManager.h"

#include <cl_common/LifetimeMap.h>
#include <cl_common/ParentInfo.h>

#include <ir/Operands.h>
#include <ir/OperandIndexSequence.h>

namespace onert
{
namespace backend
{
namespace gpu_cl
{
class TensorBuilder
{
public:
  TensorBuilder(const ir::Operands &operands, TensorManager *tensor_mgr);

  /**
   * @brief     Register tensor information to allocate on ACL-CL backend
   * @param[in] ind    Operand index
   * @param[in] info   Tensor information
   * @param[in] type   Tensor type
   */
  void registerTensorInfo(const ir::OperandIndex &ind, const ir::OperandInfo &info,
                          TensorType type);

  void notifyFirstUse(const ir::OperandIndex &);
  void notifyLastUse(const ir::OperandIndex &);

  bool isRegistered(const ir::OperandIndex &) const;

  void prepare();
  void allocate();
  void postFunctionPrepare();

  TensorManager *cl_tensor_manager(void) { return _tensor_mgr.get(); }

  void setUsesCount(const ir::OperandIndex &index, size_t num_uses)
  {
    assert(_uses_count_map.find(index) != _uses_count_map.end() ? _uses_count_map[index] == num_uses
                                                                : true);
    _uses_count_map[index] = num_uses;
  }

  void parent_map(std::unordered_map<ir::OperandIndex, cl_common::ParentInfo> &&parent_map)
  {
    _parent_map = std::move(parent_map);
  }

  bool areSubTensorsOf(const ir::OperandIndex &parent, const ir::OperandIndexSequence &seq);

  /**
   * @brief     Check child tensor is allocated as subtensor of parent tensor
   * @param[in] parent  Index of parent
   * @param[in] child   Index of child
   * @return    @c true if child is allocated as subtensor of parent, otherwise @c false
   */
  bool isSubTensorOf(const ir::OperandIndex &parent, const ir::OperandIndex &child);

private:
  void buildTensors(void);
  ir::OperandIndex findRootParent(ir::OperandIndex index);
  ir::OperandIndex addTensor(const ir::Shape &shape);

private:
  const ir::Operands &_operands;
  ir::OperandIndexMap<ir::OperandInfo> _tensor_info_map;
  ir::OperandIndexMap<ir::Layout> _tensor_layout_map;
  ir::OperandIndexMap<TensorType> _tensor_type_map;
  ir::OperandIndexMap<size_t> _uses_count_map;

  std::unique_ptr<TensorManager> _tensor_mgr;

  // for linear executor
  cl_common::LifetimeSeq _lifetime_seq;

  // Extra info for concat elimination
  ir::OperandIndexMap<cl_common::ParentInfo> _parent_map;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_TENSOR_BUILDER_H__
