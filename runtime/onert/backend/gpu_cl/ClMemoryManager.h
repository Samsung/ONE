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

#ifndef __ONERT_BACKEND_ACL_COMMON_MEMORY_MANAGER_H__
#define __ONERT_BACKEND_ACL_COMMON_MEMORY_MANAGER_H__

#include <cassert>

#include "ir/OperandIndexMap.h"
#include "ir/Shape.h"
#include "open_cl/ClContext.h"
#include "open_cl/InferenceContext.h"
#include "open_cl/Status.h"
#include "open_cl/StorageTypeUtil.h"
#include "open_cl/TensorType.h"
#include "util/logging.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

template <typename T_ITensor, typename T_Tensor> class ClMemoryManager
{
public:
  ClMemoryManager(CLContext *context) : _context{context} {}

  virtual ~ClMemoryManager() = default;

  virtual void allocate(void)
  {
    for (const auto &tensor_entry : _tensors)
    {
      auto tensor = tensor_entry.second;
      const auto &t = tensor_reserver_.Get(tensor_entry.first.value());
      const auto &shape = t.shape;
      const auto &descriptor = t.descriptor;
      if (!CreateTensor(*_context, shape, descriptor, tensor->handle()).ok())
      {
        return;
      }
    }
  }

  virtual void deallocate(void)
  {
    // NYI
  }

  virtual void startLifetime(const ir::OperandIndex &)
  { /* DO NOTHING */
  }
  virtual void finishLifetime(const ir::OperandIndex &)
  { /* DO NOTHING */
  }

  void buildTensor(const ir::OperandIndex &ind, const ir::OperandInfo &info, size_t num_use,
                   InferenceContext::CreateInferenceInfo create_info, CLCommandQueue *queue,
                   DeviceInfo &device_info)
  {
    ValueId max_id = 0;
    auto data_type = DeduceDataTypeFromPrecision(create_info.precision);
    const auto shape = info.shape();

    auto tensor = std::make_shared<T_Tensor>(shape.rank(), shape, queue, num_use);
    _tensors[ind] = tensor;

    BHWC t_shape;
    switch (shape.rank())
    {
      case 1:
        // B layout
        t_shape = BHWC(shape.dim(0), 1, 1, 1);
        break;
      case 2:
        // BC layout
        t_shape = BHWC(shape.dim(0), 1, 1, shape.dim(1));
        break;
      case 3:
        // BWC layout
        t_shape = BHWC(shape.dim(0), 1, shape.dim(1), shape.dim(2));
        break;
      case 4:
        // BHWC layout
        t_shape = BHWC(shape.dim(0), shape.dim(1), shape.dim(2), shape.dim(3));
        break;
      default:
        break;
    }

    TensorStorageType storage_type = create_info.storage_type;
    Layout layout = t_shape.b == 1 ? Layout::HWC : Layout::BHWC;

    ValueId id = ind.value();
    storage_type = SelectBestStorageType(device_info, t_shape, storage_type, data_type, layout);
    tensor_reserver_.Add(id, {t_shape, TensorDescriptor{data_type, storage_type, layout}});

    max_id = std::max(max_id, id);

    tensor_reserver_.SetNext(max_id + 1);
  }

  ir::OperandIndexMap<std::shared_ptr<T_Tensor>> &tensors(void) { return _tensors; }

  InferenceContext::TensorReserver &tensorReservers(void) { return tensor_reserver_; }

private:
  ir::OperandIndexMap<std::shared_ptr<T_Tensor>> _tensors;
  InferenceContext::TensorReserver tensor_reserver_;
  CLContext *_context;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_COMMON_MEMORY_MANAGER_H__
