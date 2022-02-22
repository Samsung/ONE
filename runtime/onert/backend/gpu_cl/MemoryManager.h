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

#ifndef __ONERT_BACKEND_GPU_CL_MEMORY_MANAGER_H__
#define __ONERT_BACKEND_GPU_CL_MEMORY_MANAGER_H__

#include "ex/InferenceContextEx.h"
#include "operand/CLTensor.h"

#include "ir/OperandIndexMap.h"
#include "ir/OperandInfo.h"
#include "util/logging.h"

#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/cl/storage_type_util.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"

#include <cassert>

namespace onert
{
namespace backend
{
namespace gpu_cl
{

class MemoryManager
{
public:
  MemoryManager(tflite::gpu::cl::CLContext *context) : _context{context} {}

  ~MemoryManager() = default;

  void allocate(void)
  {
    for (const auto &tensor_entry : _tensors)
    {
      auto tensor = tensor_entry.second;
      auto type = tensor->get_type();

      // if (type == TensorType::TENSOR_TYPE_DELETE) {
      //   continue;
      // }

      const auto &t = tensor_reserver_.Get(tensor_entry.first.value());
      const auto &shape = t->shape;
      const auto &descriptor = t->descriptor;
      if (!CreateTensor(*_context, shape, descriptor, tensor->handle()).ok())
      {
        std::runtime_error("Failed to CreateTensor");
      }
      switch (type)
      {
        case TensorType::TENSOR_TYPE_INPUT:
          tensor->writeConvertInit();
          break;
        case TensorType::TENSOR_TYPE_OUTPUT:
          tensor->readConvertInit();
          break;
        default:
          break;
      }
    }
  }

  void deallocate(void)
  {
    // NYI
  }

  void startLifetime(const ir::OperandIndex &)
  { /* DO NOTHING */
  }
  void finishLifetime(const ir::OperandIndex &)
  { /* DO NOTHING */
  }

  void buildTensor(const ir::OperandIndex &ind, const ir::OperandInfo &info,
                   tflite::gpu::cl::InferenceContext::CreateInferenceInfo create_info,
                   std::shared_ptr<tflite::gpu::cl::Environment> environment,
                   tflite::gpu::cl::DeviceInfo &device_info, TensorType type)
  {
    tflite::gpu::ValueId max_id = 0;
    auto data_type = DeduceDataTypeFromPrecision(create_info.precision);
    const auto shape = info.shape();

    auto tensor = std::make_shared<operand::CLTensor>(shape.rank(), shape, environment, type);
    _tensors[ind] = tensor;
    tflite::gpu::BHWC t_shape;
    switch (shape.rank())
    {
      case 1:
        // B layout
        t_shape = tflite::gpu::BHWC(shape.dim(0), 1, 1, 1);
        break;
      case 2:
        // BC layout
        t_shape = tflite::gpu::BHWC(shape.dim(0), 1, 1, shape.dim(1));
        break;
      case 3:
        // BWC layout
        t_shape = tflite::gpu::BHWC(shape.dim(0), 1, shape.dim(1), shape.dim(2));
        break;
      case 4:
        // BHWC layout
        t_shape = tflite::gpu::BHWC(shape.dim(0), shape.dim(1), shape.dim(2), shape.dim(3));
        break;
      default:
        break;
    }

    tflite::gpu::cl::TensorStorageType storage_type = create_info.storage_type;
    tflite::gpu::Layout layout =
      t_shape.b == 1 ? tflite::gpu::Layout::HWC : tflite::gpu::Layout::BHWC;

    tflite::gpu::ValueId id = ind.value();
    storage_type =
      tflite::gpu::cl::SelectBestStorageType(device_info, t_shape, storage_type, data_type, layout);
    auto dummy = std::make_shared<InferenceContextEx::DummyTensor>();
    dummy->shape = t_shape;
    dummy->descriptor = tflite::gpu::cl::TensorDescriptor{data_type, storage_type, layout};
    tensor_reserver_.Add(id, dummy);

    max_id = std::max(max_id, id);

    tensor_reserver_.SetNext(max_id + 1);
  }

  ir::OperandIndexMap<std::shared_ptr<operand::CLTensor>> &tensors(void) { return _tensors; }

  InferenceContextEx::TensorReserverEx &tensorReservers(void) { return tensor_reserver_; }

private:
  ir::OperandIndexMap<std::shared_ptr<operand::CLTensor>> _tensors;
  InferenceContextEx::TensorReserverEx tensor_reserver_;
  tflite::gpu::cl::CLContext *_context;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_MEMORY_MANAGER_H__
