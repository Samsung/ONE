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

#include "operand/CLTensor.h"

#include "ir/OperandIndexMap.h"
#include "ir/OperandInfo.h"
#include "util/logging.h"

#include "tensorflow/lite/delegates/gpu/spi.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/inference_context.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/storage_type_util.h"

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
  MemoryManager(tflite::gpu::cl::CLContext *context, tflite::gpu::CreateGpuModelInfo create_info,
                const std::shared_ptr<tflite::gpu::cl::Environment> &environment)
    : _context{context}, _create_info{create_info}, _environment{environment}
  {
  }

  ~MemoryManager() = default;

  void allocate(void)
  {
    std::unique_ptr<tflite::gpu::TensorObjectConverterBuilder> converter_builder =
      NewConverterBuilder(_environment.get());
    for (const auto &tensor_entry : _tensors)
    {
      auto tensor = tensor_entry.second;
      auto type = tensor->get_type();

      if (type == TensorType::TENSOR_TYPE_DELETE)
      {
        continue;
      }

      const auto &shape = tensor->get_info()._shape;
      const auto &descriptor = tensor->get_info()._desc;

      if (!CreateTensor(*_context, shape, descriptor, tensor->handle()).ok())
      {
        std::runtime_error("Failed to CreateTensor");
      }
      switch (type)
      {
        case TensorType::TENSOR_TYPE_INPUT:
          tensor->writeConvertInit(converter_builder.get(), _environment);
          break;
        case TensorType::TENSOR_TYPE_OUTPUT:
          tensor->readConvertInit(converter_builder.get(), _environment);
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

  void buildTensor(const ir::OperandIndex &ind, const ir::OperandInfo &info, TensorType type)
  {
    auto data_type = DeduceDataTypeFromPrecision(_create_info.precision);

    tflite::gpu::BHWC BHWC_shape = ToBHWC(info.shape());

    tflite::gpu::TensorStorageType storage_type = _create_info.storage_type;
    tflite::gpu::Layout layout =
      BHWC_shape.b == 1 ? tflite::gpu::Layout::HWC : tflite::gpu::Layout::BHWC;

    if (!SelectBestStorageType(_environment->device().GetInfo(), BHWC_shape, storage_type,
                               data_type, layout, &storage_type)
           .ok())
    {
      throw std::runtime_error("Failed to SelectBestStorageType");
    }
    auto tensor = std::make_shared<operand::CLTensor>(
      info.shape().rank(), type, BHWC_shape,
      tflite::gpu::TensorDescriptor{data_type, storage_type, layout});
    _tensors[ind] = tensor;
  }

  ir::OperandIndex addTensor(const ir::Shape &shape)
  {
    auto data_type = DeduceDataTypeFromPrecision(_create_info.precision);

    tflite::gpu::BHWC BHWC_shape = ToBHWC(shape);

    tflite::gpu::TensorStorageType storage_type = _create_info.storage_type;
    tflite::gpu::Layout layout =
      BHWC_shape.b == 1 ? tflite::gpu::Layout::HWC : tflite::gpu::Layout::BHWC;

    if (!SelectBestStorageType(_environment->device().GetInfo(), BHWC_shape, storage_type,
                               data_type, layout, &storage_type)
           .ok())
    {
      throw std::runtime_error("Failed to SelectBestStorageType");
    }
    auto ind = ir::OperandIndex(_new_id--);
    auto tensor = std::make_shared<operand::CLTensor>(
      shape.rank(), TensorType::TENSOR_TYPE_VALID, BHWC_shape,
      tflite::gpu::TensorDescriptor{data_type, storage_type, layout});
    _tensors[ind] = tensor;
    return ind;
  }

  ir::OperandIndexMap<std::shared_ptr<operand::CLTensor>> &tensors(void) { return _tensors; }

private:
  ir::OperandIndexMap<std::shared_ptr<operand::CLTensor>> _tensors;
  tflite::gpu::cl::CLContext *_context;
  tflite::gpu::CreateGpuModelInfo _create_info;
  std::shared_ptr<tflite::gpu::cl::Environment> _environment;
  uint32_t _new_id = UINT32_MAX;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_MEMORY_MANAGER_H__
