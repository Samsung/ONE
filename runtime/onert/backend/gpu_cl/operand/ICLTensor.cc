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

#include "ICLTensor.h"

#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/spi.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type_util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/converter.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{
namespace operand
{

using namespace tflite::gpu;
using namespace tflite::gpu::cl;
using namespace tflite::gpu::internal_tensor;

void ICLTensor::access(const std::function<void(ITensor &tensor)> &fn)
{
  if (total_size() == 0)
    return;

  fn(*this);
}

void ICLTensor::writeConvertInit(tflite::gpu::TensorObjectConverterBuilder *converter_builder,
                                 std::shared_ptr<tflite::gpu::cl::Environment> environment)
{
  _environment = environment;
  TensorObjectDef input_def;
  input_def.dimensions.b = handle()->Batch();
  input_def.dimensions.h = handle()->Height();
  input_def.dimensions.w = handle()->Width();
  input_def.dimensions.c = handle()->Channels();
  input_def.object_def.data_layout = DataLayout::BHWC;
  input_def.object_def.data_type = DataType::FLOAT32;
  input_def.object_def.object_type = ObjectType::CPU_MEMORY;
  input_def.object_def.user_provided = true;

  TensorObjectDef permute_def = input_def;
  permute_def.object_def.object_type = ToObjectType(handle()->GetStorageType());

  const auto &dims = permute_def.dimensions;
  const BHWC shape(dims.b, dims.h, dims.w, dims.c);
  const TensorDescriptor desc{
    permute_def.object_def.data_type,
    ToTensorStorageType(permute_def.object_def.object_type, permute_def.object_def.data_layout),
    Layout::BHWC};
  if (!AllocateTensorMemory(_environment->context(), shape, desc, &_cl_memory).ok())
  {
    throw std::runtime_error("Failed to AllocateTensorMemory");
  }

  TensorObjectDef output_def = permute_def;
  output_def.object_def.data_layout = ToDataLayout(handle()->GetStorageType());
  output_def.object_def.data_type = handle()->GetDataType();
  input_def.object_def.user_provided = false;

  if (!converter_builder->MakeConverter(input_def, permute_def, &_converter_to).ok())
  {
    throw std::runtime_error("Failed to make converter_to");
  }
  if (!converter_builder->MakeConverter(permute_def, output_def, &_converter_from).ok())
  {
    throw std::runtime_error("Failed to make converter_from");
  }
}

void ICLTensor::readConvertInit(tflite::gpu::TensorObjectConverterBuilder *converter_builder,
                                std::shared_ptr<tflite::gpu::cl::Environment> environment)
{
  _environment = environment;
  TensorObjectDef input_def;
  input_def.dimensions.b = handle()->Batch();
  input_def.dimensions.h = handle()->Height();
  input_def.dimensions.w = handle()->Width();
  input_def.dimensions.c = handle()->Channels();
  input_def.object_def.data_layout = ToDataLayout(handle()->GetStorageType());
  input_def.object_def.data_type = handle()->GetDataType();
  input_def.object_def.object_type = ToObjectType(handle()->GetStorageType());
  input_def.object_def.user_provided = false;

  TensorObjectDef permute_def = input_def;
  permute_def.object_def.data_layout = DataLayout::BHWC;
  permute_def.object_def.data_type = DataType::FLOAT32;
  permute_def.object_def.user_provided = true;

  const auto &dims = permute_def.dimensions;
  const BHWC shape(dims.b, dims.h, dims.w, dims.c);
  const TensorDescriptor desc{
    permute_def.object_def.data_type,
    ToTensorStorageType(permute_def.object_def.object_type, permute_def.object_def.data_layout),
    Layout::BHWC};
  if (!AllocateTensorMemory(_environment->context(), shape, desc, &_cl_memory).ok())
  {
    throw std::runtime_error("Failed to AllocateTensorMemory");
  }

  TensorObjectDef output_def = permute_def;
  output_def.object_def.object_type = ObjectType::CPU_MEMORY;

  if (!converter_builder->MakeConverter(input_def, permute_def, &_converter_from).ok())
  {
    throw std::runtime_error("Failed to make converter_from");
  }
  if (!converter_builder->MakeConverter(permute_def, output_def, &_converter_to).ok())
  {
    throw std::runtime_error("Failed to make converter_to");
  }
}

void ICLTensor::enqueueWriteBuffer(const void *ptr, bool blocking)
{
  TensorObject input_obj = MakeReadableCpuMemory(
    absl::MakeSpan(static_cast<const float *>(ptr), _info._shape.DimensionsProduct()));

  TensorObject output_obj;

  TensorObject permute_obj;
  if (ToObjectType(handle()->GetStorageType()) == ObjectType::OPENCL_TEXTURE)
  {
    permute_obj = OpenClTexture{_cl_memory.memory()};
  }
  else
  {
    permute_obj = OpenClBuffer{_cl_memory.memory()};
  }

  if (handle()->GetStorageType() == TensorStorageType::BUFFER)
  {
    output_obj = OpenClBuffer{handle()->GetMemoryPtr()};
  }
  else if (handle()->GetStorageType() == TensorStorageType::IMAGE_BUFFER)
  {
    output_obj = OpenClBuffer{handle()->GetMemoryPtrForWriting()};
  }
  else
  {
    output_obj = OpenClTexture{handle()->GetMemoryPtr()};
  }

  if (!_converter_to->Convert(input_obj, permute_obj).ok())
  {
    throw std::runtime_error("Failed to write cl buffer from cpu memory");
  }

  if (blocking && !_environment->queue()->WaitForCompletion().ok())
  {
    throw std::runtime_error("Failed to WaitForCompletion");
  }

  if (!_converter_from->Convert(permute_obj, output_obj).ok())
  {
    throw std::runtime_error("Failed to change layout");
  }
}

void ICLTensor::enqueueReadBuffer(void *ptr, bool blocking)
{
  TensorObject input_obj;

  if (handle()->GetStorageType() == TensorStorageType::BUFFER)
  {
    input_obj = OpenClBuffer{handle()->GetMemoryPtr()};
  }
  else if (handle()->GetStorageType() == TensorStorageType::IMAGE_BUFFER)
  {
    input_obj = OpenClBuffer{handle()->GetMemoryPtrForWriting()};
  }
  else
  {
    input_obj = OpenClTexture{handle()->GetMemoryPtr()};
  }

  TensorObject permute_obj;
  if (ToObjectType(handle()->GetStorageType()) == ObjectType::OPENCL_TEXTURE)
  {
    permute_obj = OpenClTexture{_cl_memory.memory()};
  }
  else
  {
    permute_obj = OpenClBuffer{_cl_memory.memory()};
  }

  TensorObject output_obj =
    MakeCpuMemory(absl::MakeSpan(static_cast<float *>(ptr), _info._shape.DimensionsProduct()));

  if (!_converter_from->Convert(input_obj, permute_obj).ok())
  {
    throw std::runtime_error("Failed to change layout");
  }
  if (!_converter_to->Convert(permute_obj, output_obj).ok())
  {
    throw std::runtime_error("Failed to read cl buffer");
  }

  if (blocking && !_environment->queue()->WaitForCompletion().ok())
  {
    throw std::runtime_error("Failed to WaitForCompletion");
  }
}

} // namespace operand
} // namespace gpu_cl
} // namespace backend
} // namespace onert
