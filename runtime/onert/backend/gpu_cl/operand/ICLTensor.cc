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

#include "open_cl/Api.h"
#include "open_cl/Spi.h"
#include "open_cl/OpenclWrapper.h"
#include "open_cl/TensorTypeUtil.h"
#include "open_cl/kernels/Converter.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{
namespace operand
{

void ICLTensor::access(const std::function<void(ITensor &tensor)> &fn)
{
  if (total_size() == 0)
    return;

  fn(*this);
}

void ICLTensor::enqueueWriteBuffer(const void *ptr, bool)
{
  const float *arr = (float *)ptr;
  TensorObject input_obj = MakeReadableCpuMemory(absl::MakeSpan(arr, total_size() / 4));

  TensorObject output_obj;

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

  TensorObjectDef input_def;
  input_def.dimensions.b = handle()->Batch();
  input_def.dimensions.h = handle()->Height();
  input_def.dimensions.w = handle()->Width();
  input_def.dimensions.c = handle()->Channels();
  input_def.object_def.data_layout = DataLayout::BHWC;
  input_def.object_def.data_type = DataType::FLOAT32;
  input_def.object_def.object_type = ObjectType::CPU_MEMORY;
  input_def.object_def.user_provided = true;

  TensorObjectDef tmp_def;
  tmp_def.dimensions.b = handle()->Batch();
  tmp_def.dimensions.h = handle()->Height();
  tmp_def.dimensions.w = handle()->Width();
  tmp_def.dimensions.c = handle()->Channels();
  tmp_def.object_def.data_layout = DataLayout::BHWC;
  tmp_def.object_def.data_type = DataType::FLOAT32;
  tmp_def.object_def.object_type = ToObjectType(handle()->GetStorageType());
  tmp_def.object_def.user_provided = true;

  auto dims = tmp_def.dimensions;
  const BHWC shape(dims.b, dims.h, dims.w, dims.c);
  const TensorDescriptor desc{
    tmp_def.object_def.data_type,
    ToTensorStorageType(tmp_def.object_def.object_type, tmp_def.object_def.data_layout),
    Layout::BHWC};
  if (!AllocateTensorMemory(_environment->context(), shape, desc, &_cl_memory).ok())
  {
    throw std::runtime_error("AllocateTensorMemory error.");
  }
  TensorObject tmp_obj;
  if (tmp_def.object_def.object_type == ObjectType::OPENCL_TEXTURE)
  {
    tmp_obj = OpenClTexture{_cl_memory.memory()};
  }
  else
  {
    tmp_obj = OpenClBuffer{_cl_memory.memory()};
  }

  TensorObjectDef output_def = input_def;
  output_def.dimensions.b = handle()->Batch();
  output_def.dimensions.h = handle()->Height();
  output_def.dimensions.w = handle()->Width();
  output_def.dimensions.c = handle()->Channels();
  output_def.object_def.data_layout = ToDataLayout(handle()->GetStorageType());
  output_def.object_def.data_type = handle()->GetDataType();
  output_def.object_def.object_type = ToObjectType(handle()->GetStorageType());

  _converter_builder = NewConverterBuilder(_environment.get());
  if (!_converter_builder->MakeConverter(input_def, tmp_def, &_converter_cpu).ok())
  {
    throw std::runtime_error("MakeConverter<_converter_cpu> error.");
  }
  if (!_converter_builder->MakeConverter(tmp_def, output_def, &_converter_bhwc).ok())
  {
    throw std::runtime_error("MakeConverter<_converter_bhwc> error.");
  }

  if (!_converter_cpu->Convert(input_obj, tmp_obj).ok())
  {
    throw std::runtime_error("[w] _converter_cpu Convert error.");
  }
  if (!_converter_bhwc->Convert(tmp_obj, output_obj).ok())
  {
    throw std::runtime_error("[w] _converter_bhwc Convert error.");
  }
}

void ICLTensor::enqueueReadBuffer(void *ptr, bool)
{
  float *arr = (float *)ptr;
  TensorObject output_obj = MakeCpuMemory(absl::MakeSpan(arr, total_size() / 4));

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

  TensorObjectDef input_def;
  input_def.dimensions.b = handle()->Batch();
  input_def.dimensions.h = handle()->Height();
  input_def.dimensions.w = handle()->Width();
  input_def.dimensions.c = handle()->Channels();
  input_def.object_def.data_layout = ToDataLayout(handle()->GetStorageType());
  input_def.object_def.data_type = handle()->GetDataType();
  input_def.object_def.object_type = ToObjectType(handle()->GetStorageType());
  input_def.object_def.user_provided = false;

  TensorObjectDef tmp_def;
  tmp_def.dimensions.b = handle()->Batch();
  tmp_def.dimensions.h = handle()->Height();
  tmp_def.dimensions.w = handle()->Width();
  tmp_def.dimensions.c = handle()->Channels();
  tmp_def.object_def.data_layout = DataLayout::BHWC;
  tmp_def.object_def.data_type = DataType::FLOAT32;
  tmp_def.object_def.object_type = ToObjectType(handle()->GetStorageType());
  tmp_def.object_def.user_provided = true;

  auto dims = tmp_def.dimensions;
  const BHWC shape(dims.b, dims.h, dims.w, dims.c);
  const TensorDescriptor desc{
    tmp_def.object_def.data_type,
    ToTensorStorageType(tmp_def.object_def.object_type, tmp_def.object_def.data_layout),
    Layout::BHWC};
  if (!AllocateTensorMemory(_environment->context(), shape, desc, &_cl_memory).ok())
  {
    throw std::runtime_error("AllocateTensorMemory error.");
  }
  TensorObject tmp_obj;
  if (tmp_def.object_def.object_type == ObjectType::OPENCL_TEXTURE)
  {
    tmp_obj = OpenClTexture{_cl_memory.memory()};
  }
  else
  {
    tmp_obj = OpenClBuffer{_cl_memory.memory()};
  }
  TensorObjectDef output_def = input_def;
  output_def.dimensions.b = handle()->Batch();
  output_def.dimensions.h = handle()->Height();
  output_def.dimensions.w = handle()->Width();
  output_def.dimensions.c = handle()->Channels();
  output_def.object_def.data_layout = DataLayout::BHWC;
  output_def.object_def.data_type = DataType::FLOAT32;
  output_def.object_def.object_type = ObjectType::CPU_MEMORY;
  output_def.object_def.user_provided = true;

  _converter_builder = NewConverterBuilder(_environment.get());
  if (!_converter_builder->MakeConverter(input_def, tmp_def, &_converter_bhwc).ok())
  {
    throw std::runtime_error("MakeConverter<_converter_bhwc> error.");
  }
  if (!_converter_builder->MakeConverter(tmp_def, output_def, &_converter_cpu).ok())
  {
    throw std::runtime_error("MakeConverter<_converter_cpu> error.");
  }

  if (!_converter_bhwc->Convert(input_obj, tmp_obj).ok())
  {
    throw std::runtime_error("[r] _converter_bhwc Convert error.");
  }
  if (!_converter_cpu->Convert(tmp_obj, output_obj).ok())
  {
    throw std::runtime_error("[r] _converter_cpu Convert error.");
  }
}

} // namespace operand
} // namespace gpu_cl
} // namespace backend
} // namespace onert
