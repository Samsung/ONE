/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_CONV_WEIGHTS_CONVERTER_H__
#define __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_CONV_WEIGHTS_CONVERTER_H__

#include "open_cl/ClCommandQueue.h"
#include "open_cl/ClKernel.h"
#include "open_cl/kernels/ConvCommon.h"
#include "open_cl/kernels/GpuOperation.h"
#include "open_cl/Status.h"
#include "open_cl/Types.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

class ConverterToConvWeights : public GPUOperation
{
public:
  ConverterToConvWeights(const OperationDef &definition,
                         const ConvWeightsDescription &conv_weights_desc);
  absl::Status BindArguments(ArgumentsBinder *args) override;
  int3 GetGridSize() const override;

  // Move only
  ConverterToConvWeights(ConverterToConvWeights &&operation);
  ConverterToConvWeights &operator=(ConverterToConvWeights &&operation);
  ConverterToConvWeights(const ConverterToConvWeights &) = delete;
  ConverterToConvWeights &operator=(const ConverterToConvWeights &) = delete;

private:
  std::string GetConverterToConvWeightsCode(const OperationDef &op_def,
                                            const ConvWeightsDescription &conv_weights_desc);

  ConvWeightsDescription conv_weights_desc_;
};

// We expect src BHWC tensor and we assume that B is O, H = H, W = W, C is I
// as dst we expect Tensor with storage type BUFFER and
// dst.b * dst.h * dst.w * dst.c = AlignByN(src.b, 4) * src.h * src.w
// AlignByN(src.c, 4)
ConverterToConvWeights
CreateConverterToConvWeights(const OperationDef &definition,
                             const ConvWeightsDescription &conv_weights_desc);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_CONV_WEIGHTS_CONVERTER_H__
