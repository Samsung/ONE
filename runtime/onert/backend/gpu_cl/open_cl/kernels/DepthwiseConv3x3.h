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

#ifndef __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_DEPTHWISE_CONV_3X3_H__
#define __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_DEPTHWISE_CONV_3X3_H__

#include <memory>
#include <vector>

#include "open_cl/Buffer.h"
#include "open_cl/kernels/GpuOperation.h"
#include "open_cl/Tensor.h"
#include "open_cl/Texture2d.h"
#include "open_cl/Util.h"
#include "open_cl/DataType.h"
#include "open_cl/Operations.h"
#include "open_cl/Shape.h"
#include "open_cl/Status.h"
#include "open_cl/Tensor.h"
#include "open_cl/Types.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

class DepthwiseConv3x3 : public GPUOperation
{
public:
  DepthwiseConv3x3() = default;
  void GetPossibleKernelWorkGroups(TuningType tuning_type, const DeviceInfo &device_info,
                                   const KernelInfo &kernel_info,
                                   std::vector<int3> *work_groups) const override;
  int3 GetGridSize() const override;

  // Move only
  DepthwiseConv3x3(DepthwiseConv3x3 &&operation);
  DepthwiseConv3x3 &operator=(DepthwiseConv3x3 &&operation);
  DepthwiseConv3x3(const DepthwiseConv3x3 &) = delete;
  DepthwiseConv3x3 &operator=(const DepthwiseConv3x3 &) = delete;

private:
  explicit DepthwiseConv3x3(const OperationDef &definition, bool weights_are_buffer,
                            bool local_mem_uploads, const DeviceInfo &device_info);
  template <DataType T>
  void UploadWeightsAndBiases(const InternalTensor<OHWI, T> &weights,
                              const InternalTensor<Linear, T> &biases, bool weights_are_buffer);

  friend DepthwiseConv3x3 CreateDepthwiseConv3x3(const DeviceInfo &device_info,
                                                 const OperationDef &definition,
                                                 const DepthwiseConvolution2DAttributes &attr);

  template <DataType S, typename T>
  void RearrangeWeightsAndBiasesData(const InternalTensor<OHWI, S> &weights,
                                     const InternalTensor<Linear, S> &biases, absl::Span<T> dst);

  std::string GenerateDepthwiseConvCode(const OperationDef &op_def, bool weights_are_buffer,
                                        bool local_mem_uploads);

  bool local_mem_uploads_;
};

template <DataType T>
void DepthwiseConv3x3::UploadWeightsAndBiases(const InternalTensor<OHWI, T> &weights,
                                              const InternalTensor<Linear, T> &biases,
                                              bool weights_are_buffer)
{
  const int src_depth = DivideRoundUp(weights.shape.i, 4);
  int texture_width = 10; // 3x3 kernel + 1 bias
  int texture_height = src_depth;
  const int elements_count = texture_width * texture_height;
  const bool fp32_weights = definition_.precision == CalculationsPrecision::F32;
  const int float4_size = fp32_weights ? 16 : 8;

  std::vector<uint8_t> data(float4_size * elements_count);
  if (fp32_weights)
  {
    float4 *ptr = reinterpret_cast<float4 *>(data.data());
    RearrangeWeightsAndBiasesData(weights, biases, absl::MakeSpan(ptr, elements_count));
  }
  // TODO
  // It doesn't support F16 yet. I will try to add it later.
  //
  // else {
  //   half4* ptr = reinterpret_cast<half4*>(data.data());
  //   RearrangeWeightsAndBiasesData(weights, biases,
  //                                 absl::MakeSpan(ptr, elements_count));
  // }

  if (weights_are_buffer)
  {
    BufferDescriptor desc;
    desc.element_type = fp32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc.element_size = 4;
    desc.size = float4_size * elements_count;
    desc.data = std::move(data);
    args_.AddObject("weights", absl::make_unique<BufferDescriptor>(std::move(desc)));
  }
  else
  {
    Texture2DDescriptor desc;
    desc.element_type = fp32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc.size = int2(texture_width, texture_height);
    desc.data = std::move(data);
    args_.AddObject("weights", absl::make_unique<Texture2DDescriptor>(std::move(desc)));
  }
}

template <DataType S, typename T>
void DepthwiseConv3x3::RearrangeWeightsAndBiasesData(const InternalTensor<OHWI, S> &weights,
                                                     const InternalTensor<Linear, S> &biases,
                                                     absl::Span<T> dst)
{
  const int src_depth = DivideRoundUp(weights.shape.i, 4);

  int counter = 0;
  for (int s = 0; s < src_depth; ++s)
  {
    for (int y = 0; y < 3; ++y)
    {
      for (int x = 0; x < 3; ++x)
      {
        T filter_val;
        for (int i = 0; i < 4; ++i)
        {
          const int s_ch = s * 4 + i;
          if (s_ch < weights.shape.i)
          {
            const int f_index = weights.shape.LinearIndex({0, y, x, s_ch});
            filter_val[i] = weights.data[f_index];
          }
          else
          {
            filter_val[i] = 0.0f;
          }
        }
        dst[counter++] = filter_val;
      }
    }

    T bias_val;
    for (int i = 0; i < 4; ++i)
    {
      const int dst_ch = s * 4 + i;
      bias_val[i] = dst_ch >= biases.shape.v ? 0.0f : biases.data[dst_ch];
    }
    dst[counter++] = bias_val;
  }
}

bool IsDepthwiseConv3x3Supported(const DepthwiseConvolution2DAttributes &attr);

DepthwiseConv3x3 CreateDepthwiseConv3x3(const DeviceInfo &device_info,
                                        const OperationDef &definition,
                                        const DepthwiseConvolution2DAttributes &attr);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_DEPTHWISE_CONV_3X3_H__
