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

#ifndef __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_CONV_BUFFER_1X1_H__
#define __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_CONV_BUFFER_1X1_H__

#include "open_cl/Buffer.h"
#include "open_cl/ClKernel.h"
#include "open_cl/kernels/ConvCommon.h"
#include "open_cl/kernels/GpuOperation.h"
#include "open_cl/kernels/Util.h"
#include "open_cl/LinearStorage.h"
#include "open_cl/Precision.h"
#include "open_cl/InternalTensor.h"
#include "open_cl/Util.h"
#include "open_cl/DataType.h"
#include "open_cl/Operations.h"
#include "open_cl/Shape.h"
#include "open_cl/Status.h"
#include "open_cl/Tensor.h"
#include "open_cl/Types.h"
#include "open_cl/WinogradUtil.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

class ConvBuffer1x1 : public GPUOperation
{
public:
  ConvBuffer1x1() = default;

  // Move only
  ConvBuffer1x1(ConvBuffer1x1 &&operation);
  ConvBuffer1x1 &operator=(ConvBuffer1x1 &&operation);
  ConvBuffer1x1(const ConvBuffer1x1 &) = delete;
  ConvBuffer1x1 &operator=(const ConvBuffer1x1 &) = delete;

  void GetPossibleKernelWorkGroups(TuningType tuning_type, const DeviceInfo &device_info,
                                   const KernelInfo &kernel_info,
                                   std::vector<int3> *work_groups) const override;
  int3 GetGridSize() const override;

  ConvWeightsDescription GetConvWeightsDescription() const
  {
    ConvWeightsDescription desc;
    desc.layout = ConvWeightsLayout::kOHWIOGroupI4O4;
    desc.output_group_size = conv_params_.block_size.z;
    return desc;
  }

  struct ConvParams
  {
    int3 block_size = int3(1, 1, 1);
    int element_size = 4; // can be 4, 8 or 16

    // By default in 2d convolution we have the same weights for WH dims, but in
    // some cases we need separate weights for H dimension and convolution
    // kernel requires very small modifications to support it.
    bool different_weights_for_height = false;
  };

private:
  ConvBuffer1x1(const OperationDef &definition, const ConvParams &conv_params);
  friend ConvBuffer1x1 CreateConvBuffer1x1(const DeviceInfo &device_info,
                                           const OperationDef &definition,
                                           const Convolution2DAttributes &attr, const BHWC *shape);
  friend ConvBuffer1x1 CreateConvBuffer1x1(const DeviceInfo &device_info,
                                           const OperationDef &definition,
                                           const FullyConnectedAttributes &attr, const BHWC *shape);
  friend ConvBuffer1x1 CreateConvBuffer1x1Wino4x4To6x6(const DeviceInfo &device_info,
                                                       const OperationDef &definition,
                                                       const Convolution2DAttributes &attr,
                                                       const BHWC *shape);
  friend ConvBuffer1x1 CreateConvBuffer1x1DynamicWeights(const DeviceInfo &device_info,
                                                         const OperationDef &definition,
                                                         const Convolution2DAttributes &attr,
                                                         const BHWC &weights_shape,
                                                         const BHWC *dst_shape);

  template <DataType T>
  void UploadData(const InternalTensor<OHWI, T> &weights, const InternalTensor<Linear, T> &biases);
  template <DataType T> void UploadDataForWinograd4x4To6x6(const InternalTensor<OHWI, T> &weights);

  template <DataType T> void UploadWeights(const InternalTensor<OHWI, T> &weights);

  template <DataType T> void UploadBiases(const InternalTensor<Linear, T> &biases);

  std::string GenerateConvBuffer1x1(const OperationDef &op_def,
                                    const ConvBuffer1x1::ConvParams &conv_params, Arguments *args);

  ConvParams conv_params_;
};

template <DataType T>
void ConvBuffer1x1::UploadData(const InternalTensor<OHWI, T> &weights,
                               const InternalTensor<Linear, T> &biases)
{
  UploadWeights(weights);
  UploadBiases(biases);
}

template <DataType T>
void ConvBuffer1x1::UploadDataForWinograd4x4To6x6(const InternalTensor<OHWI, T> &weights)
{
  InternalTensor<OHWI, T> wino_weights;
  RearrangeWeightsToWinograd4x4To6x6Weights(weights, &wino_weights);
  UploadWeights(wino_weights);
  InternalTensor<Linear, DataType::FLOAT32> bias;
  bias.shape = Linear(weights.shape.o);
  bias.data.resize(weights.shape.o, 0.0f);
  UploadBiases(bias);
}

template <DataType T> void ConvBuffer1x1::UploadWeights(const InternalTensor<OHWI, T> &weights)
{
  const int dst_depth = DivideRoundUp(weights.shape.o, 4);
  const int src_depth = DivideRoundUp(weights.shape.i, 4);

  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;
  const int float4_size = sizeof(float4);
  // TODO
  // f32_weights ? sizeof(float4) : sizeof(half4);

  const int dst_depth_aligned = AlignByN(dst_depth, conv_params_.block_size.z);
  const int elements_count = weights.shape.h * weights.shape.w * src_depth * dst_depth_aligned * 4;

  BufferDescriptor desc;
  desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
  desc.element_size = 16;
  desc.memory_type = MemoryType::GLOBAL;
  desc.size = float4_size * elements_count;
  desc.data.resize(desc.size);

  if (f32_weights)
  {
    float4 *ptr = reinterpret_cast<float4 *>(desc.data.data());
    RearrangeWeightsToOHWIOGroupI4O4(weights, conv_params_.block_size.z,
                                     absl::MakeSpan(ptr, elements_count));
  }
  //   else
  //   {
  //     half4 *ptr = reinterpret_cast<half4 *>(desc.data.data());
  //     RearrangeWeightsToOHWIOGroupI4O4(weights, conv_params_.block_size.z,
  //                                      absl::MakeSpan(ptr, elements_count));
  //   }

  args_.AddObject("weights", absl::make_unique<BufferDescriptor>(std::move(desc)));
}

template <DataType T> void ConvBuffer1x1::UploadBiases(const InternalTensor<Linear, T> &biases)
{
  TensorLinearDescriptor desc;
  desc.storage_type = LinearStorageType::BUFFER;
  desc.element_type = definition_.GetDataType();
  int depth = AlignByN(biases.shape.v, 4 * conv_params_.block_size.z) / 4;
  desc.UploadLinearData(biases, depth);
  args_.AddObject("biases", absl::make_unique<TensorLinearDescriptor>(std::move(desc)));
}

bool IsConvBuffer1x1Supported(const OperationDef &definition, const Convolution2DAttributes &attr);

bool IsConvBuffer1x1Supported(const OperationDef &definition, const BHWC &weights_shape,
                              const Convolution2DAttributes &attr);

ConvBuffer1x1 CreateConvBuffer1x1(const DeviceInfo &device_info, const OperationDef &definition,
                                  const Convolution2DAttributes &attr, const BHWC *shape = nullptr);

ConvBuffer1x1 CreateConvBuffer1x1(const DeviceInfo &device_info, const OperationDef &definition,
                                  const FullyConnectedAttributes &attr,
                                  const BHWC *shape = nullptr);

ConvBuffer1x1 CreateConvBuffer1x1DynamicWeights(const DeviceInfo &device_info,
                                                const OperationDef &definition,
                                                const Convolution2DAttributes &attr,
                                                const BHWC &weights_shape,
                                                const BHWC *dst_shape = nullptr);

ConvBuffer1x1 CreateConvBuffer1x1Wino4x4To6x6(const DeviceInfo &device_info,
                                              const OperationDef &definition,
                                              const Convolution2DAttributes &attr,
                                              const BHWC *shape = nullptr);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_CONV_BUFFER_1X1_H__
