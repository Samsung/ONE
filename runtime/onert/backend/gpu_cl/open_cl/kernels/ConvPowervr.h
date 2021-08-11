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

#ifndef __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_CONV_POWERVR_H__
#define __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_CONV_POWERVR_H__

#include <cstring>
#include <vector>

#include "open_cl/Buffer.h"
#include "open_cl/ClDevice.h"
#include "open_cl/kernels/ConvCommon.h"
#include "open_cl/kernels/GpuOperation.h"
#include "open_cl/kernels/Util.h"
#include "open_cl/LinearStorage.h"
#include "open_cl/Tensor.h"
#include "open_cl/Texture2d.h"
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

class ConvPowerVR : public GPUOperation
{
public:
  ConvPowerVR() = default;
  void GetPossibleKernelWorkGroups(TuningType tuning_type, const DeviceInfo &device_info,
                                   const KernelInfo &kernel_info,
                                   std::vector<int3> *work_groups) const override;
  absl::Status BindArguments(ArgumentsBinder *args) override;
  int3 GetGridSize() const override;

  ConvWeightsDescription GetConvWeightsDescription() const
  {
    ConvWeightsDescription desc;
    desc.layout = ConvWeightsLayout::kOHWIOGroupI4O4;
    desc.output_group_size = conv_params_.block_size.w;
    return desc;
  }

  // Move only
  ConvPowerVR(ConvPowerVR &&operation);
  ConvPowerVR &operator=(ConvPowerVR &&operation);
  ConvPowerVR(const ConvPowerVR &) = delete;
  ConvPowerVR &operator=(const ConvPowerVR &) = delete;

private:
  enum class WeightsUploadType
  {
    LOCAL_MEM_ASYNC_SUBGROUP, // we use it for PowerVR with workgroup size = 32
    LOCAL_MEM_BY_THREADS,
    GLOBAL_MEM,
    CONSTANT_MEM,
    PRIVATE_MEM_SIMD_BROADCAST,
    TEXTURES_MEM_X4, // 4 textures for weights
  };

  struct ConvParams
  {
    // Usually we use this combinations for CalculationPrecision:
    // F32: all F32
    // F16: all F16
    // F32_F16: all besides accumulator is F16, including weights
    // But for PowerVR we can achieve better performance in F32_F16 with F32
    // weights, so for PowerVR in this kernel we have F32 weights for
    // F32_F16 precision mode
    DataType weights_data_type; // used for weights and biases
    int4 block_size;            // WHDS
    bool fixed_work_group_size;
    bool linear_spatial; // spatial dimensions are Width/Height/Depth
    bool different_weights_for_height;
    int src_depth_loop_size;
    WeightsUploadType weights_upload_type;
    bool x_kernel_is_1;
    bool y_kernel_is_1;
    bool z_kernel_is_1;

    // used only with PRIVATE_MEM_SIMD_BROADCAST
    int simd_size = 1;

    bool AreWeightsBuffer() const
    {
      return weights_upload_type != WeightsUploadType::TEXTURES_MEM_X4;
    }

    bool IsPrivateMemBroadcast() const
    {
      return weights_upload_type == WeightsUploadType::PRIVATE_MEM_SIMD_BROADCAST;
    }
  };

  ConvPowerVR(const OperationDef &definition, const Convolution2DAttributes &attr,
              const DeviceInfo &device_info, const BHWC *dst_shape = nullptr);
  ConvPowerVR(const OperationDef &definition, const Convolution2DAttributes &attr,
              const BHWC &weights_shape, const DeviceInfo &device_info,
              const BHWC *dst_shape = nullptr);
  ConvPowerVR(const OperationDef &definition, const FullyConnectedAttributes &attr,
              const DeviceInfo &device_info, const BHWC *dst_shape = nullptr);
  explicit ConvPowerVR(const OperationDef &definition);
  ConvPowerVR(const OperationDef &definition, const Convolution3DAttributes &attr,
              const DeviceInfo &device_info, const BHWDC *dst_shape = nullptr);

  void GenerateCode(const DeviceInfo &device_info);

  template <DataType T>
  void UploadData(const InternalTensor<OHWI, T> &weights, const InternalTensor<Linear, T> &biases);
  template <DataType T> void UploadDataForWinograd4x4To6x6(const InternalTensor<OHWI, T> &weights);

  template <DataType T> void UploadWeights(const InternalTensor<OHWI, T> &weights);

  template <DataType T> void UploadWeights(const InternalTensor<OHWDI, T> &weights);

  template <DataType T> void UploadBias(const InternalTensor<Linear, T> &bias);

  friend ConvPowerVR CreateConvPowerVR(const DeviceInfo &device_info,
                                       const OperationDef &definition,
                                       const Convolution2DAttributes &attr, const BHWC *dst_shape);

  friend ConvPowerVR CreateConvPowerVR(const DeviceInfo &device_info,
                                       const OperationDef &definition,
                                       const FullyConnectedAttributes &attr, const BHWC *dst_shape);

  friend ConvPowerVR CreateConvPowerVRDynamicWeights(const DeviceInfo &device_info,
                                                     const OperationDef &definition,
                                                     const Convolution2DAttributes &attr,
                                                     const BHWC &weights_shape,
                                                     const BHWC *dst_shape);

  friend ConvPowerVR CreateConvPowerVRWino4x4To6x6(const DeviceInfo &device_info,
                                                   const OperationDef &definition,
                                                   const Convolution2DAttributes &attr,
                                                   const BHWC *dst_shape);

  friend ConvPowerVR CreateConvPowerVR3D(const DeviceInfo &device_info,
                                         const OperationDef &definition,
                                         const Convolution3DAttributes &attr,
                                         const BHWDC *dst_shape);

  ConvParams GuessBestParams(const DeviceInfo &device_info, const OperationDef &definition,
                             const Convolution2DAttributes &attr, const BHWC *dst_shape = nullptr);
  ConvParams GuessBestParams(const DeviceInfo &device_info, const OperationDef &definition,
                             const Convolution2DAttributes &attr, const BHWC &weights_shape,
                             const BHWC *dst_shape = nullptr);
  ConvParams GuessBestParams(const DeviceInfo &device_info, const OperationDef &definition,
                             const FullyConnectedAttributes &attr, const BHWC *dst_shape = nullptr);
  ConvParams GuessBestParamsWinograd(const DeviceInfo &device_info, const OperationDef &definition,
                                     const Convolution2DAttributes &attr,
                                     const BHWC *dst_shape = nullptr);
  ConvParams GuessBestParams(const DeviceInfo &device_info, const OperationDef &definition,
                             const Convolution3DAttributes &attr, const BHWDC *dst_shape = nullptr);
  ConvParams GuessBestParams(const DeviceInfo &device_info, const OperationDef &definition,
                             int src_depth, int dst_depth, bool x_kernel_is_1, bool y_kernel_is_1,
                             bool different_weights_for_height, const BHWC *dst_shape = nullptr);

  std::string GenerateConv(const DeviceInfo &device_info, const OperationDef &op_def,
                           bool stride_correction, const ConvParams &conv_params);

  int4 stride_;
  int4 padding_;
  int4 kernel_size_;
  int4 dilation_;
  ConvParams conv_params_;
};

template <DataType T>
void ConvPowerVR::UploadData(const InternalTensor<OHWI, T> &weights,
                             const InternalTensor<Linear, T> &biases)
{
  UploadWeights(weights);
  UploadBias(biases);
}

template <DataType T>
void ConvPowerVR::UploadDataForWinograd4x4To6x6(const InternalTensor<OHWI, T> &weights)
{
  InternalTensor<OHWI, T> wino_weights;
  RearrangeWeightsToWinograd4x4To6x6Weights(weights, &wino_weights);
  UploadWeights(wino_weights);
  InternalTensor<Linear, DataType::FLOAT32> biases;
  biases.shape = Linear(weights.shape.o);
  biases.data.resize(weights.shape.o, 0.0f);
  UploadBias(biases);
}

template <DataType T> void ConvPowerVR::UploadBias(const InternalTensor<Linear, T> &bias)
{
  BufferDescriptor desc;
  desc.element_type = conv_params_.weights_data_type;
  desc.element_size = 4;
  desc.memory_type =
    conv_params_.weights_upload_type == ConvPowerVR::WeightsUploadType::CONSTANT_MEM
      ? MemoryType::CONSTANT
      : MemoryType::GLOBAL;
  const int float_size = sizeof(float);
  // TODO
  // conv_params_.weights_data_type == DataType::FLOAT32 ? sizeof(float) : sizeof(half);
  int aligned_channels = AlignByN(bias.shape.v, 4 * conv_params_.block_size.w);
  desc.size = float_size * aligned_channels;
  desc.data.resize(desc.size);
  if (conv_params_.weights_data_type == DataType::FLOAT32)
  {
    float *gpu_data = reinterpret_cast<float *>(desc.data.data());
    for (int i = 0; i < aligned_channels; ++i)
    {
      gpu_data[i] = i < bias.shape.v ? bias.data[i] : 0.0f;
    }
  }
  //   else
  //   {
  //     half *gpu_data = reinterpret_cast<half *>(desc.data.data());
  //     for (int i = 0; i < aligned_channels; ++i)
  //     {
  //       gpu_data[i] = i < bias.shape.v ? bias.data[i] : 0.0f;
  //     }
  //   }
  args_.AddObject("biases", absl::make_unique<BufferDescriptor>(std::move(desc)));
}

template <DataType T> void ConvPowerVR::UploadWeights(const InternalTensor<OHWI, T> &weights)
{
  const int dst_slices = AlignByN(DivideRoundUp(weights.shape.o, 4), conv_params_.block_size.w);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);

  const bool f32_weights = conv_params_.weights_data_type == DataType::FLOAT32;
  const int float4_size = sizeof(float4);
  // TODO
  // f32_weights ? sizeof(float4) : sizeof(half4);

  const int elements_count = weights.shape.h * weights.shape.w * src_slices * dst_slices * 4;

  std::vector<uint8_t> data(float4_size * elements_count);

  if (f32_weights)
  {
    float4 *ptr = reinterpret_cast<float4 *>(data.data());
    if (conv_params_.AreWeightsBuffer())
    {
      RearrangeWeightsToOHWIOGroupI4O4(weights, conv_params_.block_size.w,
                                       absl::MakeSpan(ptr, elements_count));
    }
    else
    {
      RearrangeWeightsToI4HWIOOGroupO4(weights, conv_params_.block_size.w,
                                       absl::MakeSpan(ptr, elements_count));
    }
  }
  //   else
  //   {
  //     half4 *ptr = reinterpret_cast<half4 *>(data.data());
  //     if (conv_params_.AreWeightsBuffer())
  //     {
  //       RearrangeWeightsToOHWIOGroupI4O4(weights, conv_params_.block_size.w,
  //                                        absl::MakeSpan(ptr, elements_count));
  //     }
  //     else
  //     {
  //       RearrangeWeightsToI4HWIOOGroupO4(weights, conv_params_.block_size.w,
  //                                        absl::MakeSpan(ptr, elements_count));
  //     }
  //   }
  if (conv_params_.AreWeightsBuffer())
  {
    BufferDescriptor desc;
    desc.element_type = conv_params_.weights_data_type;
    desc.element_size = 4;
    desc.memory_type =
      conv_params_.weights_upload_type == ConvPowerVR::WeightsUploadType::CONSTANT_MEM
        ? MemoryType::CONSTANT
        : MemoryType::GLOBAL;
    desc.size = float4_size * elements_count;
    desc.data = std::move(data);
    args_.AddObject("weights", absl::make_unique<BufferDescriptor>(std::move(desc)));
  }
  else
  {
    const int texture_width = dst_slices;
    const int texture_height = src_slices * weights.shape.h * weights.shape.w;
    const int sub_size = float4_size * texture_width * texture_height;
    for (int i = 0; i < 4; ++i)
    {
      Texture2DDescriptor desc;
      desc.element_type = conv_params_.weights_data_type;
      desc.size = int2(texture_width, texture_height);
      desc.data.resize(sub_size);
      std::memcpy(desc.data.data(), data.data() + sub_size * i, sub_size);
      const std::string name = "weights" + std::to_string(i);
      args_.AddObject(name, absl::make_unique<Texture2DDescriptor>(std::move(desc)));
    }
  }
}

template <DataType T> void ConvPowerVR::UploadWeights(const InternalTensor<OHWDI, T> &weights)
{
  const int block_size = conv_params_.block_size.w;
  const int dst_slices = AlignByN(DivideRoundUp(weights.shape.o, 4), block_size);
  const int src_slices = DivideRoundUp(weights.shape.i, 4);

  const int elements_count =
    weights.shape.d * weights.shape.h * weights.shape.w * src_slices * dst_slices * 4;
  const bool f32_weights = definition_.precision == CalculationsPrecision::F32;

  const int float4_size = f32_weights ? 16 : 8;

  std::vector<uint8_t> data(float4_size * elements_count);

  if (f32_weights)
  {
    float4 *ptr = reinterpret_cast<float4 *>(data.data());
    if (conv_params_.AreWeightsBuffer())
    {
      RearrangeWeightsToODHWIOGroupI4O4(weights, conv_params_.block_size.w,
                                        absl::MakeSpan(ptr, elements_count));
    }
    else
    {
      RearrangeWeightsToI4DHWIOOGroupO4(weights, conv_params_.block_size.w,
                                        absl::MakeSpan(ptr, elements_count));
    }
  }
  //   else
  //   {
  //     half4 *ptr = reinterpret_cast<half4 *>(data.data());
  //     if (conv_params_.AreWeightsBuffer())
  //     {
  //       RearrangeWeightsToODHWIOGroupI4O4(weights, conv_params_.block_size.w,
  //                                         absl::MakeSpan(ptr, elements_count));
  //     }
  //     else
  //     {
  //       RearrangeWeightsToI4DHWIOOGroupO4(weights, conv_params_.block_size.w,
  //                                         absl::MakeSpan(ptr, elements_count));
  //     }
  //   }

  if (conv_params_.AreWeightsBuffer())
  {
    BufferDescriptor desc;
    desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
    desc.element_size = 4;
    desc.size = float4_size * elements_count;
    desc.data = std::move(data);
    args_.AddObject("weights", absl::make_unique<BufferDescriptor>(std::move(desc)));
  }
  else
  {
    const int texture_width = dst_slices;
    const int texture_height = src_slices * weights.shape.d * weights.shape.h * weights.shape.w;
    int sub_size = float4_size * texture_width * texture_height;
    for (int i = 0; i < 4; ++i)
    {
      Texture2DDescriptor desc;
      desc.element_type = f32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
      desc.size = int2(texture_width, texture_height);
      desc.data.resize(sub_size);
      memcpy(desc.data.data(), data.data() + sub_size * i, sub_size);
      const std::string name = "weights" + std::to_string(i);
      args_.AddObject(name, absl::make_unique<Texture2DDescriptor>(std::move(desc)));
    }
  }
}

ConvPowerVR CreateConvPowerVR(const DeviceInfo &device_info, const OperationDef &definition,
                              const Convolution2DAttributes &attr, const BHWC *dst_shape = nullptr);

ConvPowerVR CreateConvPowerVR(const DeviceInfo &device_info, const OperationDef &definition,
                              const FullyConnectedAttributes &attr,
                              const BHWC *dst_shape = nullptr);

ConvPowerVR CreateConvPowerVRDynamicWeights(const DeviceInfo &device_info,
                                            const OperationDef &definition,
                                            const Convolution2DAttributes &attr,
                                            const BHWC &weights_shape,
                                            const BHWC *dst_shape = nullptr);

ConvPowerVR CreateConvPowerVRWino4x4To6x6(const DeviceInfo &device_info,
                                          const OperationDef &definition,
                                          const Convolution2DAttributes &attr,
                                          const BHWC *dst_shape = nullptr);

ConvPowerVR CreateConvPowerVR3D(const DeviceInfo &device_info, const OperationDef &definition,
                                const Convolution3DAttributes &attr,
                                const BHWDC *dst_shape = nullptr);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_OPENCL_KERNELS_CONV_POWERVR_H__
