/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Convert.h"

#include "Swizzle.h"
#include "ir/DataType.h"
#include <memory>

namespace
{

::arm_compute::DataLayout asDataLayout(onert::ir::Layout layout)
{
  switch (layout)
  {
    case onert::ir::Layout::NHWC:
      return ::arm_compute::DataLayout::NHWC;
    case onert::ir::Layout::NCHW:
      return ::arm_compute::DataLayout::NCHW;
    default:
      return ::arm_compute::DataLayout::UNKNOWN;
  }
}

} // namespace

namespace onert
{
namespace backend
{
namespace acl_common
{

::arm_compute::TensorShape asTensorShape(const ir::Shape &shape, ir::Layout frontend_layout,
                                         ir::Layout backend_layout, bool apply_dim_correction)
{
  // If shape's rank is 0, the tensor is a scalar
  // Sometimes, some ACL kernel can use a scalar as tensor. But ACL does not allocate buffer for
  // tensor having rank as 0.
  const auto tensor_shape = shape.rank() == 0 ? ir::Shape{1} : shape;

  const uint32_t rank = tensor_shape.rank();

  ::arm_compute::TensorShape res{};

  res.set_num_dimensions(rank);

  for (uint32_t axis = 0; axis < rank; ++axis)
  {
    // NOTE In some cases, in incorrect dimensions is required.
    // For example, intput_size is 1 in LSTM. The input-to-input weights([num_units, input_size]) of
    // LSTM is used as the weight of the FullyConnected.
    // The FullyConnected's weight must be greater or equal than 2-dimensions.
    // However, if the dimension correction is applied to input_to_input_weights with input_size
    // equal to 1, it will be changed to 1-D.
    // So input_to_input_weights is not used by the weight of FullyConnected.
    res.set(ToARMComputeAxis(rank, axis, frontend_layout, backend_layout).value(),
            tensor_shape.dim(axis), apply_dim_correction);
  }

  return res;
}

::arm_compute::Coordinates asTensorCoordinate(const ir::Coordinates &coord,
                                              ir::Layout frontend_layout, ir::Layout backend_layout)
{
  const uint32_t rank = coord.size();

  ::arm_compute::Coordinates res{};

  res.set_num_dimensions(rank);

  for (uint32_t axis = 0; axis < rank; ++axis)
  {
    res.set(ToARMComputeAxis(rank, axis, frontend_layout, backend_layout).value(), coord[axis]);
  }

  return res;
}

::arm_compute::DataType asDataType(const ir::DataType type)
{
  switch (type)
  {
    case ir::DataType::FLOAT32:
      return ::arm_compute::DataType::F32;
    case ir::DataType::INT32:
      return ::arm_compute::DataType::S32;
    case ir::DataType::UINT32:
      return ::arm_compute::DataType::U32;
    case ir::DataType::QUANT_UINT8_ASYMM:
      return ::arm_compute::DataType::QASYMM8;
    case ir::DataType::BOOL8:
    case ir::DataType::UINT8:
      return ::arm_compute::DataType::U8;
    case ir::DataType::QUANT_INT8_SYMM:
      return ::arm_compute::DataType::S8;
    case ir::DataType::FLOAT16:
      return ::arm_compute::DataType::F16;
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

::arm_compute::QuantizationInfo asQuantizationInfo(const float scale, const int32_t offset)
{
  return ::arm_compute::QuantizationInfo(scale, offset);
}

::arm_compute::TensorInfo asTensorInfo(const ir::Shape &shape, const ir::TypeInfo &typeInfo,
                                       ir::Layout frontend_layout, ir::Layout backend_layout,
                                       bool apply_dim_correction)
{
  ::arm_compute::TensorInfo info(
      asTensorShape(shape, frontend_layout, backend_layout, apply_dim_correction), 1,
      asDataType(typeInfo.type()), asQuantizationInfo(typeInfo.scale(), typeInfo.offset()));
  info.set_data_layout(asDataLayout(backend_layout));
  return info;
}

::arm_compute::PadStrideInfo asPadStrideInfo(const ir::ExplicitPadding &padding,
                                             const ir::Stride &stride)
{
  return ::arm_compute::PadStrideInfo{stride.horizontal,
                                      stride.vertical,
                                      padding.left,
                                      padding.right,
                                      padding.top,
                                      padding.bottom,
                                      ::arm_compute::DimensionRoundingType::FLOOR};
}

::arm_compute::ActivationLayerInfo asActivationLayerInfo(const ir::Activation act_code)
{
  switch (act_code)
  {
    case ir::Activation::NONE:
      return ::arm_compute::ActivationLayerInfo{};
    case ir::Activation::RELU:
      return ::arm_compute::ActivationLayerInfo{
          ::arm_compute::ActivationLayerInfo::ActivationFunction::RELU};
    case ir::Activation::RELU1:
      return ::arm_compute::ActivationLayerInfo{
          ::arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 1.0f, -1.0f};
    case ir::Activation::RELU6:
      return ::arm_compute::ActivationLayerInfo{
          ::arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.0f, 0.0f};
    // Cases for activation of LSTM.
    case ir::Activation::TANH:
      return ::arm_compute::ActivationLayerInfo{
          ::arm_compute::ActivationLayerInfo::ActivationFunction::TANH, 1.0f, 1.0f};
    case ir::Activation::SIGMOID:
      // NOTE The sigmoid function is a special case of the Logistic function when L=1, k=1, x0=0.
      // TODO In ACL and nnapi sepc, currently, Logistic's L always is 1, k always is 1, x0 always
      // 0(always sigmoid) regardless of values of the parameter.
      //      If ACL support non-sigmoid logistic, should fix param values.
      return ::arm_compute::ActivationLayerInfo{
          ::arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC, 0.0f, 0.0f};
    default:
      throw std::runtime_error{"Not supported, yet"};
      break;
  }
}

arm_compute::Coordinates asCoordinates(const ir::Operand &operand, int32_t rank,
                                       ir::Layout frontend_layout, ir::Layout backend_layout)
{
  std::set<uint32_t> axes = asSet(operand, rank, frontend_layout, backend_layout);

  arm_compute::Coordinates reduce_axes;
  for (const int32_t axis : axes)
  {
    reduce_axes.set(reduce_axes.num_dimensions(), axis);
  }

  return reduce_axes;
}

std::set<uint32_t> asSet(const ir::Operand &operand, int32_t rank, ir::Layout frontend_layout,
                         ir::Layout backend_layout)
{
  std::set<std::uint32_t> axes;

  for (size_t i = 0; i < operand.shape().num_elements(); ++i)
  {
    int32_t axis = 0;
    switch (operand.typeInfo().type())
    {
      case ir::DataType::INT32:
        axis = reinterpret_cast<const int32_t *>(operand.data()->base())[i];
        break;
      case ir::DataType::INT64:
        axis = reinterpret_cast<const int64_t *>(operand.data()->base())[i];
        break;
      default:
        throw std::runtime_error("acl_common::asSet: Not supported data type");
    }
    if (axis < 0)
      axis += rank;
    axes.insert(ToARMComputeAxis(rank, axis, frontend_layout, backend_layout).value());
  }

  return axes;
}

std::unique_ptr<AclFunction> asAclFunction(std::unique_ptr<::arm_compute::IFunction> &&layer)
{
  return std::make_unique<AclFunction>(std::move(layer));
}

std::unique_ptr<AclClFunction> asAclClFunction(std::unique_ptr<::arm_compute::IFunction> &&layer)
{
  return std::make_unique<AclClFunction>(std::move(layer));
}

ir::Layout asRuntimeLayout(::arm_compute::DataLayout data_layout)
{
  switch (data_layout)
  {
    case ::arm_compute::DataLayout::NHWC:
      return ir::Layout::NHWC;
    case ::arm_compute::DataLayout::NCHW:
      return ir::Layout::NCHW;
    default:
      return ir::Layout::UNKNOWN;
  }
}

ir::DataType asRuntimeDataType(::arm_compute::DataType data_type)
{
  switch (data_type)
  {
    case ::arm_compute::DataType::F32:
      return ir::DataType::FLOAT32;
    case ::arm_compute::DataType::S32:
      return ir::DataType::INT32;
    case ::arm_compute::DataType::U32:
      return ir::DataType::UINT32;
    case ::arm_compute::DataType::QASYMM8:
      return ir::DataType::QUANT_UINT8_ASYMM;
    case ::arm_compute::DataType::U8:
      return ir::DataType::UINT8;
    case ::arm_compute::DataType::QSYMM8:
      return ir::DataType::QUANT_INT8_SYMM;
    case ::arm_compute::DataType::F16:
      return ir::DataType::FLOAT16;
    default:
      throw std::runtime_error{"Not supported, yet"};
      break;
  }
}

arm_compute::ReduceOperation convertReduceType(ir::operation::Reduce::ReduceType reduce_type_ir)
{
  switch (reduce_type_ir)
  {
    case ir::operation::Reduce::ReduceType::MAX:
      return arm_compute::ReduceOperation::MAX;
    case ir::operation::Reduce::ReduceType::MIN:
      return arm_compute::ReduceOperation::MIN;
    case ir::operation::Reduce::ReduceType::SUM:
      return arm_compute::ReduceOperation::SUM;
    default:
      throw std::runtime_error("convertReduceType: Not supported operation yet");
  }
}

} // namespace acl_common
} // namespace backend
} // namespace onert
