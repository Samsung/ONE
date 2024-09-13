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
#include "ir/operation/ElementwiseActivation.h"
#include <memory>

namespace onert
{
namespace backend
{
namespace acl_common
{

::arm_compute::TensorShape asTensorShape(const ir::Shape &shape, bool apply_dim_correction)
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
    res.set(ToARMComputeAxis(rank, axis).value(), tensor_shape.dim(axis), apply_dim_correction);
  }

  return res;
}

::arm_compute::Coordinates asTensorCoordinate(const ir::Coordinates &coord)
{
  const uint32_t rank = coord.size();

  ::arm_compute::Coordinates res{};

  res.set_num_dimensions(rank);

  for (uint32_t axis = 0; axis < rank; ++axis)
  {
    res.set(ToARMComputeAxis(rank, axis).value(), coord[axis]);
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
      return ::arm_compute::DataType::QSYMM8;
    case ir::DataType::QUANT_INT8_ASYMM:
      return ::arm_compute::DataType::QASYMM8_SIGNED;
    case ir::DataType::FLOAT16:
      return ::arm_compute::DataType::F16;
    case ir::DataType::INT64:
      return ::arm_compute::DataType::S64;
    case ir::DataType::QUANT_INT16_ASYMM:
      return ::arm_compute::DataType::QASYMM16;
    case ir::DataType::QUANT_INT8_SYMM_PER_CHANNEL:
      return ::arm_compute::DataType::QSYMM8_PER_CHANNEL;
    default:
      throw std::runtime_error("Not supported internal data type, yet");
      break;
  }
}

::arm_compute::QuantizationInfo asQuantizationInfo(const float scale, const int32_t offset)
{
  return ::arm_compute::QuantizationInfo(scale, offset);
}

::arm_compute::TensorInfo asTensorInfo(const ir::Shape &shape, const ir::TypeInfo &typeInfo,
                                       bool apply_dim_correction)
{
  ::arm_compute::TensorInfo info(asTensorShape(shape, apply_dim_correction), 1,
                                 asDataType(typeInfo.type()),
                                 asQuantizationInfo(typeInfo.scale(), typeInfo.zero_point()));
  info.set_data_layout(::arm_compute::DataLayout::NHWC);
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
      throw std::runtime_error{"Not supported internal activation, yet"};
      break;
  }
}

::arm_compute::ActivationLayerInfo
asActivationLayerInfo(const ir::operation::ElementwiseActivation::Type op_type, float alpha,
                      float beta)
{
  switch (op_type)
  {
    case ir::operation::ElementwiseActivation::Type::RELU:
      if (beta == 0.f)
      {
        if (alpha == ir::operation::ElementwiseActivation::infinity)
        {
          return ::arm_compute::ActivationLayerInfo{
            ::arm_compute::ActivationLayerInfo::ActivationFunction::RELU};
        }
        else
        {
          return ::arm_compute::ActivationLayerInfo{
            ::arm_compute::ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, alpha};
        }
      }
      else
      {
        return ::arm_compute::ActivationLayerInfo{
          ::arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, alpha, beta};
      }
    case ir::operation::ElementwiseActivation::Type::TANH:
      return ::arm_compute::ActivationLayerInfo{
        ::arm_compute::ActivationLayerInfo::ActivationFunction::TANH, alpha, beta};
    case ir::operation::ElementwiseActivation::Type::LOGISTIC:
      // NOTE The sigmoid function is a special case of the Logistic function when L=1, k=1, x0=0.
      // TODO In ACL and nnapi sepc, currently, Logistic's L always is 1, k always is 1, x0 always
      // 0(always sigmoid) regardless of values of the parameter.
      //      If ACL support non-sigmoid logistic, should fix param values.
      return ::arm_compute::ActivationLayerInfo{
        ::arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC};
    case ir::operation::ElementwiseActivation::Type::LEAKY_RELU:
      return ::arm_compute::ActivationLayerInfo{
        ::arm_compute::ActivationLayerInfo::ActivationFunction::LEAKY_RELU, alpha};
    default:
      throw std::runtime_error{"Not supported internal elementwise activation, yet"};
      break;
  }
}

arm_compute::Coordinates asCoordinates(const ir::Operand &operand, int32_t rank)
{
  std::set<uint32_t> axes = asSet(operand, rank);

  arm_compute::Coordinates reduce_axes;
  for (const int32_t axis : axes)
  {
    reduce_axes.set(reduce_axes.num_dimensions(), axis);
  }

  return reduce_axes;
}

std::set<uint32_t> asSet(const ir::Operand &operand, int32_t rank)
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
    axes.insert(ToARMComputeAxis(rank, axis).value());
  }

  return axes;
}

std::unique_ptr<AclFunction> asAclFunction(std::unique_ptr<::arm_compute::IFunction> &&layer)
{
  return std::make_unique<AclFunction>(std::move(layer));
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
    case ::arm_compute::DataType::QASYMM8_SIGNED:
      return ir::DataType::QUANT_INT8_ASYMM;
    case ::arm_compute::DataType::U8:
      return ir::DataType::UINT8;
    case ::arm_compute::DataType::QSYMM8:
      return ir::DataType::QUANT_INT8_SYMM;
    case ::arm_compute::DataType::F16:
      return ir::DataType::FLOAT16;
    case ::arm_compute::DataType::S64:
      return ir::DataType::INT64;
    default:
      throw std::runtime_error{"Not supported acl data type, yet"};
      break;
  }
}

arm_compute::PoolingType convertPoolType(ir::operation::Pool2D::PoolType pool_type_ir)
{
  switch (pool_type_ir)
  {
    case ir::operation::Pool2D::PoolType::AVG:
      return arm_compute::PoolingType::AVG;
    case ir::operation::Pool2D::PoolType::L2:
      return arm_compute::PoolingType::L2;
    case ir::operation::Pool2D::PoolType::MAX:
      return arm_compute::PoolingType::MAX;
    default:
      throw std::runtime_error("convertPoolType: Not supported operation yet");
  }
}

arm_compute::ReductionOperation convertReduceType(ir::operation::Reduce::ReduceType reduce_type_ir)
{
  switch (reduce_type_ir)
  {
    case ir::operation::Reduce::ReduceType::MAX:
      return arm_compute::ReductionOperation::MAX;
    case ir::operation::Reduce::ReduceType::MIN:
      return arm_compute::ReductionOperation::MIN;
    case ir::operation::Reduce::ReduceType::SUM:
      return arm_compute::ReductionOperation::SUM;
    default:
      throw std::runtime_error("convertReduceType: Not supported operation yet");
  }
}

arm_compute::PixelValue asPixelValue(const ir::Operand &operand)
{
  assert(operand.isConstant());
  assert(operand.shape().num_elements() == 1);
  switch (operand.typeInfo().type())
  {
    case ir::DataType::INT32:
      return arm_compute::PixelValue(operand.asScalar<int32_t>());
    case ir::DataType::INT64:
      return arm_compute::PixelValue(operand.asScalar<int64_t>());
    case ir::DataType::UINT32:
      return arm_compute::PixelValue(operand.asScalar<uint64_t>());
    case ir::DataType::UINT8:
      return arm_compute::PixelValue(operand.asScalar<uint8_t>());
    case ir::DataType::FLOAT32:
      return arm_compute::PixelValue(operand.asScalar<float>());
    default:
      throw std::runtime_error("asPixelValue : Not supported datatype yet");
  }
}

arm_compute::Size2D asDilation(uint32_t dilation_width, uint32_t dilation_height)
{
  assert(dilation_width != 0);
  assert(dilation_height != 0);

  return arm_compute::Size2D(dilation_width, dilation_height);
}

} // namespace acl_common
} // namespace backend
} // namespace onert
