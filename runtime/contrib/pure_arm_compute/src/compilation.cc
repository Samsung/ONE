/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

/**
 * @file     compilation.cc
 * @brief    This file contains ANeuralNetworksCompilation APIs and related classes
 * @ingroup  COM_AI_RUNTIME
 */

#include <NeuralNetworks.h>

// For CLKernelLibraryEx initialization
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"

#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#include <arm_compute/runtime/CL/CLSubTensor.h>
#include <arm_compute/runtime/CL/CLFunctions.h>   // Include all ARM Compute CL functions
#include <arm_compute/runtime/CL/CLFunctionsEx.h> // Include all ARM Compute EX CL functions

#include <arm_compute/runtime/SubTensor.h>
#include <arm_compute/runtime/NEON/NEFunctions.h>   // Include all ARM Compute NEON functions
#include <arm_compute/runtime/NEON/NEFunctionsEx.h> // Include all ARM Compute EX NEON functions

#include "internal/arm_compute.h"
#include "internal/arm_compute/Cast.h"
#include "internal/arm_compute/matrix/View.h"
#include "internal/arm_compute/kernel/View.h"
#include "internal/nnapi/matrix/Reader.h"
#include "internal/nnapi/kernel/Reader.h"
#include "internal/nnapi/feature/Reader.h"
#include "internal/nnapi/feature/View.h"
#include "internal/nnapi/tensor/Reader.h"
#include "internal/arm_compute/feature/View.h"
#include "internal/arm_compute/tensor/View.h"

#include <arm_compute/runtime/misc/functions/GenericReshapeLayer.h>
#include <arm_compute/runtime/misc/functions/GenericGather.h>

#include "misc/matrix/IndexIterator.h"
#include "misc/kernel/IndexIterator.h"
#include "misc/feature/IndexIterator.h"
#include "misc/tensor/IndexIterator.h"

#include <memory>

#include "compilation.h"
#include "model.h"
#include "logging.h"

using namespace arm_compute::misc;

template <typename T> T from_env(const char *);

template <> bool from_env(const char *s)
{
  if (s == nullptr)
  {
    return false;
  }

  return std::stoi(s) != 0;
}

const char *to_string(const PaddingCode &code)
{
  assert((ANEURALNETWORKS_PADDING_SAME == code) || (ANEURALNETWORKS_PADDING_VALID == code));

  switch (code)
  {
    case ANEURALNETWORKS_PADDING_SAME:
      return "ANEURALNETWORKS_PADDING_SAME";
    case ANEURALNETWORKS_PADDING_VALID:
      return "ANEURALNETWORKS_PADDING_VALID";
  }

  return nullptr;
}

struct Padding
{
  uint32_t top;
  uint32_t bottom;
  uint32_t left;
  uint32_t right;
};

struct Stride
{
  uint32_t vertical;
  uint32_t horizontal;
};

Padding valid_padding(void)
{
  //
  // ANEURALNETWORKS_PADDING_VALID
  //
  // VALID padding. No padding.
  //
  // When the input size is not evenly divisible by the filter size,
  // the input at the end that could not fill the whole filter tile
  // will simply be ignored.
  //
  Padding padding;

  padding.top = 0;
  padding.bottom = 0;
  padding.left = 0;
  padding.right = 0;

  return padding;
}

Padding same_padding(const nnfw::misc::feature::Shape &ifm_shape,
                     const nnfw::misc::feature::Shape &ofm_shape, const Stride &stride, uint32_t kw,
                     uint32_t kh)
{
  Padding padding;

  // ANEURALNETWORKS_PADDING_SAME (from NNAPI spec)
  //
  // SAME padding. Padding on both ends are the "same":
  //
  // padding_to_beginning = total_padding / 2
  // padding_to_end = (total_padding + 1)/2.
  //
  const int32_t vertical_needed_input = (ofm_shape.H - 1) * stride.vertical + kh;
  const int32_t vertical_total_padding = std::max(0, vertical_needed_input - ifm_shape.H);

  const int32_t horizontal_needed_input = (ofm_shape.W - 1) * stride.horizontal + kw;
  const int32_t horizontal_total_padding = std::max(0, horizontal_needed_input - ifm_shape.W);

  padding.top = vertical_total_padding / 2;
  padding.bottom = (vertical_total_padding + 1) / 2;
  padding.left = horizontal_total_padding / 2;
  padding.right = (horizontal_total_padding + 1) / 2;

  return padding;
}

::arm_compute::PadStrideInfo asPadStrideInfo(const Padding &padding, const Stride &stride)
{
  return ::arm_compute::PadStrideInfo{stride.horizontal,
                                      stride.vertical,
                                      padding.left,
                                      padding.right,
                                      padding.top,
                                      padding.bottom,
                                      ::arm_compute::DimensionRoundingType::FLOOR};
}

::arm_compute::ActivationLayerInfo asActInfo(FuseCode act)
{
  if (act == ANEURALNETWORKS_FUSED_NONE)
  {
    return ::arm_compute::ActivationLayerInfo();
  }
  else if (act == ANEURALNETWORKS_FUSED_RELU)
  {
    return ::arm_compute::ActivationLayerInfo(
        ::arm_compute::ActivationLayerInfo::ActivationFunction::RELU);
  }
  else if (act == ANEURALNETWORKS_FUSED_RELU1)
  {
    return ::arm_compute::ActivationLayerInfo(
        ::arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 1.0f, -1.0f);
  }
  else if (act == ANEURALNETWORKS_FUSED_RELU6)
  {
    return ::arm_compute::ActivationLayerInfo(
        ::arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.0f, 0.0f);
  }
  else
  {
    throw std::runtime_error("Not supported, yet");
  }
}

struct IAllocationContext
{
  virtual ~IAllocationContext() = default;

  virtual ::arm_compute::ITensor *at(const ::internal::tflite::operand::Index &ind) const = 0;
};

#include "internal/IExecutionBuilder.h"

using Initializer = std::function<void(::arm_compute::ITensor &)>;
using Stage = std::function<void(const IAllocationContext &, IExecutionBuilder &)>;

using namespace std::placeholders;

template <typename T>
static void initFeatureTensor(::arm_compute::ITensor &tensor,
                              const nnfw::misc::feature::Shape &feature_shape,
                              const uint8_t *feature_base, const size_t feature_size)
{
  const ::internal::nnapi::feature::Reader<T> from{
      feature_shape, reinterpret_cast<const T *>(feature_base), feature_size};
  ::internal::arm_compute::feature::View<T> into{&tensor};

  ::nnfw::misc::feature::iterate(feature_shape)
      << [&](uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) {
           const auto value = from.at(batch, ch, row, col);
           into.at(batch, ch, row, col) = value;
         };
}

template <typename T>
static void initVectorTensor(::arm_compute::ITensor &tensor, const uint8_t *vec_base,
                             const size_t vec_size)
{
  for (uint32_t n = 0; n < vec_size; ++n)
  {
    const ::arm_compute::Coordinates coordinate{n};

    T *into = reinterpret_cast<T *>(tensor.ptr_to_element(coordinate));

    const T *from = reinterpret_cast<const T *>(vec_base) + n;
    const auto value = *from;

    *into = value;
  }
}

template <typename T>
static void initTensor3D(::arm_compute::ITensor &tensor,
                         const nnfw::misc::tensor::Shape &tensor_shape, const uint8_t *tensor_base,
                         const size_t tensor_size)
{
  const ::internal::nnapi::tensor::Reader<T> from{
      tensor_shape, reinterpret_cast<const T *>(tensor_base), tensor_size};
  ::internal::arm_compute::tensor::View<T> into{&tensor};

  ::nnfw::misc::tensor::iterate(tensor_shape) << [&](const nnfw::misc::tensor::Index &index_nnapi) {
    ::nnfw::misc::tensor::Index index_ACL = ::nnfw::misc::tensor::copy_reverse(index_nnapi);
    into.at(index_ACL) = from.at(index_nnapi);
  };
}

template <typename T>
static void initMatrixTensor(::arm_compute::ITensor &tensor,
                             const nnfw::misc::matrix::Shape &matrix_shape,
                             const uint8_t *matrix_base, const size_t matrix_size)
{
  const ::internal::nnapi::matrix::Reader<T> from{
      matrix_shape, reinterpret_cast<const T *>(matrix_base), matrix_size};
  ::internal::arm_compute::matrix::View<T> into{&tensor};

  ::nnfw::misc::matrix::iterate(matrix_shape) << [&](uint32_t row, uint32_t col) {
    const auto value = from.at(row, col);
    into.at(row, col) = value;
  };
}

template <typename T>
static void initReorderVectorTensor(::arm_compute::ITensor &tensor, const uint8_t *vec_base,
                                    const size_t vec_size)
{
  for (uint32_t n = 0; n < vec_size; ++n)
  {
    const ::arm_compute::Coordinates coordinate{ToARMComputeAxis(vec_size, n).value()};

    T *into = reinterpret_cast<T *>(tensor.ptr_to_element(coordinate));

    const T *from = reinterpret_cast<const T *>(vec_base) + n;
    const auto value = *from;

    *into = value;
  }
}

template <typename T>
static void initKernelTensor(::arm_compute::ITensor &tensor,
                             const nnfw::misc::kernel::Shape &kernel_shape,
                             const uint8_t *kernel_base, const size_t kernel_size)
{
  const ::internal::nnapi::kernel::Reader<T> from{
      kernel_shape, reinterpret_cast<const T *>(kernel_base), kernel_size};
  ::internal::arm_compute::kernel::View<T> into{&tensor};

  ::nnfw::misc::kernel::iterate(kernel_shape)
      << [&](uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) {
           const auto value = from.at(nth, ch, row, col);
           into.at(nth, ch, row, col) = value;
         };
}

/**
 * @brief Structure to provide interface methods of compilation plan builder
 */
struct IPlanBuilder
{
  /**
   * @brief  Destruct IPlanBuilder object using default destructor
   */
  virtual ~IPlanBuilder() = default;

  /**
   * @brief  Add TensorInfo with Shape Constraints
   * @param [in] ind   Index of operand
   * @param [in] info  TensorInfo value to set to index of operand
   * @return  N/A
   */
  virtual void addShapeConstr(const ::internal::tflite::operand::Index &ind,
                              const ::arm_compute::TensorInfo &info) = 0;
  /**
   * @brief  Add Subsumption constraints
   * @param [in] ind  Index of operand
   * @param [in] base  Index of base operand of Subsumption
   * @param [in] offset  Offset of Subsumption
   * @param [in] shape  Shape of Subsumption
   * @param [in] extend_parent  extend_parent value of Subsumption
   * @return  N/A
   */
  virtual void addSubsumptionConstr(const ::internal::tflite::operand::Index &ind,
                                    const ::internal::tflite::operand::Index &base,
                                    const ::arm_compute::Coordinates &offset,
                                    const ::arm_compute::TensorShape &shape,
                                    bool extend_parent = false) = 0;
  /**
   * @brief  Add Initializer lambda with ITensor param
   * @param [in] ind  Index of operand
   * @param [in] initializer  Initializer to add
   * @return  N/A
   */
  virtual void addInitializer(const ::internal::tflite::operand::Index &ind,
                              const Initializer &initializer) = 0;
  /**
   * @brief  Add Stage lambda with IAllocationContext and IExecutionBuilder params
   * @param [in] stage Stage to add
   * @return  N/A
   */
  virtual void addStage(const Stage &stage) = 0;
};

//
// ActivationBuilder
//
class ActivationBuilder
{
public:
  ActivationBuilder(IExecutionBuilder &builder) : _builder(builder)
  {
    // DO NOTHING
  }

private:
  void appendReLU(::arm_compute::ITensor *tensor);
  void appendReLU6(::arm_compute::ITensor *tensor);
  void appendReLU1(::arm_compute::ITensor *tensor);

public:
  void append(FuseCode code, ::arm_compute::ITensor *tensor);

private:
  IExecutionBuilder &_builder;
};

void ActivationBuilder::appendReLU(::arm_compute::ITensor *ifm_alloc)
{
  const ::arm_compute::ActivationLayerInfo act_info{
      ::arm_compute::ActivationLayerInfo::ActivationFunction::RELU};

  if (::internal::arm_compute::isGpuMode())
  {
    auto fn = std::make_unique<::arm_compute::CLActivationLayer>();

    fn->configure(CAST_CL(ifm_alloc), nullptr, act_info);

    _builder.append("ReLU", std::move(fn));
  }
  else
  {
    auto fn = std::make_unique<::arm_compute::NEActivationLayer>();

    fn->configure(ifm_alloc, nullptr, act_info);

    _builder.append("ReLU", std::move(fn));
  }
}

void ActivationBuilder::appendReLU1(::arm_compute::ITensor *ifm_alloc)
{
  const ::arm_compute::ActivationLayerInfo act_info{
      ::arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 1.0f, -1.0f};

  if (::internal::arm_compute::isGpuMode())
  {
    auto fn = std::make_unique<::arm_compute::CLActivationLayer>();

    fn->configure(CAST_CL(ifm_alloc), nullptr, act_info);

    _builder.append("ReLU1", std::move(fn));
  }
  else
  {
    auto fn = std::make_unique<::arm_compute::NEActivationLayer>();

    fn->configure(ifm_alloc, nullptr, act_info);

    _builder.append("ReLU1", std::move(fn));
  }
}

void ActivationBuilder::appendReLU6(::arm_compute::ITensor *ifm_alloc)
{
  const ::arm_compute::ActivationLayerInfo act_info{
      ::arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.0f, 0.0f};

  if (::internal::arm_compute::isGpuMode())
  {
    auto fn = std::make_unique<::arm_compute::CLActivationLayer>();

    fn->configure(CAST_CL(ifm_alloc), nullptr, act_info);

    _builder.append("ReLU6", std::move(fn));
  }
  else
  {
    auto fn = std::make_unique<::arm_compute::NEActivationLayer>();

    fn->configure(ifm_alloc, nullptr, act_info);

    _builder.append("ReLU6", std::move(fn));
  }
}

void ActivationBuilder::append(FuseCode code, ::arm_compute::ITensor *ifm_alloc)
{
  switch (code)
  {
    case ANEURALNETWORKS_FUSED_NONE:
    {
      // DO NOTHING
      break;
    }
    case ANEURALNETWORKS_FUSED_RELU:
    {
      appendReLU(ifm_alloc);
      break;
    }
    case ANEURALNETWORKS_FUSED_RELU1:
    {
      appendReLU1(ifm_alloc);
      break;
    }
    case ANEURALNETWORKS_FUSED_RELU6:
    {
      appendReLU6(ifm_alloc);
      break;
    }
    default:
    {
      throw std::runtime_error("Not supported, yet");
    }
  }
}

class Planner : public ::internal::tflite::op::NodeVisitor
{
public:
  Planner(const ::internal::tflite::operand::Set &ctx, IPlanBuilder &builder)
      : _ctx{ctx}, _builder{builder}
  {
    // DO NOTHING
  }

public:
  void visit(const ::internal::tflite::op::Add::Node &node) override;
  void visit(const ::internal::tflite::op::Sub::Node &node) override;
  void visit(const ::internal::tflite::op::Mul::Node &node) override;
  void visit(const ::internal::tflite::op::Div::Node &node) override;
  void visit(const ::internal::tflite::op::Conv2D::Implicit::Node &node) override;
  void visit(const ::internal::tflite::op::Conv2D::Explicit::Node &node) override;
  void visit(const ::internal::tflite::op::DepthwiseConv2D::Implicit::Node &node) override;
  void visit(const ::internal::tflite::op::DepthwiseConv2D::Explicit::Node &node) override;
  void visit(const ::internal::tflite::op::Dequantize::Node &node) override;
  void visit(const ::internal::tflite::op::MaxPool2D::Implicit::Node &node) override;
  void visit(const ::internal::tflite::op::MaxPool2D::Explicit::Node &node) override;
  void visit(const ::internal::tflite::op::AvgPool2D::Implicit::Node &node) override;
  void visit(const ::internal::tflite::op::AvgPool2D::Explicit::Node &node) override;
  void visit(const ::internal::tflite::op::Concat::Node &node) override;
  void visit(const ::internal::tflite::op::FullyConnected::Node &node) override;
  void visit(const ::internal::tflite::op::ResizeBilinear::Node &node) override;
  void visit(const ::internal::tflite::op::Reshape::Node &node) override;
  void visit(const ::internal::tflite::op::Squeeze::Node &node) override;
  void visit(const ::internal::tflite::op::Softmax::Node &node) override;
  void visit(const ::internal::tflite::op::StridedSlice::Node &node) override;
  void visit(const ::internal::tflite::op::ReduceMax::Node &node) override;
  void visit(const ::internal::tflite::op::ReduceMin::Node &node) override;
  void visit(const ::internal::tflite::op::Cast::Node &node) override;
  void visit(const ::internal::tflite::op::TopKV2::Node &node) override;
  void visit(const ::internal::tflite::op::Gather::Node &node) override;
  void visit(const ::internal::tflite::op::PReLU::Node &node) override;
  void visit(const ::internal::tflite::op::ReLU::Node &node) override;
  void visit(const ::internal::tflite::op::ReLU1::Node &node) override;
  void visit(const ::internal::tflite::op::ReLU6::Node &node) override;
  void visit(const ::internal::tflite::op::Tanh::Node &node) override;
  void visit(const ::internal::tflite::op::Logistic::Node &node) override;
  void visit(const ::internal::tflite::op::Mean::Node &node) override;
  void visit(const ::internal::tflite::op::RNN::Node &node) override;
  void visit(const ::internal::tflite::op::Transpose::Node &node) override;
  void visit(const ::internal::tflite::op::LSTM::Node &node) override;
  void visit(const ::internal::tflite::op::Floor::Node &node) override;
  void visit(const ::internal::tflite::op::Split::Node &node) override;
  void visit(const ::internal::tflite::op::ArgMax::Node &node) override;
  void visit(const ::internal::tflite::op::RSQRT::Node &node) override;
  void visit(const ::internal::tflite::op::SQRT::Node &node) override;
  void visit(const ::internal::tflite::op::Pad::Node &node) override;
  void visit(const ::internal::tflite::op::SpaceToDepth::Node &node) override;
  void visit(const ::internal::tflite::op::SpaceToBatchND::Node &node) override;
  void visit(const ::internal::tflite::op::BatchToSpaceNd::Node &node) override;
  void visit(const ::internal::tflite::op::L2Pool2D::Implicit::Node &node) override;
  void visit(const ::internal::tflite::op::L2Pool2D::Explicit::Node &node) override;
  void visit(const ::internal::tflite::op::EmbeddingLookup::Node &node) override;
  void visit(const ::internal::tflite::op::HashtableLookup::Node &node) override;
  void visit(const ::internal::tflite::op::L2Normalization::Node &node) override;
  void visit(const ::internal::tflite::op::SquaredDifference::Node &node) override;
  void visit(const ::internal::tflite::op::LocalResponseNormalization::Node &node) override;
  void visit(const ::internal::tflite::op::DepthToSpace::Node &node) override;
  void visit(const ::internal::tflite::op::Unpack::Node &node) override;
  void visit(const ::internal::tflite::op::Neg::Node &node) override;
  void visit(const ::internal::tflite::op::Exp::Node &node) override;
  void visit(const ::internal::tflite::op::ReduceSum::Node &node) override;
  void visit(const ::internal::tflite::op::Equal::Node &node) override;
  void visit(const ::internal::tflite::op::TransposeConv::Node &node) override;
  void visit(const ::internal::tflite::op::Pack::Node &node) override;
  void visit(const ::internal::tflite::op::Abs::Node &node) override;
  void visit(const ::internal::tflite::op::NotEqual::Node &node) override;
  void visit(const ::internal::tflite::op::LogicalAnd::Node &node) override;
  void visit(const ::internal::tflite::op::LogicalNot::Node &node) override;
  void visit(const ::internal::tflite::op::LogicalOr::Node &node) override;

private:
  const ::internal::tflite::operand::Set &_ctx;
  IPlanBuilder &_builder;
};

void Planner::visit(const ::internal::tflite::op::Add::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index lhs_index{node.param().lhs_index};
  const ::internal::tflite::operand::Index rhs_index{node.param().rhs_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));

  if (!(_ctx.at(lhs_index).shape() == _ctx.at(rhs_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(lhs_index).shape().rank(), _ctx.at(rhs_index).shape().rank());
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(lhs_index).shape())
        .extendRank(broadcast_rank);
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(rhs_index).shape())
        .extendRank(broadcast_rank);
  }
  _builder.addShapeConstr(
      lhs_index, asTensorInfo(asTensorShape(_ctx.at(lhs_index).shape()), _ctx.at(lhs_index).type(),
                              _ctx.at(lhs_index).scale(), _ctx.at(lhs_index).zeroPoint()));
  _builder.addShapeConstr(
      rhs_index, asTensorInfo(asTensorShape(_ctx.at(rhs_index).shape()), _ctx.at(rhs_index).type(),
                              _ctx.at(rhs_index).scale(), _ctx.at(rhs_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int lhs_index;
    int rhs_index;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.lhs_index = lhs_index.asInt();
  param.rhs_index = rhs_index.asInt();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  const auto lhs_shape = _ctx.at(lhs_index).shape();
  const auto rhs_shape = _ctx.at(rhs_index).shape();
  auto stage = [param, lhs_shape, rhs_shape](const IAllocationContext &ctx,
                                             IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto lhs_alloc = ctx.at(::internal::tflite::operand::Index{param.lhs_index});
    auto rhs_alloc = ctx.at(::internal::tflite::operand::Index{param.rhs_index});

    std::unique_ptr<::arm_compute::IFunction> fn;

    {
      if (::internal::arm_compute::isGpuMode())
      {
        auto l = std::make_unique<::arm_compute::CLArithmeticAddition>();

        // TODO Decide ConvertPolicy (WARP? SATURATE?) according to NN API specification
        l->configure(CAST_CL(lhs_alloc), CAST_CL(rhs_alloc), CAST_CL(ofm_alloc),
                     ::arm_compute::ConvertPolicy::SATURATE);

        fn = std::move(l);
      }
      else // NEON
      {
        auto l = std::make_unique<::arm_compute::NEArithmeticAddition>();

        // TODO Decide ConvertPolicy (WARP? SATURATE?) according to NN API specification
        l->configure(lhs_alloc, rhs_alloc, ofm_alloc, ::arm_compute::ConvertPolicy::SATURATE);

        fn = std::move(l);
      }
    }

    builder.append("Add", std::move(fn));

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Sub::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index lhs_index{node.param().lhs_index};
  const ::internal::tflite::operand::Index rhs_index{node.param().rhs_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));

  if (!(_ctx.at(lhs_index).shape() == _ctx.at(rhs_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(lhs_index).shape().rank(), _ctx.at(rhs_index).shape().rank());
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(lhs_index).shape())
        .extendRank(broadcast_rank);
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(rhs_index).shape())
        .extendRank(broadcast_rank);
  }
  _builder.addShapeConstr(
      lhs_index, asTensorInfo(asTensorShape(_ctx.at(lhs_index).shape()), _ctx.at(lhs_index).type(),
                              _ctx.at(lhs_index).scale(), _ctx.at(lhs_index).zeroPoint()));
  _builder.addShapeConstr(
      rhs_index, asTensorInfo(asTensorShape(_ctx.at(rhs_index).shape()), _ctx.at(rhs_index).type(),
                              _ctx.at(rhs_index).scale(), _ctx.at(rhs_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int lhs_index;
    int rhs_index;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.lhs_index = lhs_index.asInt();
  param.rhs_index = rhs_index.asInt();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto lhs_alloc = ctx.at(::internal::tflite::operand::Index{param.lhs_index});
    auto rhs_alloc = ctx.at(::internal::tflite::operand::Index{param.rhs_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLArithmeticSubtraction>();

      // TODO Decide ConvertPolicy (WARP? SATURATE?) according to NN API specification
      fn->configure(CAST_CL(lhs_alloc), CAST_CL(rhs_alloc), CAST_CL(ofm_alloc),
                    ::arm_compute::ConvertPolicy::SATURATE);

      builder.append("Sub", std::move(fn));
    }
    else // NEON
    {
      auto fn = std::make_unique<::arm_compute::NEArithmeticSubtraction>();

      // TODO Decide ConvertPolicy (WARP? SATURATE?) according to NN API specification
      fn->configure(lhs_alloc, rhs_alloc, ofm_alloc, ::arm_compute::ConvertPolicy::SATURATE);

      builder.append("Sub", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Mul::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index lhs_index{node.param().lhs_index};
  const ::internal::tflite::operand::Index rhs_index{node.param().rhs_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  if (_ctx.at(ofm_index).scale() > 0)
  {
    assert(_ctx.at(ofm_index).scale() > _ctx.at(lhs_index).scale() * _ctx.at(rhs_index).scale());
  }
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));

  if (!(_ctx.at(lhs_index).shape() == _ctx.at(rhs_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(lhs_index).shape().rank(), _ctx.at(rhs_index).shape().rank());
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(lhs_index).shape())
        .extendRank(broadcast_rank);
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(rhs_index).shape())
        .extendRank(broadcast_rank);
  }
  _builder.addShapeConstr(
      lhs_index, asTensorInfo(asTensorShape(_ctx.at(lhs_index).shape()), _ctx.at(lhs_index).type(),
                              _ctx.at(lhs_index).scale(), _ctx.at(lhs_index).zeroPoint()));
  _builder.addShapeConstr(
      rhs_index, asTensorInfo(asTensorShape(_ctx.at(rhs_index).shape()), _ctx.at(rhs_index).type(),
                              _ctx.at(rhs_index).scale(), _ctx.at(rhs_index).zeroPoint()));

  struct Param
  {
    int ofm_index;
    int lhs_index;
    int rhs_index;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.lhs_index = lhs_index.asInt();
  param.rhs_index = rhs_index.asInt();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {

    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto lhs_input_alloc = ctx.at(::internal::tflite::operand::Index{param.lhs_index});
    auto rhs_input_alloc = ctx.at(::internal::tflite::operand::Index{param.rhs_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLPixelWiseMultiplication>();

      fn->configure(CAST_CL(lhs_input_alloc), CAST_CL(rhs_input_alloc), CAST_CL(output_alloc),
                    1.0, // scale
                    arm_compute::ConvertPolicy::SATURATE,
                    arm_compute::RoundingPolicy::TO_NEAREST_EVEN);

      builder.append("Mul", std::move(fn));
    }
    else // NEON
    {
      auto fn = std::make_unique<::arm_compute::NEPixelWiseMultiplication>();

      fn->configure(lhs_input_alloc, rhs_input_alloc, output_alloc,
                    1.0, // scale
                    arm_compute::ConvertPolicy::SATURATE, arm_compute::RoundingPolicy::TO_ZERO);

      builder.append("Mul", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, output_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Div::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};

  const ::internal::tflite::operand::Index lhs_index{node.param().lhs_index};
  const ::internal::tflite::operand::Index rhs_index{node.param().rhs_index};

  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));

  if (!(_ctx.at(lhs_index).shape() == _ctx.at(rhs_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(lhs_index).shape().rank(), _ctx.at(rhs_index).shape().rank());
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(lhs_index).shape())
        .extendRank(broadcast_rank);
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(rhs_index).shape())
        .extendRank(broadcast_rank);
  }

  _builder.addShapeConstr(
      lhs_index, asTensorInfo(asTensorShape(_ctx.at(lhs_index).shape()), _ctx.at(lhs_index).type(),
                              _ctx.at(lhs_index).scale(), _ctx.at(lhs_index).zeroPoint()));
  _builder.addShapeConstr(
      rhs_index, asTensorInfo(asTensorShape(_ctx.at(rhs_index).shape()), _ctx.at(rhs_index).type(),
                              _ctx.at(rhs_index).scale(), _ctx.at(rhs_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int lhs_index;
    int rhs_index;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.lhs_index = lhs_index.asInt();
  param.rhs_index = rhs_index.asInt();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto lhs_alloc = ctx.at(::internal::tflite::operand::Index{param.lhs_index});
    auto rhs_alloc = ctx.at(::internal::tflite::operand::Index{param.rhs_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLArithmeticDivision>();

      fn->configure(CAST_CL(lhs_alloc), CAST_CL(rhs_alloc), CAST_CL(ofm_alloc));

      builder.append("Div", std::move(fn));
    }
    else
      throw std::runtime_error("Not supported, yet");

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Conv2D::Implicit::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};

  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index ker_index{node.param().ker_index};
  const ::internal::tflite::operand::Index bias_index{node.param().bias_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_index{node.param().padding_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();
  const auto ker_shape = _ctx.at(ker_index).shape().asKernel();
  const auto bias_size = _ctx.at(bias_index).shape().asVector();

  const PaddingCode padding_type =
      static_cast<PaddingCode>(_ctx.at(padding_index).asScalar<int32_t>());

  Stride stride;

  stride.vertical = _ctx.at(vstride_index).asScalar<int32_t>();
  stride.horizontal = _ctx.at(hstride_index).asScalar<int32_t>();

  assert((ANEURALNETWORKS_PADDING_SAME == padding_type) ||
         (ANEURALNETWORKS_PADDING_VALID == padding_type));

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));
  _builder.addShapeConstr(
      ker_index, asTensorInfo(asTensorShape(_ctx.at(ker_index).shape()), _ctx.at(ker_index).type(),
                              _ctx.at(ker_index).scale(), _ctx.at(ker_index).zeroPoint()));
  _builder.addShapeConstr(bias_index,
                          asTensorInfo(asTensorShape(_ctx.at(bias_index).shape()),
                                       _ctx.at(bias_index).type(), _ctx.at(bias_index).scale(),
                                       _ctx.at(bias_index).zeroPoint()));

  // Set initializer for kernel
  {
    auto ker_base = _ctx.at(ker_index).data().base();
    auto ker_size = _ctx.at(ker_index).data().size();
    auto ker_type = _ctx.at(ker_index).type();

    switch (ker_type)
    {
      case ANEURALNETWORKS_TENSOR_FLOAT32:
      {
        auto initializer = std::bind(initKernelTensor<float>, _1, ker_shape, ker_base, ker_size);
        _builder.addInitializer(ker_index, initializer);
        break;
      }
      case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
      {
        auto initializer = std::bind(initKernelTensor<uint8_t>, _1, ker_shape, ker_base, ker_size);
        _builder.addInitializer(ker_index, initializer);
        break;
      }
      default:
      {
        throw std::runtime_error("Not supported");
      }
    }
  }

  // Set initializer for bias
  {
    auto bias_base = _ctx.at(bias_index).data().base();
    auto bias_type = _ctx.at(bias_index).type();

    switch (bias_type)
    {
      case ANEURALNETWORKS_TENSOR_FLOAT32:
      {
        auto initializer = std::bind(initVectorTensor<float>, _1, bias_base, bias_size);
        _builder.addInitializer(bias_index, initializer);
        break;
      }
      case ANEURALNETWORKS_TENSOR_INT32:
      {
        auto initializer = std::bind(initVectorTensor<int32_t>, _1, bias_base, bias_size);
        _builder.addInitializer(bias_index, initializer);
        break;
      }
      default:
      {
        throw std::runtime_error("Not supported");
      }
    }
  }

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    int ker_index;
    int bias_index;

    Padding padding;
    Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.ker_index = ker_index.asInt();
  param.bias_index = bias_index.asInt();

  param.stride = stride;
  param.padding = (padding_type == ANEURALNETWORKS_PADDING_SAME)
                      ? same_padding(ifm_shape, ofm_shape, stride, ker_shape.W, ker_shape.H)
                      : valid_padding();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});
    auto ker_alloc = ctx.at(::internal::tflite::operand::Index{param.ker_index});
    auto bias_alloc = ctx.at(::internal::tflite::operand::Index{param.bias_index});

    const auto conv_info = asPadStrideInfo(param.padding, param.stride);
    const auto fused_act = asActInfo(param.activation);

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::CLConvolutionLayer> fn{new ::arm_compute::CLConvolutionLayer};

      // To pass the fused_act parameter, it calls the WeightsInfo() and Size2D(1U, 1U) (dilation)
      // functions like the default parameter.
      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ker_alloc), CAST_CL(bias_alloc), CAST_CL(ofm_alloc),
                    conv_info, ::arm_compute::WeightsInfo(), ::arm_compute::Size2D(1U, 1U),
                    fused_act);

      builder.append("Conv2D", std::move(fn));
    }
    else
    {
      std::unique_ptr<::arm_compute::NEConvolutionLayer> fn{new ::arm_compute::NEConvolutionLayer};

      // To pass the fused_act parameter, it calls the WeightsInfo() and Size2D(1U, 1U) (dilation)
      // functions like the default parameter.
      fn->configure(ifm_alloc, ker_alloc, bias_alloc, ofm_alloc, conv_info,
                    ::arm_compute::WeightsInfo(), ::arm_compute::Size2D(1U, 1U), fused_act);

      builder.append("Conv2D", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Conv2D::Explicit::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};

  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index ker_index{node.param().ker_index};
  const ::internal::tflite::operand::Index bias_index{node.param().bias_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_left_index{node.param().padding_left_index};
  const ::internal::tflite::operand::Index padding_right_index{node.param().padding_right_index};
  const ::internal::tflite::operand::Index padding_top_index{node.param().padding_top_index};
  const ::internal::tflite::operand::Index padding_bottom_index{node.param().padding_bottom_index};

  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  const int32_t padding_left = _ctx.at(padding_left_index).asScalar<int32_t>();
  const int32_t padding_right = _ctx.at(padding_right_index).asScalar<int32_t>();
  const int32_t padding_top = _ctx.at(padding_top_index).asScalar<int32_t>();
  const int32_t padding_bottom = _ctx.at(padding_bottom_index).asScalar<int32_t>();

  Stride stride;

  stride.vertical = _ctx.at(vstride_index).asScalar<int32_t>();
  stride.horizontal = _ctx.at(hstride_index).asScalar<int32_t>();

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));
  _builder.addShapeConstr(
      ker_index, asTensorInfo(asTensorShape(_ctx.at(ker_index).shape()), _ctx.at(ker_index).type(),
                              _ctx.at(ker_index).scale(), _ctx.at(ker_index).zeroPoint()));
  _builder.addShapeConstr(bias_index,
                          asTensorInfo(asTensorShape(_ctx.at(bias_index).shape()),
                                       _ctx.at(bias_index).type(), _ctx.at(bias_index).scale(),
                                       _ctx.at(bias_index).zeroPoint()));

  // Set initializer for kernel
  // Workaround for https://github.sec.samsung.net/STAR/nnfw/issues/2319
  if (_ctx.at(ker_index).hasData())
  {
    const auto ker_shape = _ctx.at(ker_index).shape().asKernel();
    auto ker_base = _ctx.at(ker_index).data().base();
    auto ker_size = _ctx.at(ker_index).data().size();
    auto ker_type = _ctx.at(ker_index).type();

    switch (ker_type)
    {
      case ANEURALNETWORKS_TENSOR_FLOAT32:
      {
        auto initializer = std::bind(initKernelTensor<float>, _1, ker_shape, ker_base, ker_size);
        _builder.addInitializer(ker_index, initializer);
        break;
      }
      case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
      {
        auto initializer = std::bind(initKernelTensor<uint8_t>, _1, ker_shape, ker_base, ker_size);
        _builder.addInitializer(ker_index, initializer);
        break;
      }
      default:
      {
        throw std::runtime_error("Not supported");
      }
    }
  }

  // Set initializer for bias
  // See above comment.
  if (_ctx.at(bias_index).hasData())
  {
    const auto bias_size = _ctx.at(bias_index).shape().asVector();
    auto bias_base = _ctx.at(bias_index).data().base();
    auto bias_type = _ctx.at(bias_index).type();

    switch (bias_type)
    {
      case ANEURALNETWORKS_TENSOR_FLOAT32:
      {
        auto initializer = std::bind(initVectorTensor<float>, _1, bias_base, bias_size);
        _builder.addInitializer(bias_index, initializer);
        break;
      }
      case ANEURALNETWORKS_TENSOR_INT32:
      {
        auto initializer = std::bind(initVectorTensor<int32_t>, _1, bias_base, bias_size);
        _builder.addInitializer(bias_index, initializer);
        break;
      }
      default:
      {
        throw std::runtime_error("Not supported");
      }
    }
  }

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    int ker_index;
    int bias_index;

    Padding padding;
    Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.ker_index = ker_index.asInt();
  param.bias_index = bias_index.asInt();

  param.stride = stride;

  param.padding.left = padding_left;
  param.padding.right = padding_right;
  param.padding.top = padding_top;
  param.padding.bottom = padding_bottom;

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});
    auto ker_alloc = ctx.at(::internal::tflite::operand::Index{param.ker_index});
    auto bias_alloc = ctx.at(::internal::tflite::operand::Index{param.bias_index});

    const auto conv_info = asPadStrideInfo(param.padding, param.stride);
    const auto fused_act = asActInfo(param.activation);

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::CLConvolutionLayer> fn{new ::arm_compute::CLConvolutionLayer};

      // To pass the fused_act parameter, it calls the WeightsInfo() and Size2D(1U, 1U) (dilation)
      // functions like the default parameter.
      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ker_alloc), CAST_CL(bias_alloc), CAST_CL(ofm_alloc),
                    conv_info, ::arm_compute::WeightsInfo(), ::arm_compute::Size2D(1U, 1U),
                    fused_act);

      builder.append("Conv2D", std::move(fn));
    }
    else
    {
      std::unique_ptr<::arm_compute::NEConvolutionLayer> fn{new ::arm_compute::NEConvolutionLayer};

      // To pass the fused_act parameter, it calls the WeightsInfo() and Size2D(1U, 1U) (dilation)
      // functions like the default parameter.
      fn->configure(ifm_alloc, ker_alloc, bias_alloc, ofm_alloc, conv_info,
                    ::arm_compute::WeightsInfo(), ::arm_compute::Size2D(1U, 1U), fused_act);

      builder.append("Conv2D", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::DepthwiseConv2D::Implicit::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};

  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index ker_index{node.param().ker_index};
  const ::internal::tflite::operand::Index bias_index{node.param().bias_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_index{node.param().padding_index};
  const ::internal::tflite::operand::Index multiplier_index{node.param().multiplier_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();

  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();
  const auto ker_shape = _ctx.at(ker_index).shape().asFeature();
  const auto bias_size = _ctx.at(bias_index).shape().asVector();

  auto multiplier = _ctx.at(multiplier_index).asScalar<int>();

  assert(ker_shape.C == bias_size);
  assert(ker_shape.C == ifm_shape.C * multiplier);

  const PaddingCode padding_type =
      static_cast<PaddingCode>(_ctx.at(padding_index).asScalar<int32_t>());

  Stride stride;

  stride.vertical = _ctx.at(vstride_index).asScalar<int32_t>();
  stride.horizontal = _ctx.at(hstride_index).asScalar<int32_t>();

  assert((ANEURALNETWORKS_PADDING_SAME == padding_type) ||
         (ANEURALNETWORKS_PADDING_VALID == padding_type));

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));
  // NOTE DepthwiseConv2D kernel is of shape [1, KER_W, KER_H, IFM_C * MULTIPLIER]
  _builder.addShapeConstr(
      ker_index, asTensorInfo(asTensorShape(_ctx.at(ker_index).shape()), _ctx.at(ker_index).type(),
                              _ctx.at(ker_index).scale(), _ctx.at(ker_index).zeroPoint()));
  _builder.addShapeConstr(bias_index,
                          asTensorInfo(asTensorShape(_ctx.at(bias_index).shape()),
                                       _ctx.at(bias_index).type(), _ctx.at(bias_index).scale(),
                                       _ctx.at(bias_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    int ker_index;
    int bias_index;

    Padding padding;
    Stride stride;

    int multipler;
    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.ker_index = ker_index.asInt();
  param.bias_index = bias_index.asInt();

  param.stride = stride;
  param.padding = (padding_type == ANEURALNETWORKS_PADDING_SAME)
                      ? same_padding(ifm_shape, ofm_shape, stride, ker_shape.W, ker_shape.H)
                      : valid_padding();

  param.multipler = multiplier;
  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  VERBOSE(DepthwiseConv2D) << "OFM_C: " << ofm_shape.C << std::endl;
  VERBOSE(DepthwiseConv2D) << "OFM_H: " << ofm_shape.H << std::endl;
  VERBOSE(DepthwiseConv2D) << "OFM_W: " << ofm_shape.W << std::endl;

  VERBOSE(DepthwiseConv2D) << "IFM_C: " << ifm_shape.C << std::endl;
  VERBOSE(DepthwiseConv2D) << "IFM_H: " << ifm_shape.H << std::endl;
  VERBOSE(DepthwiseConv2D) << "IFM_W: " << ifm_shape.W << std::endl;

  VERBOSE(DepthwiseConv2D) << "KER_C: " << ker_shape.C << std::endl;
  VERBOSE(DepthwiseConv2D) << "KER_H: " << ker_shape.H << std::endl;
  VERBOSE(DepthwiseConv2D) << "KER_W: " << ker_shape.W << std::endl;

  VERBOSE(DepthwiseConv2D) << "STRIDE_H: " << param.stride.vertical << std::endl;
  VERBOSE(DepthwiseConv2D) << "STRIDE_W: " << param.stride.horizontal << std::endl;

  VERBOSE(DepthwiseConv2D) << "ACTIVATION: " << param.activation << std::endl;

  VERBOSE(DepthwiseConv2D) << "PAD(T): " << param.padding.top << std::endl;
  VERBOSE(DepthwiseConv2D) << "PAD(B): " << param.padding.bottom << std::endl;
  VERBOSE(DepthwiseConv2D) << "PAD(L): " << param.padding.left << std::endl;
  VERBOSE(DepthwiseConv2D) << "PAD(R): " << param.padding.right << std::endl;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});
    auto ker_alloc = ctx.at(::internal::tflite::operand::Index{param.ker_index});
    auto bias_alloc = ctx.at(::internal::tflite::operand::Index{param.bias_index});

    const auto conv_info = asPadStrideInfo(param.padding, param.stride);

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLDepthwiseConvolutionLayer>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ker_alloc), CAST_CL(bias_alloc), CAST_CL(ofm_alloc),
                    conv_info, param.multipler);

      builder.append("DepthwiseConv2D", std::move(fn));
    }
    else
    {
      auto fn = std::make_unique<::arm_compute::NEDepthwiseConvolutionLayer>();

      fn->configure(ifm_alloc, ker_alloc, bias_alloc, ofm_alloc, conv_info, param.multipler);

      builder.append("DepthwiseConv2D", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::DepthwiseConv2D::Explicit::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};

  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index ker_index{node.param().ker_index};
  const ::internal::tflite::operand::Index bias_index{node.param().bias_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_left_index{node.param().padding_left_index};
  const ::internal::tflite::operand::Index padding_right_index{node.param().padding_right_index};
  const ::internal::tflite::operand::Index padding_top_index{node.param().padding_top_index};
  const ::internal::tflite::operand::Index padding_bottom_index{node.param().padding_bottom_index};

  const ::internal::tflite::operand::Index multiplier_index{node.param().multiplier_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();

  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();
  const auto ker_shape = _ctx.at(ker_index).shape().asFeature();
  const auto bias_size = _ctx.at(bias_index).shape().asVector();

  auto multiplier = _ctx.at(multiplier_index).asScalar<int>();

  assert(ker_shape.C == bias_size);
  assert(ker_shape.C == ifm_shape.C * multiplier);

  const int32_t padding_left = _ctx.at(padding_left_index).asScalar<int32_t>();
  const int32_t padding_right = _ctx.at(padding_right_index).asScalar<int32_t>();
  const int32_t padding_top = _ctx.at(padding_top_index).asScalar<int32_t>();
  const int32_t padding_bottom = _ctx.at(padding_bottom_index).asScalar<int32_t>();

  Stride stride;

  stride.vertical = _ctx.at(vstride_index).asScalar<int32_t>();
  stride.horizontal = _ctx.at(hstride_index).asScalar<int32_t>();

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));
  // NOTE DepthwiseConv2D kernel is of shape [1, KER_W, KER_H, IFM_C * MULTIPLIER]
  _builder.addShapeConstr(
      ker_index, asTensorInfo(asTensorShape(_ctx.at(ker_index).shape()), _ctx.at(ker_index).type(),
                              _ctx.at(ker_index).scale(), _ctx.at(ker_index).zeroPoint()));
  _builder.addShapeConstr(bias_index,
                          asTensorInfo(asTensorShape(_ctx.at(bias_index).shape()),
                                       _ctx.at(bias_index).type(), _ctx.at(bias_index).scale(),
                                       _ctx.at(bias_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    int ker_index;
    int bias_index;

    Padding padding;
    Stride stride;

    int multipler;
    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.ker_index = ker_index.asInt();
  param.bias_index = bias_index.asInt();

  param.stride = stride;

  param.padding.left = padding_left;
  param.padding.right = padding_right;
  param.padding.top = padding_top;
  param.padding.bottom = padding_bottom;

  param.multipler = multiplier;
  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  VERBOSE(DepthwiseConv2D) << "OFM_C: " << ofm_shape.C << std::endl;
  VERBOSE(DepthwiseConv2D) << "OFM_H: " << ofm_shape.H << std::endl;
  VERBOSE(DepthwiseConv2D) << "OFM_W: " << ofm_shape.W << std::endl;

  VERBOSE(DepthwiseConv2D) << "IFM_C: " << ifm_shape.C << std::endl;
  VERBOSE(DepthwiseConv2D) << "IFM_H: " << ifm_shape.H << std::endl;
  VERBOSE(DepthwiseConv2D) << "IFM_W: " << ifm_shape.W << std::endl;

  VERBOSE(DepthwiseConv2D) << "KER_C: " << ker_shape.C << std::endl;
  VERBOSE(DepthwiseConv2D) << "KER_H: " << ker_shape.H << std::endl;
  VERBOSE(DepthwiseConv2D) << "KER_W: " << ker_shape.W << std::endl;

  VERBOSE(DepthwiseConv2D) << "STRIDE_H: " << param.stride.vertical << std::endl;
  VERBOSE(DepthwiseConv2D) << "STRIDE_W: " << param.stride.horizontal << std::endl;

  VERBOSE(DepthwiseConv2D) << "ACTIVATION: " << param.activation << std::endl;

  VERBOSE(DepthwiseConv2D) << "PAD(T): " << param.padding.top << std::endl;
  VERBOSE(DepthwiseConv2D) << "PAD(B): " << param.padding.bottom << std::endl;
  VERBOSE(DepthwiseConv2D) << "PAD(L): " << param.padding.left << std::endl;
  VERBOSE(DepthwiseConv2D) << "PAD(R): " << param.padding.right << std::endl;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});
    auto ker_alloc = ctx.at(::internal::tflite::operand::Index{param.ker_index});
    auto bias_alloc = ctx.at(::internal::tflite::operand::Index{param.bias_index});

    const auto conv_info = asPadStrideInfo(param.padding, param.stride);

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLDepthwiseConvolutionLayer>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ker_alloc), CAST_CL(bias_alloc), CAST_CL(ofm_alloc),
                    conv_info, param.multipler);

      builder.append("DepthwiseConv2D", std::move(fn));
    }
    else
    {
      auto fn = std::make_unique<::arm_compute::NEDepthwiseConvolutionLayer>();

      fn->configure(ifm_alloc, ker_alloc, bias_alloc, ofm_alloc, conv_info, param.multipler);

      builder.append("DepthwiseConv2D", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Dequantize::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};

  assert(_ctx.at(input_index).shape().rank() >= 0 && _ctx.at(input_index).shape().rank() <= 4);
  assert(_ctx.at(input_index).shape() == _ctx.at(output_index).shape());
  assert(_ctx.at(input_index).type() == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);
  assert(_ctx.at(output_index).type() == ANEURALNETWORKS_TENSOR_FLOAT32);

  // Set Shape Constraints
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(input_index,
                          asTensorInfo(asTensorShape(_ctx.at(input_index).shape()),
                                       _ctx.at(input_index).type(), _ctx.at(input_index).scale(),
                                       _ctx.at(input_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int output_index;
    int input_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});

    std::unique_ptr<::arm_compute::IFunction> fn;

    {
      if (::internal::arm_compute::isGpuMode())
      {
        auto l = std::make_unique<::arm_compute::CLCast>();

        l->configure(CAST_CL(input_alloc), CAST_CL(output_alloc));
        fn = std::move(l);
      }
      else
        throw std::runtime_error("Not supported, yet");
    }

    builder.append("Dequantize", std::move(fn));
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::MaxPool2D::Implicit::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  const ::internal::tflite::operand::Index kh_index{node.param().kh_index};
  const ::internal::tflite::operand::Index kw_index{node.param().kw_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_index{node.param().padding_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();

  const int32_t kh = _ctx.at(kh_index).asScalar<int32_t>();
  const int32_t kw = _ctx.at(kw_index).asScalar<int32_t>();

  const int32_t vstride = _ctx.at(vstride_index).asScalar<int32_t>();
  const int32_t hstride = _ctx.at(hstride_index).asScalar<int32_t>();

  const PaddingCode padding_type =
      static_cast<PaddingCode>(_ctx.at(padding_index).asScalar<int32_t>());

  assert((ANEURALNETWORKS_PADDING_SAME == padding_type) ||
         (ANEURALNETWORKS_PADDING_VALID == padding_type));

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;

    uint32_t kw;
    uint32_t kh;

    Padding padding;
    Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.kh = kh;
  param.kw = kw;

  param.stride.vertical = vstride;
  param.stride.horizontal = hstride;

  param.padding = (padding_type == ANEURALNETWORKS_PADDING_SAME)
                      ? same_padding(ifm_shape, ofm_shape, param.stride, kw, kh)
                      : valid_padding();
  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  VERBOSE(MaxPool2D) << "IFM_H: " << ifm_shape.H << std::endl;
  VERBOSE(MaxPool2D) << "IFM_W: " << ifm_shape.W << std::endl;
  VERBOSE(MaxPool2D) << "OFM_H: " << ofm_shape.H << std::endl;
  VERBOSE(MaxPool2D) << "OFM_W: " << ofm_shape.W << std::endl;
  VERBOSE(MaxPool2D) << "KER_H: " << kh << std::endl;
  VERBOSE(MaxPool2D) << "KER_W: " << kw << std::endl;
  VERBOSE(MaxPool2D) << "STRIDE_H: " << vstride << std::endl;
  VERBOSE(MaxPool2D) << "STRIDE_W: " << hstride << std::endl;
  VERBOSE(MaxPool2D) << "PAD(T): " << param.padding.top << std::endl;
  VERBOSE(MaxPool2D) << "PAD(B): " << param.padding.bottom << std::endl;
  VERBOSE(MaxPool2D) << "PAD(L): " << param.padding.left << std::endl;
  VERBOSE(MaxPool2D) << "PAD(R): " << param.padding.right << std::endl;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    ::arm_compute::PoolingLayerInfo info{::arm_compute::PoolingType::MAX,
                                         ::arm_compute::Size2D{param.kw, param.kh},
                                         asPadStrideInfo(param.padding, param.stride)};

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::CLPoolingLayer> fn{new ::arm_compute::CLPoolingLayer};

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), info);

      builder.append("MaxPool2D", std::move(fn));
    }
    else
    {
      std::unique_ptr<::arm_compute::NEPoolingLayer> fn{new ::arm_compute::NEPoolingLayer};

      fn->configure(ifm_alloc, ofm_alloc, info);

      builder.append("MaxPool2D", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::MaxPool2D::Explicit::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  const ::internal::tflite::operand::Index kh_index{node.param().kh_index};
  const ::internal::tflite::operand::Index kw_index{node.param().kw_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_left_index{node.param().padding_left_index};
  const ::internal::tflite::operand::Index padding_right_index{node.param().padding_right_index};
  const ::internal::tflite::operand::Index padding_top_index{node.param().padding_top_index};
  const ::internal::tflite::operand::Index padding_bottom_index{node.param().padding_bottom_index};

  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();

  const int32_t kh = _ctx.at(kh_index).asScalar<int32_t>();
  const int32_t kw = _ctx.at(kw_index).asScalar<int32_t>();

  const int32_t vstride = _ctx.at(vstride_index).asScalar<int32_t>();
  const int32_t hstride = _ctx.at(hstride_index).asScalar<int32_t>();

  const int32_t padding_left = _ctx.at(padding_left_index).asScalar<int32_t>();
  const int32_t padding_right = _ctx.at(padding_right_index).asScalar<int32_t>();
  const int32_t padding_top = _ctx.at(padding_top_index).asScalar<int32_t>();
  const int32_t padding_bottom = _ctx.at(padding_bottom_index).asScalar<int32_t>();

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;

    uint32_t kw;
    uint32_t kh;

    Padding padding;
    Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.kh = kh;
  param.kw = kw;

  param.stride.vertical = vstride;
  param.stride.horizontal = hstride;

  param.padding.left = padding_left;
  param.padding.right = padding_right;
  param.padding.top = padding_top;
  param.padding.bottom = padding_bottom;

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  VERBOSE(MaxPool2D) << "IFM_H: " << ifm_shape.H << std::endl;
  VERBOSE(MaxPool2D) << "IFM_W: " << ifm_shape.W << std::endl;
  VERBOSE(MaxPool2D) << "OFM_H: " << ofm_shape.H << std::endl;
  VERBOSE(MaxPool2D) << "OFM_W: " << ofm_shape.W << std::endl;
  VERBOSE(MaxPool2D) << "KER_H: " << kh << std::endl;
  VERBOSE(MaxPool2D) << "KER_W: " << kw << std::endl;
  VERBOSE(MaxPool2D) << "STRIDE_H: " << vstride << std::endl;
  VERBOSE(MaxPool2D) << "STRIDE_W: " << hstride << std::endl;
  VERBOSE(MaxPool2D) << "PAD(T): " << param.padding.top << std::endl;
  VERBOSE(MaxPool2D) << "PAD(B): " << param.padding.bottom << std::endl;
  VERBOSE(MaxPool2D) << "PAD(L): " << param.padding.left << std::endl;
  VERBOSE(MaxPool2D) << "PAD(R): " << param.padding.right << std::endl;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    ::arm_compute::PoolingLayerInfo info{::arm_compute::PoolingType::MAX,
                                         ::arm_compute::Size2D{param.kw, param.kh},
                                         asPadStrideInfo(param.padding, param.stride)};

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::CLPoolingLayer> fn{new ::arm_compute::CLPoolingLayer};

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), info);

      builder.append("MaxPool2D", std::move(fn));
    }
    else
    {
      std::unique_ptr<::arm_compute::NEPoolingLayer> fn{new ::arm_compute::NEPoolingLayer};

      fn->configure(ifm_alloc, ofm_alloc, info);

      builder.append("MaxPool2D", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::AvgPool2D::Implicit::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  const ::internal::tflite::operand::Index kh_index{node.param().kh_index};
  const ::internal::tflite::operand::Index kw_index{node.param().kw_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_index{node.param().padding_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();

  const int32_t kh = _ctx.at(kh_index).asScalar<int32_t>();
  const int32_t kw = _ctx.at(kw_index).asScalar<int32_t>();

  const int32_t vstride = _ctx.at(vstride_index).asScalar<int32_t>();
  const int32_t hstride = _ctx.at(hstride_index).asScalar<int32_t>();

  const PaddingCode padding_type =
      static_cast<PaddingCode>(_ctx.at(padding_index).asScalar<int32_t>());

  assert((ANEURALNETWORKS_PADDING_SAME == padding_type) ||
         (ANEURALNETWORKS_PADDING_VALID == padding_type));

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;

    uint32_t kw;
    uint32_t kh;

    Padding padding;
    Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.kh = kh;
  param.kw = kw;

  param.stride.vertical = vstride;
  param.stride.horizontal = hstride;

  param.padding = (padding_type == ANEURALNETWORKS_PADDING_SAME)
                      ? same_padding(ifm_shape, ofm_shape, param.stride, kw, kh)
                      : valid_padding();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  VERBOSE(AvgPool2D) << "IFM_H: " << ifm_shape.H << std::endl;
  VERBOSE(AvgPool2D) << "IFM_W: " << ifm_shape.W << std::endl;
  VERBOSE(AvgPool2D) << "OFM_H: " << ofm_shape.H << std::endl;
  VERBOSE(AvgPool2D) << "OFM_W: " << ofm_shape.W << std::endl;
  VERBOSE(AvgPool2D) << "KER_H: " << kh << std::endl;
  VERBOSE(AvgPool2D) << "KER_W: " << kw << std::endl;
  VERBOSE(AvgPool2D) << "STRIDE_H: " << vstride << std::endl;
  VERBOSE(AvgPool2D) << "STRIDE_W: " << hstride << std::endl;
  VERBOSE(AvgPool2D) << "PAD: " << to_string(padding_type) << std::endl;
  VERBOSE(AvgPool2D) << "PAD(T): " << param.padding.top << std::endl;
  VERBOSE(AvgPool2D) << "PAD(B): " << param.padding.bottom << std::endl;
  VERBOSE(AvgPool2D) << "PAD(L): " << param.padding.left << std::endl;
  VERBOSE(AvgPool2D) << "PAD(R): " << param.padding.right << std::endl;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    ::arm_compute::PoolingLayerInfo info{
        ::arm_compute::PoolingType::AVG, ::arm_compute::Size2D{param.kw, param.kh},
        asPadStrideInfo(param.padding, param.stride), true /* exclude_padding */};

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::CLPoolingLayer> fn{new ::arm_compute::CLPoolingLayer};

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), info);

      builder.append("AvgPool2D", std::move(fn));
    }
    else
    {
      std::unique_ptr<::arm_compute::NEPoolingLayer> fn{new ::arm_compute::NEPoolingLayer};

      fn->configure(ifm_alloc, ofm_alloc, info);

      builder.append("AvgPool2D", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::AvgPool2D::Explicit::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  const ::internal::tflite::operand::Index kh_index{node.param().kh_index};
  const ::internal::tflite::operand::Index kw_index{node.param().kw_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_left_index{node.param().padding_left_index};
  const ::internal::tflite::operand::Index padding_right_index{node.param().padding_right_index};
  const ::internal::tflite::operand::Index padding_top_index{node.param().padding_top_index};
  const ::internal::tflite::operand::Index padding_bottom_index{node.param().padding_bottom_index};

  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();

  const int32_t kh = _ctx.at(kh_index).asScalar<int32_t>();
  const int32_t kw = _ctx.at(kw_index).asScalar<int32_t>();

  const int32_t vstride = _ctx.at(vstride_index).asScalar<int32_t>();
  const int32_t hstride = _ctx.at(hstride_index).asScalar<int32_t>();

  const int32_t padding_left = _ctx.at(padding_left_index).asScalar<int32_t>();
  const int32_t padding_right = _ctx.at(padding_right_index).asScalar<int32_t>();
  const int32_t padding_top = _ctx.at(padding_top_index).asScalar<int32_t>();
  const int32_t padding_bottom = _ctx.at(padding_bottom_index).asScalar<int32_t>();

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;

    uint32_t kw;
    uint32_t kh;

    Padding padding;
    Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.kh = kh;
  param.kw = kw;

  param.stride.vertical = vstride;
  param.stride.horizontal = hstride;

  param.padding.left = padding_left;
  param.padding.right = padding_right;
  param.padding.top = padding_top;
  param.padding.bottom = padding_bottom;

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  VERBOSE(AvgPool2D) << "IFM_H: " << ifm_shape.H << std::endl;
  VERBOSE(AvgPool2D) << "IFM_W: " << ifm_shape.W << std::endl;
  VERBOSE(AvgPool2D) << "OFM_H: " << ofm_shape.H << std::endl;
  VERBOSE(AvgPool2D) << "OFM_W: " << ofm_shape.W << std::endl;
  VERBOSE(AvgPool2D) << "KER_H: " << kh << std::endl;
  VERBOSE(AvgPool2D) << "KER_W: " << kw << std::endl;
  VERBOSE(AvgPool2D) << "STRIDE_H: " << vstride << std::endl;
  VERBOSE(AvgPool2D) << "STRIDE_W: " << hstride << std::endl;
  VERBOSE(AvgPool2D) << "PAD(T): " << param.padding.top << std::endl;
  VERBOSE(AvgPool2D) << "PAD(B): " << param.padding.bottom << std::endl;
  VERBOSE(AvgPool2D) << "PAD(L): " << param.padding.left << std::endl;
  VERBOSE(AvgPool2D) << "PAD(R): " << param.padding.right << std::endl;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    ::arm_compute::PoolingLayerInfo info{
        ::arm_compute::PoolingType::AVG, ::arm_compute::Size2D{param.kw, param.kh},
        asPadStrideInfo(param.padding, param.stride), true /* exclude_padding */};

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::CLPoolingLayer> fn{new ::arm_compute::CLPoolingLayer};

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), info);

      builder.append("AvgPool2D", std::move(fn));
    }
    else
    {
      std::unique_ptr<::arm_compute::NEPoolingLayer> fn{new ::arm_compute::NEPoolingLayer};

      fn->configure(ifm_alloc, ofm_alloc, info);

      builder.append("AvgPool2D", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Concat::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index axis_index{node.param().axis_index};

  // NOTE This implementation assumes that inputs and output are a feature
  const auto ofm_shape = _ctx.at(ofm_index).shape();
  uint32_t input_rank = ofm_shape.rank();
  int32_t axis = _ctx.at(axis_index).asScalar<int32_t>();

  // Handle negative axis
  if (axis < 0)
  {
    axis += input_rank;
  }

  // Set Shape Constraints and TensorInfo (for output)
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));

  // Set Shape Constraints and TensorInfo (for input)
  const uint32_t coord_index = ToARMComputeAxis(input_rank, axis).value();
  uint32_t depth = 0;

  ::arm_compute::Coordinates coordinates;
  coordinates.set_num_dimensions(input_rank);

  for (const auto &index : node.param().ifm_indexes)
  {
    const ::internal::tflite::operand::Index ifm_index{index};
    const auto ifm_shape = _ctx.at(ifm_index).shape();

    coordinates[coord_index] = depth;

    _builder.addSubsumptionConstr(ifm_index, ofm_index, coordinates,
                                  asTensorShape(_ctx.at(ifm_index).shape()), true);

    depth += ifm_shape.dim(axis);
  }

  // NOTE Concat has no actual operation!
  // However, dummy stage is added because profiler assumes every operation make a stage.
  auto stage = [](const IAllocationContext &ctx, IExecutionBuilder &builder) {};
  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::FullyConnected::Node &node)
{
  VERBOSE(FullyConnected) << "Configure FULLY_CONNECTED operation" << std::endl;

  const ::internal::tflite::operand::Index output_index{node.param().output_index};

  const ::internal::tflite::operand::Index input_index{node.param().input_index};
  const ::internal::tflite::operand::Index weight_index{node.param().weight_index};
  const ::internal::tflite::operand::Index bias_index{node.param().bias_index};

  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  assert(_ctx.at(input_index).shape().rank() >= 2);
  assert(_ctx.at(output_index).shape().rank() == 2);
  assert(_ctx.at(weight_index).shape().rank() == 2);
  assert(_ctx.at(bias_index).shape().rank() == 1);

  const auto input_rank = _ctx.at(input_index).shape().rank();
  // TODO Currently we are not handling where the case is that the input's rank is 3.
  // The handling should be added in the future.
  assert(input_rank != 3);

  const auto output_size = _ctx.at(output_index).shape().dim(1);
  assert(_ctx.at(bias_index).shape().dim(0) == output_size);
  assert(_ctx.at(weight_index).shape().dim(0) == output_size);
  const auto batch_size = _ctx.at(output_index).shape().dim(0);
  const auto input_size = _ctx.at(weight_index).shape().dim(1);

  // Check for reshaping input's shape into rank-2
  bool needs_reshape = false;
  internal::tflite::operand::Shape reshape(2);
  if (input_rank == 4)
  {
    nnfw::misc::feature::Shape ifm_shape_feature = _ctx.at(input_index).shape().asFeature();
    auto feature_size =
        ifm_shape_feature.N * ifm_shape_feature.C * ifm_shape_feature.H * ifm_shape_feature.W;
    assert(feature_size == batch_size * input_size);

    _builder.addShapeConstr(input_index,
                            asTensorInfo(asTensorShape(_ctx.at(input_index).shape(), false),
                                         _ctx.at(input_index).type(), _ctx.at(input_index).scale(),
                                         _ctx.at(input_index).zeroPoint()));

    // for reshaping
    needs_reshape = true;
    reshape.dim(0) = batch_size; /* H */
    reshape.dim(1) = input_size; /* W */
  }
  else if (input_rank == 2)
  {
    auto ifm_shape = _ctx.at(input_index).shape();
    nnfw::misc::matrix::Shape ifm_shape_matrix = ifm_shape.asMatrix();
    assert(ifm_shape.dim(0) == batch_size);
    assert(ifm_shape.dim(1) == input_size);

    _builder.addShapeConstr(input_index,
                            asTensorInfo(asTensorShape(_ctx.at(input_index).shape()),
                                         _ctx.at(input_index).type(), _ctx.at(input_index).scale(),
                                         _ctx.at(input_index).zeroPoint()));
  }

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(weight_index,
                          asTensorInfo(asTensorShape(_ctx.at(weight_index).shape()),
                                       _ctx.at(weight_index).type(), _ctx.at(weight_index).scale(),
                                       _ctx.at(weight_index).zeroPoint()));
  _builder.addShapeConstr(bias_index,
                          asTensorInfo(asTensorShape(_ctx.at(bias_index).shape()),
                                       _ctx.at(bias_index).type(), _ctx.at(bias_index).scale(),
                                       _ctx.at(bias_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int output_index;

    int input_index;
    int weight_index;
    int bias_index;

    FuseCode activation;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();
  param.weight_index = weight_index.asInt();
  param.bias_index = bias_index.asInt();

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto stage = [param, needs_reshape, reshape](const IAllocationContext &ctx,
                                               IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});
    auto weight_alloc = ctx.at(::internal::tflite::operand::Index{param.weight_index});
    auto bias_alloc = ctx.at(::internal::tflite::operand::Index{param.bias_index});

    auto fn = std::make_unique<arm_compute::CLFullyConnectedReshapingLayer>();

    fn->configure(CAST_CL(input_alloc), CAST_CL(weight_alloc), CAST_CL(bias_alloc),
                  CAST_CL(output_alloc), needs_reshape, asTensorShape(reshape));

    builder.append("FullyConnected", std::move(fn));

    ActivationBuilder{builder}.append(param.activation, output_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::ResizeBilinear::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};

  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index height_index{node.param().height_index};
  const ::internal::tflite::operand::Index width_index{node.param().width_index};

  // TODO Should move to the place where the operand is handled, if it is possible.
  // Set Shape Constraints
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  struct Param
  {
    int ofm_index;
    int ifm_index;

    int new_height;
    int new_width;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.new_height = _ctx.at(height_index).asScalar<int32_t>();
  param.new_width = _ctx.at(width_index).asScalar<int32_t>();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLScale>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc),
                    ::arm_compute::InterpolationPolicy::BILINEAR,
                    ::arm_compute::BorderMode::REPLICATE, ::arm_compute::PixelValue(0.f),
                    ::arm_compute::SamplingPolicy::TOP_LEFT);

      builder.append("ResizeBilinear", std::move(fn));
    }
    else
      throw std::runtime_error("Not supported, yet");
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Reshape::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};

  auto input_shape = asTensorShape(_ctx.at(input_index).shape(), false);
  auto output_shape = asTensorShape(_ctx.at(output_index).shape(), false);

  assert(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] ==
         output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]);

  // TODO Should move to the place where the operand is handled, if it is possible.
  _builder.addShapeConstr(output_index, asTensorInfo(output_shape, _ctx.at(output_index).type(),
                                                     _ctx.at(output_index).scale(),
                                                     _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(input_index, asTensorInfo(input_shape, _ctx.at(input_index).type(),
                                                    _ctx.at(input_index).scale(),
                                                    _ctx.at(input_index).zeroPoint()));

  struct Param
  {
    int output_index;
    int input_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});

    if (::internal::arm_compute::isGpuMode())
    {
      // GenericReshape first apply NCHW->NHWC permutation, and apply reshape
      auto fn = std::make_unique<GenericReshapeLayer>();

      fn->configure(CAST_CL(input_alloc), CAST_CL(output_alloc));

      builder.append("Reshape", std::move(fn));
    }
    else
    {
      auto fn = std::make_unique<GenericReshapeLayer>();

      fn->configure(input_alloc, output_alloc);

      builder.append("Reshape", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Squeeze::Node &node)
{
  // node.param().dims_index_optional is ignored since output tensor already has squeezed shape
  // by freezer and toco
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};

  // Set Shape Constraints
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(input_index,
                          asTensorInfo(asTensorShape(_ctx.at(input_index).shape()),
                                       _ctx.at(input_index).type(), _ctx.at(input_index).scale(),
                                       _ctx.at(input_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int output_index;
    int input_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLReshapeLayer>();

      fn->configure(CAST_CL(input_alloc), CAST_CL(output_alloc));

      builder.append("Squeeze", std::move(fn));
    }
    else
    {
      auto fn = std::make_unique<::arm_compute::NEReshapeLayer>();

      fn->configure(input_alloc, output_alloc);

      builder.append("Squeeze", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Softmax::Node &node)
{
  VERBOSE(Softmax) << "Configure SOFTMAX operation" << std::endl;

  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};
  const ::internal::tflite::operand::Index scale_index{node.param().scale_index};

  assert(_ctx.at(output_index).shape().rank() == _ctx.at(input_index).shape().rank());
  assert(_ctx.at(scale_index).shape().rank() == 0);

  // TODO Should move to the place where the operand is handled, if it is possible.
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(input_index,
                          asTensorInfo(asTensorShape(_ctx.at(input_index).shape()),
                                       _ctx.at(input_index).type(), _ctx.at(input_index).scale(),
                                       _ctx.at(input_index).zeroPoint()));

  struct Param
  {
    int output_index;
    int input_index;
    float scale;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();
  param.scale = _ctx.at(scale_index).asScalar<float>();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLSoftmaxLayer>();

      fn->configure(CAST_CL(input_alloc), CAST_CL(output_alloc), param.scale);

      builder.append("Softmax", std::move(fn));
    }
    else
    {
      auto fn = std::make_unique<::arm_compute::NESoftmaxLayer>();

      fn->configure(input_alloc, output_alloc, param.scale);

      builder.append("Softmax", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::StridedSlice::Node &node)
{
  VERBOSE(StridedSlice) << "Configure STRIDED_SLICE operation" << std::endl;

  const ::internal::tflite::operand::Index outputData_index{node.param().outputData_index};

  const ::internal::tflite::operand::Index inputData_index{node.param().inputData_index};
  const ::internal::tflite::operand::Index startData_index{node.param().startData_index};
  const ::internal::tflite::operand::Index endData_index{node.param().endData_index};
  const ::internal::tflite::operand::Index stridesData_index{node.param().stridesData_index};
  const ::internal::tflite::operand::Index beginMask_index{node.param().beginMask_index};
  const ::internal::tflite::operand::Index endMask_index{node.param().endMask_index};
  const ::internal::tflite::operand::Index shrinkAxisMask_index{node.param().shrinkAxisMask_index};

  // Set Shape Constraints
  _builder.addShapeConstr(outputData_index,
                          asTensorInfo(asTensorShape(_ctx.at(outputData_index).shape()),
                                       _ctx.at(outputData_index).type(),
                                       _ctx.at(outputData_index).scale(),
                                       _ctx.at(outputData_index).zeroPoint()));
  _builder.addShapeConstr(
      inputData_index,
      asTensorInfo(asTensorShape(_ctx.at(inputData_index).shape()), _ctx.at(inputData_index).type(),
                   _ctx.at(inputData_index).scale(), _ctx.at(inputData_index).zeroPoint()));

  assert(_ctx.at(startData_index).shape().rank() == 1);
  assert(_ctx.at(endData_index).shape().rank() == 1);
  assert(_ctx.at(stridesData_index).shape().rank() == 1);
  _builder.addShapeConstr(
      startData_index,
      asTensorInfo(asTensorShape(_ctx.at(startData_index).shape()), _ctx.at(startData_index).type(),
                   _ctx.at(startData_index).scale(), _ctx.at(startData_index).zeroPoint()));
  _builder.addShapeConstr(endData_index, asTensorInfo(asTensorShape(_ctx.at(endData_index).shape()),
                                                      _ctx.at(endData_index).type(),
                                                      _ctx.at(endData_index).scale(),
                                                      _ctx.at(endData_index).zeroPoint()));
  _builder.addShapeConstr(
      stridesData_index,
      asTensorInfo(asTensorShape(_ctx.at(endData_index).shape()), _ctx.at(stridesData_index).type(),
                   _ctx.at(stridesData_index).scale(), _ctx.at(stridesData_index).zeroPoint()));

  // Set initializers for indices data such as order of inputData
  int input_rank = _ctx.at(inputData_index).shape().rank();
  std::vector<int32_t> starts;
  std::vector<int32_t> ends;
  std::vector<int32_t> strides;
  starts.resize(input_rank, 0);
  ends.resize(input_rank, 0);
  strides.resize(input_rank, 0);
  {
    auto input_shape = _ctx.at(inputData_index).shape();
    auto startData_base = _ctx.at(startData_index).data().base();
    auto endData_base = _ctx.at(endData_index).data().base();
    auto stridesData_base = _ctx.at(stridesData_index).data().base();
    const auto startData_size = _ctx.at(startData_index).shape().asVector();
    const auto endData_size = _ctx.at(endData_index).shape().asVector();
    const auto stridesData_size = _ctx.at(stridesData_index).shape().asVector();

    assert(_ctx.at(startData_index).type() == ANEURALNETWORKS_TENSOR_INT32);
    assert(_ctx.at(endData_index).type() == ANEURALNETWORKS_TENSOR_INT32);
    assert(_ctx.at(stridesData_index).type() == ANEURALNETWORKS_TENSOR_INT32);
    assert(startData_size == input_rank);
    assert(endData_size == input_rank);
    assert(stridesData_size == input_rank);

    assert(startData_base != nullptr);
    for (uint32_t n = 0; n < input_rank; ++n)
    {
      auto axis = ToARMComputeAxis(input_rank, n).value();

      int32_t start_value = *(reinterpret_cast<const int32_t *>(startData_base) + n);
      starts[axis] = start_value;

      int32_t end_value = *(reinterpret_cast<const int32_t *>(endData_base) + n);
      ends[axis] = end_value;

      int32_t strides_value = *(reinterpret_cast<const int32_t *>(stridesData_base) + n);
      strides[axis] = strides_value;
    }
  }

  struct Param
  {
    int32_t outputData_index;
    int32_t inputData_index;

    std::vector<int32_t> starts;
    std::vector<int32_t> ends;
    std::vector<int32_t> strides;

    int32_t beginMask;
    int32_t endMask;
    int32_t shrinkAxisMask;
  };

  Param param;
  param.outputData_index = outputData_index.asInt();
  param.inputData_index = inputData_index.asInt();

  param.starts = starts;
  param.ends = ends;
  param.strides = strides;

  // Set mask bits such as order of inputData
  param.beginMask = _ctx.at(beginMask_index).asReorderBits<int32_t>(input_rank);
  param.endMask = _ctx.at(endMask_index).asReorderBits<int32_t>(input_rank);
  param.shrinkAxisMask = _ctx.at(shrinkAxisMask_index).asReorderBits<int32_t>(input_rank);

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto outputData_alloc = ctx.at(::internal::tflite::operand::Index{param.outputData_index});
    auto inputData_alloc = ctx.at(::internal::tflite::operand::Index{param.inputData_index});

    ::arm_compute::Coordinates starts;
    ::arm_compute::Coordinates ends;
    ::arm_compute::BiStrides strides;
    for (int i = 0; i < param.starts.size(); ++i)
    {
      starts.set(i, param.starts[i]);
      ends.set(i, param.ends[i]);
      strides.set(i, param.strides[i]);
    }

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLStridedSlice>();

      fn->configure(CAST_CL(inputData_alloc), CAST_CL(outputData_alloc), starts, ends, strides,
                    param.beginMask, param.endMask, param.shrinkAxisMask);

      builder.append("StridedSlice", std::move(fn));
    }
    else
      throw std::runtime_error("Not supported, yet");
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::ReduceMin::Node &node)
{
  VERBOSE(ReduceMin) << "Configure REDUCEMIN operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index axis_index{node.param().axis_index};

  auto ifm_shape = _ctx.at(ifm_index).shape();
  auto ofm_shape = _ctx.at(ofm_index).shape();
  auto axis_shape = _ctx.at(axis_index).shape();
  assert(ifm_shape.rank() <= 4);
  assert(ofm_shape.rank() <= ifm_shape.rank());
  assert(_ctx.at(axis_index).hasData());
  assert(axis_shape.rank() == 0 || axis_shape.rank() == 1);

  // NOTE For the 4-dimensions, if the rank of input and output are different, this runtime only
  // supports cases reducing height and width or reducing depth.
  // TODO We have to support all cases of dimensions up to 4.
  // For correct permuting, we have to set output's shape to be equal in dimension position of the
  // input. But the positions of the same dimensions in the input and output may be set differently.
  // For example {2,3,4,5}(input's shape) can be reduced to {3,5}(output's shape). The original
  // output shape should be {1,3,1,5}, but real output shape may be {3,5}. If you simply try to
  // extend it in 4 dimensions, it should be {1,1,3,5}.
  // Even if output shape is changed to {1,3,1,5}, there is another problem. It is that shape of
  // output tensor used at next operation is changed to {1,3,1,5} after this operation even if the
  // next operation is not desired.
  if (ifm_shape.rank() == 4 && ifm_shape.rank() != ofm_shape.rank())
  {
    if (ofm_shape.rank() == 2)
    {
      // Reducing HW
      assert(ifm_shape.dim(0) == ofm_shape.dim(0) && ifm_shape.dim(3) == ofm_shape.dim(1));
    }
    else if (ofm_shape.rank() == 3)
    {
      // Reducing C or
      // (Reducing H and C(ifm and ofm) == 1) or (Reducing W and C(ifm and ofm) == 1)
      assert(ifm_shape.dim(0) == ofm_shape.dim(0) && ifm_shape.dim(1) == ofm_shape.dim(1) &&
                 ifm_shape.dim(2) == ofm_shape.dim(2) ||
             (ifm_shape.dim(0) == ofm_shape.dim(0) &&
              (ifm_shape.dim(1) == ofm_shape.dim(1) || ifm_shape.dim(2) == ofm_shape.dim(1)) &&
              ifm_shape.dim(3) == 1 && ofm_shape.dim(2) == 1));
    }
  }

  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  std::set<uint32_t> axis;
  {
    const auto ifm_rank = ifm_shape.rank();
    switch (axis_shape.rank())
    {
      case 0: // scalar
      {
        int32_t axis_value = _ctx.at(axis_index).asScalar<int32_t>();
        if (axis_value < 0)
        {
          axis_value += ifm_rank;
        }
        axis.insert(ToARMComputeAxis(ifm_rank, axis_value).value());
        break;
      }
      case 1: // vector
      {
        const auto axis_base = _ctx.at(axis_index).data().base();
        const auto axis_size = _ctx.at(axis_index).shape().asVector();

        // If axis's data does not exist as constant values and can be gotten as input data, we have
        // to find a way to infer output shape when sinking output.
        assert(axis_base != nullptr);
        for (uint32_t n = 0; n < axis_size; ++n)
        {
          int32_t axis_value = *(reinterpret_cast<const int32_t *>(axis_base) + n);
          if (axis_value < 0)
          {
            axis_value += ifm_rank;
          }
          axis.insert(ToARMComputeAxis(ifm_rank, axis_value).value());
        }
        break;
      }
      default:
        throw std::runtime_error("Not supported");
        break;
    }
  }

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    std::set<uint32_t> axis;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.axis = axis;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLReduceOperation>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), param.axis,
                    ::arm_compute::ReduceOperation::MIN);

      builder.append("ReduceMin", std::move(fn));
    }
    else
      throw std::runtime_error("Not supported, yet");
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::ReduceMax::Node &node)
{
  VERBOSE(ReduceMax) << "Configure REDUCEMAX operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index axis_index{node.param().axis_index};

  auto ifm_shape = _ctx.at(ifm_index).shape();
  auto ofm_shape = _ctx.at(ofm_index).shape();
  auto axis_shape = _ctx.at(axis_index).shape();
  assert(ifm_shape.rank() <= 4);
  assert(ofm_shape.rank() <= ifm_shape.rank());
  assert(_ctx.at(axis_index).hasData());
  assert(axis_shape.rank() == 0 || axis_shape.rank() == 1);

  // NOTE For the 4-dimensions, if the rank of input and output are different, this runtime only
  // supports cases reducing height and width or reducing depth.
  // TODO We have to support all cases of dimensions up to 4.
  // For correct permuting, we have to set output's shape to be equal in dimension position of the
  // input. But the positions of the same dimensions in the input and output may be set differently.
  // For example {2,3,4,5}(input's shape) can be reduced to {3,5}(output's shape). The original
  // output shape should be {1,3,1,5}, but real output shape may be {3,5}. If you simply try to
  // extend it in 4 dimensions, it should be {1,1,3,5}.
  // Even if output shape is changed to {1,3,1,5}, there is another problem. It is that shape of
  // output tensor used at next operation is changed to {1,3,1,5} after this operation even if the
  // next operation is not desired.
  if (ifm_shape.rank() == 4 && ifm_shape.rank() != ofm_shape.rank())
  {
    if (ofm_shape.rank() == 2)
    {
      // Reducing HW
      assert(ifm_shape.dim(0) == ofm_shape.dim(0) && ifm_shape.dim(3) == ofm_shape.dim(1));
    }
    else if (ofm_shape.rank() == 3)
    {
      // Reducing C or
      // (Reducing H and C(ifm and ofm) == 1) or (Reducing W and C(ifm and ofm) == 1)
      assert(ifm_shape.dim(0) == ofm_shape.dim(0) && ifm_shape.dim(1) == ofm_shape.dim(1) &&
                 ifm_shape.dim(2) == ofm_shape.dim(2) ||
             (ifm_shape.dim(0) == ofm_shape.dim(0) &&
              (ifm_shape.dim(1) == ofm_shape.dim(1) || ifm_shape.dim(2) == ofm_shape.dim(1)) &&
              ifm_shape.dim(3) == 1 && ofm_shape.dim(2) == 1));
    }
  }

  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  std::set<uint32_t> axis;
  {
    const auto ifm_rank = ifm_shape.rank();
    switch (axis_shape.rank())
    {
      case 0: // scalar
      {
        int32_t axis_value = _ctx.at(axis_index).asScalar<int32_t>();
        if (axis_value < 0)
        {
          axis_value += ifm_rank;
        }
        axis.insert(ToARMComputeAxis(ifm_rank, axis_value).value());
        break;
      }
      case 1: // vector
      {
        const auto axis_base = _ctx.at(axis_index).data().base();
        const auto axis_size = _ctx.at(axis_index).shape().asVector();

        // If axis's data does not exist as constant values and can be gotten as input data, we have
        // to find a way to infer output shape when sinking output.
        assert(axis_base != nullptr);
        for (uint32_t n = 0; n < axis_size; ++n)
        {
          int32_t axis_value = *(reinterpret_cast<const int32_t *>(axis_base) + n);
          if (axis_value < 0)
          {
            axis_value += ifm_rank;
          }
          axis.insert(ToARMComputeAxis(ifm_rank, axis_value).value());
        }
        break;
      }
      default:
        throw std::runtime_error("Not supported");
        break;
    }
  }

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    std::set<uint32_t> axis;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.axis = axis;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLReduceOperation>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), param.axis,
                    ::arm_compute::ReduceOperation::MAX);

      builder.append("ReduceMax", std::move(fn));
    }
    else
      throw std::runtime_error("Not supported, yet");
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Cast::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};

  assert(_ctx.at(output_index).shape() == _ctx.at(input_index).shape());

  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(input_index,
                          asTensorInfo(asTensorShape(_ctx.at(input_index).shape()),
                                       _ctx.at(input_index).type(), _ctx.at(input_index).scale(),
                                       _ctx.at(input_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int input_index;
    int output_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});

    std::unique_ptr<::arm_compute::IFunction> fn;

    {
      if (::internal::arm_compute::isGpuMode())
      {
        auto l = std::make_unique<::arm_compute::CLCast>();

        l->configure(CAST_CL(input_alloc), CAST_CL(output_alloc));
        fn = std::move(l);
      }
      else
        throw std::runtime_error("Not supported, yet");
    }

    builder.append("Cast", std::move(fn));
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::TopKV2::Node &node)
{
  const ::internal::tflite::operand::Index outputValues_index{node.param().outputValues_index};
  const ::internal::tflite::operand::Index outputIndices_index{node.param().outputIndices_index};

  const ::internal::tflite::operand::Index inputData_index{node.param().inputData_index};
  const ::internal::tflite::operand::Index k_index{node.param().k_index};

  // Currently, we only support the vector input.
  assert(_ctx.at(inputData_index).shape().rank() == 1 ||
         _ctx.at(inputData_index).shape().rank() == 2);

  const int32_t k = _ctx.at(k_index).asScalar<int32_t>();

  // Set shape constraints
  _builder.addShapeConstr(outputValues_index,
                          asTensorInfo(asTensorShape(_ctx.at(outputValues_index).shape()),
                                       _ctx.at(outputValues_index).type(),
                                       _ctx.at(outputValues_index).scale(),
                                       _ctx.at(outputValues_index).zeroPoint()));
  _builder.addShapeConstr(outputIndices_index,
                          asTensorInfo(asTensorShape(_ctx.at(outputIndices_index).shape()),
                                       _ctx.at(outputIndices_index).type(),
                                       _ctx.at(outputIndices_index).scale(),
                                       _ctx.at(outputIndices_index).zeroPoint()));
  _builder.addShapeConstr(
      inputData_index,
      asTensorInfo(asTensorShape(_ctx.at(inputData_index).shape()), _ctx.at(inputData_index).type(),
                   _ctx.at(inputData_index).scale(), _ctx.at(inputData_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int32_t outputValues_index;
    int32_t outputIndices_index;

    int32_t inputData_index;
    int32_t k;
  };

  Param param;

  param.outputValues_index = outputValues_index.asInt();
  param.outputIndices_index = outputIndices_index.asInt();
  param.inputData_index = inputData_index.asInt();
  param.k = k;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto values_alloc = ctx.at(::internal::tflite::operand::Index{param.outputValues_index});
    auto indices_alloc = ctx.at(::internal::tflite::operand::Index{param.outputIndices_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.inputData_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLTopKV2>();

      fn->configure(CAST_CL(input_alloc), param.k, CAST_CL(values_alloc), CAST_CL(indices_alloc));

      builder.append("TopKV2", std::move(fn));
    }
    else
      throw std::runtime_error("Not supported, yet");
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Gather::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};

  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index indices_index{node.param().indices_index};

  const ::internal::tflite::operand::Index axis_index{node.param().axis_index};

  const auto ifm_shape = _ctx.at(ifm_index).shape();
  const auto indices_shape = _ctx.at(indices_index).shape();
  const auto axis_shape = _ctx.at(axis_index).shape();
  const auto ofm_shape = _ctx.at(ofm_index).shape();

  assert(ifm_shape.rank() <= 4);
  assert(indices_shape.rank() <= 3);
  assert(ofm_shape.rank() <= 4);
  assert(_ctx.at(axis_index).hasData());
  assert(axis_shape.rank() == 0);

  // Set Shape Constraints
  _builder.addShapeConstr(ofm_index,
                          asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape(), false),
                                       _ctx.at(ofm_index).type(), _ctx.at(ofm_index).scale(),
                                       _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(ifm_index,
                          asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape(), false),
                                       _ctx.at(ifm_index).type(), _ctx.at(ifm_index).scale(),
                                       _ctx.at(ifm_index).zeroPoint()));
  _builder.addShapeConstr(
      indices_index, asTensorInfo(asTensorShape(_ctx.at(indices_index).shape(), false),
                                  _ctx.at(indices_index).type(), _ctx.at(indices_index).scale(),
                                  _ctx.at(indices_index).zeroPoint()));

  const int32_t axis_value = static_cast<int>(_ctx.at(axis_index).asScalar<int32_t>());
  const int axis = ToARMComputeAxis(ifm_shape.rank(), axis_value).value();

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    int indices_index;

    int axis;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.indices_index = indices_index.asInt();

  param.axis = axis;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});
    auto indices_alloc = ctx.at(::internal::tflite::operand::Index{param.indices_index});

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::IFunction> fn;

      auto l = std::make_unique<GenericGather>();
      l->configure(CAST_CL(ifm_alloc), CAST_CL(indices_alloc), CAST_CL(ofm_alloc), param.axis);
      fn = std::move(l);
      builder.append("Gather", std::move(fn));
    }
    else
      throw std::runtime_error("Not supported, yet");
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::PReLU::Node &node)
{
  VERBOSE(PReLU) << "Configure PReLU operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index alpha_index{node.param().alpha_index};

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));

  if (!(_ctx.at(ifm_index).shape() == _ctx.at(alpha_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(ifm_index).shape().rank(), _ctx.at(alpha_index).shape().rank());
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(ifm_index).shape())
        .extendRank(broadcast_rank);
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(alpha_index).shape())
        .extendRank(broadcast_rank);
  }
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  _builder.addShapeConstr(alpha_index,
                          asTensorInfo(asTensorShape(_ctx.at(alpha_index).shape()),
                                       _ctx.at(alpha_index).type(), _ctx.at(alpha_index).scale(),
                                       _ctx.at(alpha_index).zeroPoint()));

  struct Param
  {
    int ofm_index;
    int ifm_index;
    int alpha_index;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.alpha_index = alpha_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});
    auto alpha_alloc = ctx.at(::internal::tflite::operand::Index{param.alpha_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLPReLU>();
      fn->configure(CAST_CL(ifm_alloc), CAST_CL(alpha_alloc), CAST_CL(ofm_alloc));
      builder.append("PReLU", std::move(fn));
    }
    else
    {
      // TODO Add NEON support

      throw std::runtime_error("Not supported, yet");
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::ReLU::Node &node)
{
  VERBOSE(ReLU) << "Configure ReLU operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  // Set shape constraints
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  struct Param
  {
    int ofm_index;
    int ifm_index;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    const ::arm_compute::ActivationLayerInfo act_info{
        ::arm_compute::ActivationLayerInfo::ActivationFunction::RELU};

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLActivationLayer>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), act_info);

      builder.append("ReLU", std::move(fn));
    }
    else
    {
      auto fn = std::make_unique<::arm_compute::NEActivationLayer>();

      fn->configure(ifm_alloc, ofm_alloc, act_info);

      builder.append("ReLU", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::ReLU1::Node &node)
{
  VERBOSE(ReLU1) << "Configure ReLU1 operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  // Set shape constraints
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  struct Param
  {
    int ofm_index;
    int ifm_index;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    const ::arm_compute::ActivationLayerInfo act_info{
        ::arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 1.0f, -1.0f};

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLActivationLayer>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), act_info);

      builder.append("ReLU1", std::move(fn));
    }
    else
    {
      auto fn = std::make_unique<::arm_compute::NEActivationLayer>();

      fn->configure(ifm_alloc, ofm_alloc, act_info);

      builder.append("ReLU1", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::ReLU6::Node &node)
{
  VERBOSE(ReLU6) << "Configure ReLU6 operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  // Set shape constraints
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  struct Param
  {
    int ofm_index;
    int ifm_index;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    const ::arm_compute::ActivationLayerInfo act_info{
        ::arm_compute::ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.0f};

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLActivationLayer>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), act_info);

      builder.append("ReLU6", std::move(fn));
    }
    else
    {
      auto fn = std::make_unique<::arm_compute::NEActivationLayer>();

      fn->configure(ifm_alloc, ofm_alloc, act_info);

      builder.append("ReLU6", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Tanh::Node &node)
{
  VERBOSE(Tanh) << "Configure Tanh operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  // Set shape constraints
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));

  struct Param
  {
    int ofm_index;
    int ifm_index;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    const ::arm_compute::ActivationLayerInfo act_info{
        ::arm_compute::ActivationLayerInfo::ActivationFunction::TANH, 1.0f, 1.0f};

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLActivationLayer>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), act_info);

      builder.append("Tanh", std::move(fn));
    }
    else
    {
      auto fn = std::make_unique<::arm_compute::NEActivationLayer>();

      fn->configure(ifm_alloc, ofm_alloc, act_info);

      builder.append("Tanh", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Logistic::Node &node)
{
  VERBOSE(Logistic) << "Configure Logistic operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  // Set shape constraints
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  struct Param
  {
    int ofm_index;
    int ifm_index;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    const ::arm_compute::ActivationLayerInfo act_info{
        ::arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC};

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLActivationLayer>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), act_info);

      builder.append("Logistic", std::move(fn));
    }
    else
    {
      auto fn = std::make_unique<::arm_compute::NEActivationLayer>();

      fn->configure(ifm_alloc, ofm_alloc, act_info);

      builder.append("Logistic", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

// Reduce Mean
void Planner::visit(const ::internal::tflite::op::Mean::Node &node)
{
  VERBOSE(Mean) << "Configure Mean operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index axis_index{node.param().axis_index};
  const ::internal::tflite::operand::Index keep_dims_index{node.param().keep_dims_index};
  const int keep_dims = _ctx.at(keep_dims_index).asScalar<int>();

  const auto ifm_shape = _ctx.at(ifm_index).shape();
  const auto ofm_shape = _ctx.at(ofm_index).shape();

  // NOTE For the 4-dimensions, if the rank of input and output are different, this runtime only
  // supports cases reducing height and width or reducing depth.
  // TODO We have to support all cases of dimensions up to 4.
  // For correct permuting, we have to set output's shape to be equal in dimension position of the
  // input. But the positions of the same dimensions in the input and output may be set differently.
  // For example {2,3,4,5}(input's shape) can be reduced to {3,5}(output's shape). The original
  // output shape should be {1,3,1,5}, but real output shape may be {3,5}. If you simply try to
  // extend it in 4 dimensions, it should be {1,1,3,5}.
  // Even if output shape is changed to {1,3,1,5}, there is another problem. It is that shape of
  // output tensor used at next operation is changed to {1,3,1,5} after this operation even if the
  // next operation is not desired.
  if (ifm_shape.rank() == 4 && ifm_shape.rank() != ofm_shape.rank())
  {
    if (ofm_shape.rank() == 2)
    {
      // Reducing HW
      assert(ifm_shape.dim(0) == ofm_shape.dim(0) && ifm_shape.dim(3) == ofm_shape.dim(1));
    }
    else if (ofm_shape.rank() == 3)
    {
      // Reducing C or
      // (Reducing H and C(ifm and ofm) == 1) or (Reducing W and C(ifm and ofm) == 1)
      assert(ifm_shape.dim(0) == ofm_shape.dim(0) && ifm_shape.dim(1) == ofm_shape.dim(1) &&
                 ifm_shape.dim(2) == ofm_shape.dim(2) ||
             (ifm_shape.dim(0) == ofm_shape.dim(0) &&
              (ifm_shape.dim(1) == ofm_shape.dim(1) || ifm_shape.dim(2) == ofm_shape.dim(1)) &&
              ifm_shape.dim(3) == 1 && ofm_shape.dim(2) == 1));
    }
  }

  // Set shape constraints
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));
  _builder.addShapeConstr(axis_index,
                          asTensorInfo(asTensorShape(_ctx.at(axis_index).shape()),
                                       _ctx.at(axis_index).type(), _ctx.at(axis_index).scale(),
                                       _ctx.at(axis_index).zeroPoint()));

  std::set<uint32_t> axis;
  {
    const auto ifm_rank = ifm_shape.rank();
    const auto axis_shape = _ctx.at(axis_index).shape();
    switch (axis_shape.rank())
    {
      case 0: // scalar
      {
        int32_t axis_value = _ctx.at(axis_index).asScalar<int32_t>();
        if (axis_value < 0)
        {
          axis_value += ifm_rank;
        }
        axis.insert(ToARMComputeAxis(ifm_rank, axis_value).value());
        break;
      }
      case 1: // vector
      {
        const auto axis_base = _ctx.at(axis_index).data().base();
        const auto axis_size = _ctx.at(axis_index).shape().asVector();

        // If axis's data does not exist as constant values and can be gotten as input data, we have
        // to find a way to infer output shape when sinking output.
        assert(axis_base != nullptr);
        for (uint32_t n = 0; n < axis_size; ++n)
        {
          int32_t axis_value = *(reinterpret_cast<const int32_t *>(axis_base) + n);
          if (axis_value < 0)
          {
            axis_value += ifm_rank;
          }
          axis.insert(ToARMComputeAxis(ifm_rank, axis_value).value());
        }
        break;
      }
      default:
        throw std::runtime_error("Not supported");
        break;
    }
  }

  struct Param
  {
    int ofm_index;
    int ifm_index;
    bool keep_dims;
    std::set<uint32_t> axis;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.keep_dims = keep_dims > 0 ? true : false;
  param.axis = axis;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    ::arm_compute::Coordinates reduction_axis;
    size_t i = 0;
    for (auto index : param.axis)
    {
      reduction_axis.set(i++, index);
    }

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLReduceMean>();

      fn->configure(CAST_CL(ifm_alloc), reduction_axis, param.keep_dims, CAST_CL(ofm_alloc));

      builder.append("Mean", std::move(fn));
    }
    else
      throw std::runtime_error("Not supported, yet");
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::RNN::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index hidden_state_out_index{
      node.param().hidden_state_out_index};

  const ::internal::tflite::operand::Index input_index{node.param().input_index};
  const ::internal::tflite::operand::Index weights_index{node.param().weights_index};
  const ::internal::tflite::operand::Index recurrent_weights_index{
      node.param().recurrent_weights_index};
  const ::internal::tflite::operand::Index bias_index{node.param().bias_index};
  const ::internal::tflite::operand::Index hidden_state_in_index{
      node.param().hidden_state_in_index};
  const ::internal::tflite::operand::Index fused_activation_index{
      node.param().fused_activation_index};

  assert(_ctx.at(output_index).shape().rank() == 2 &&
         _ctx.at(hidden_state_out_index).shape().rank() == 2 &&
         _ctx.at(input_index).shape().rank() == 2 && _ctx.at(weights_index).shape().rank() == 2 &&
         _ctx.at(recurrent_weights_index).shape().rank() == 2 &&
         _ctx.at(hidden_state_in_index).shape().rank() == 2);
  assert(_ctx.at(bias_index).shape().rank() == 1);

  const auto batch_size = _ctx.at(output_index).shape().dim(0);
  assert(batch_size == _ctx.at(input_index).shape().dim(0) &&
         batch_size == _ctx.at(hidden_state_in_index).shape().dim(0) &&
         batch_size == _ctx.at(hidden_state_out_index).shape().dim(0));
  assert(_ctx.at(input_index).shape().dim(1) == _ctx.at(weights_index).shape().dim(1));

  const auto num_units = _ctx.at(output_index).shape().dim(1);
  assert(num_units == _ctx.at(weights_index).shape().dim(0) &&
         num_units == _ctx.at(recurrent_weights_index).shape().dim(0) &&
         num_units == _ctx.at(bias_index).shape().dim(0));
  assert(num_units == _ctx.at(output_index).shape().dim(1) &&
         num_units == _ctx.at(recurrent_weights_index).shape().dim(1) &&
         num_units == _ctx.at(hidden_state_in_index).shape().dim(1) &&
         num_units == _ctx.at(hidden_state_out_index).shape().dim(1));

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(hidden_state_out_index,
                          asTensorInfo(asTensorShape(_ctx.at(hidden_state_out_index).shape()),
                                       _ctx.at(hidden_state_out_index).type(),
                                       _ctx.at(hidden_state_out_index).scale(),
                                       _ctx.at(hidden_state_out_index).zeroPoint()));
  _builder.addShapeConstr(input_index,
                          asTensorInfo(asTensorShape(_ctx.at(input_index).shape()),
                                       _ctx.at(input_index).type(), _ctx.at(input_index).scale(),
                                       _ctx.at(input_index).zeroPoint()));
  _builder.addShapeConstr(weights_index, asTensorInfo(asTensorShape(_ctx.at(weights_index).shape()),
                                                      _ctx.at(weights_index).type(),
                                                      _ctx.at(weights_index).scale(),
                                                      _ctx.at(weights_index).zeroPoint()));
  _builder.addShapeConstr(recurrent_weights_index,
                          asTensorInfo(asTensorShape(_ctx.at(recurrent_weights_index).shape()),
                                       _ctx.at(recurrent_weights_index).type(),
                                       _ctx.at(recurrent_weights_index).scale(),
                                       _ctx.at(recurrent_weights_index).zeroPoint()));
  _builder.addShapeConstr(bias_index,
                          asTensorInfo(asTensorShape(_ctx.at(bias_index).shape()),
                                       _ctx.at(bias_index).type(), _ctx.at(bias_index).scale(),
                                       _ctx.at(bias_index).zeroPoint()));
  _builder.addShapeConstr(hidden_state_in_index,
                          asTensorInfo(asTensorShape(_ctx.at(hidden_state_in_index).shape()),
                                       _ctx.at(hidden_state_in_index).type(),
                                       _ctx.at(hidden_state_in_index).scale(),
                                       _ctx.at(hidden_state_in_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int output_index;
    int hidden_state_out_index;

    int input_index;
    int weights_index;
    int recurrent_weights_index;
    int bias_index;
    int hidden_state_in_index;

    FuseCode activation;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.hidden_state_out_index = hidden_state_out_index.asInt();

  param.input_index = input_index.asInt();
  param.weights_index = weights_index.asInt();
  param.recurrent_weights_index = recurrent_weights_index.asInt();
  param.bias_index = bias_index.asInt();
  param.hidden_state_in_index = hidden_state_in_index.asInt();
  param.activation = static_cast<FuseCode>(_ctx.at(fused_activation_index).asScalar<int32_t>());

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto hidden_state_out_alloc =
        ctx.at(::internal::tflite::operand::Index{param.hidden_state_out_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});
    auto weights_alloc = ctx.at(::internal::tflite::operand::Index{param.weights_index});
    auto recurrent_weights_alloc =
        ctx.at(::internal::tflite::operand::Index{param.recurrent_weights_index});
    auto bias_alloc = ctx.at(::internal::tflite::operand::Index{param.bias_index});
    auto hidden_state_in_alloc =
        ctx.at(::internal::tflite::operand::Index{param.hidden_state_in_index});
    auto act_info = asActivationInfo(param.activation);

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::CLCopy> copy_fn{new ::arm_compute::CLCopy};
      copy_fn->configure(CAST_CL(hidden_state_in_alloc), CAST_CL(hidden_state_out_alloc));
      builder.append("COPY", std::move(copy_fn));

      std::unique_ptr<::arm_compute::CLRNNLayer> rnn_fn{new ::arm_compute::CLRNNLayer};

      // The hidden_state_in's data must be copied to hidden_state_out_alloc before fn->run() is
      // performed.
      rnn_fn->configure(CAST_CL(input_alloc), CAST_CL(weights_alloc),
                        CAST_CL(recurrent_weights_alloc), CAST_CL(bias_alloc),
                        CAST_CL(hidden_state_out_alloc), CAST_CL(output_alloc), act_info);

      builder.append("RNN", std::move(rnn_fn));
    }
    else
      throw std::runtime_error("Not supported, yet");
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::LSTM::Node &node)
{
  // TODO Implement LSTM op
  throw std::runtime_error("Not supported, yet");
}

void Planner::visit(const ::internal::tflite::op::Transpose::Node &node)
{
  VERBOSE(Transpose) << "Configure Transpose operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index permu_index{node.param().permu_index};

  assert(_ctx.at(ifm_index).shape().rank() == _ctx.at(ofm_index).shape().rank());
  assert(_ctx.at(permu_index).hasData() == true);

  // Set shape constraints
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  struct Param
  {
    int ofm_index;
    int ifm_index;
    const int32_t *pv;
    int rank;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.pv = reinterpret_cast<const int32_t *>(_ctx.at(permu_index).data().base());
  param.rank = _ctx.at(ifm_index).shape().rank();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {

    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    const auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLPermute>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc),
                    getARMComputePermutationVector(param.rank, param.pv));

      builder.append("Transpose", std::move(fn));
    }
    else
    {
      throw std::runtime_error("Not supported, yet");
    }

  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Floor::Node &node)
{
  VERBOSE(Floor) << "Configure Floor operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().output_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().input_index};

  // Set shape constraints
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  struct Param
  {
    int ofm_index;
    int ifm_index;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLFloor>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc));

      builder.append("Floor", std::move(fn));
    }
    else
    {
      auto fn = std::make_unique<::arm_compute::NEFloor>();

      fn->configure(ifm_alloc, ofm_alloc);

      builder.append("Floor", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::ArgMax::Node &node)
{
  VERBOSE(ArgMax) << "Configure ARGMAX operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index axis_index{node.param().axis_index};

  auto ifm_shape = _ctx.at(ifm_index).shape();
  auto ofm_shape = _ctx.at(ofm_index).shape();
  auto axis_shape = _ctx.at(axis_index).shape();

  assert(_ctx.at(axis_index).hasData());
  // Axis dimension is always 1.
  assert(axis_shape.rank() == 1);
  assert((ifm_shape.rank() - 1) == ofm_shape.rank());

  _builder.addShapeConstr(ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape(), false),
                                                  _ctx.at(ofm_index).type()));
  _builder.addShapeConstr(ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape(), false),
                                                  _ctx.at(ifm_index).type()));

  std::vector<uint32_t> l_axis;
  const auto axis_size = _ctx.at(axis_index).shape().asVector();
  auto axis_base = _ctx.at(axis_index).data().base();
  auto axis_type = _ctx.at(axis_index).type();
  // TODO Should support axis size > 1.
  assert(axis_size == 1);
  // axis is tensor with 1 dimension - always a vector.
  assert(axis_base != nullptr);
  for (uint32_t n = 0; n < axis_size; ++n)
  {
    int32_t axis_value = *(reinterpret_cast<const int32_t *>(axis_base) + n);
    if (axis_value < 0)
    {
      axis_value += ifm_shape.rank();
    }
    l_axis.push_back(ToARMComputeAxis(ifm_shape.rank(), axis_value).value());
  }

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    std::vector<uint32_t> axis;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.axis = l_axis;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    {

      if (::internal::arm_compute::isGpuMode())
      {
        auto fn = std::make_unique<::arm_compute::CLArgOperation>();

        fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), param.axis,
                      ::arm_compute::ArgOperation::MAX);

        builder.append("ArgMax", std::move(fn));
      }
      else
        throw std::runtime_error("Not supported, yet");
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::SQRT::Node &node)
{
  VERBOSE(SQRT) << "Configure SQRT operation" << std::endl;

  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};

  // Set shape constraints
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(input_index,
                          asTensorInfo(asTensorShape(_ctx.at(input_index).shape()),
                                       _ctx.at(input_index).type(), _ctx.at(input_index).scale(),
                                       _ctx.at(input_index).zeroPoint()));

  struct Param
  {
    int output_index;
    int input_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});

    const ::arm_compute::ActivationLayerInfo act_info{
        ::arm_compute::ActivationLayerInfo::ActivationFunction::SQRT};

    {
      if (::internal::arm_compute::isGpuMode())
      {
        auto fn = std::make_unique<::arm_compute::CLActivationLayer>();

        fn->configure(CAST_CL(input_alloc), CAST_CL(output_alloc), act_info);

        builder.append("SQRT", std::move(fn));
      }
      else
      {
        auto fn = std::make_unique<::arm_compute::NEActivationLayer>();

        fn->configure(input_alloc, output_alloc, act_info);

        builder.append("SQRT", std::move(fn));
      }
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::RSQRT::Node &node)
{
  VERBOSE(RSQRT) << "Configure Rsqrt operation" << std::endl;

  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};

  // Set shape constraints
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(input_index,
                          asTensorInfo(asTensorShape(_ctx.at(input_index).shape()),
                                       _ctx.at(input_index).type(), _ctx.at(input_index).scale(),
                                       _ctx.at(input_index).zeroPoint()));

  struct Param
  {
    int output_index;
    int input_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLRsqrtLayer>();

      fn->configure(CAST_CL(input_alloc), CAST_CL(output_alloc));

      builder.append("RSQRT", std::move(fn));
    }
    else
      throw std::runtime_error("Not supported, yet");
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Equal::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input1_index{node.param().input1_index};
  const ::internal::tflite::operand::Index input2_index{node.param().input2_index};

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));

  if (!(_ctx.at(input1_index).shape() == _ctx.at(input2_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(input1_index).shape().rank(), _ctx.at(input2_index).shape().rank());
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(input1_index).shape())
        .extendRank(broadcast_rank);
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(input2_index).shape())
        .extendRank(broadcast_rank);
  }
  _builder.addShapeConstr(input1_index,
                          asTensorInfo(asTensorShape(_ctx.at(input1_index).shape()),
                                       _ctx.at(input1_index).type(), _ctx.at(input1_index).scale(),
                                       _ctx.at(input1_index).zeroPoint()));
  _builder.addShapeConstr(input2_index,
                          asTensorInfo(asTensorShape(_ctx.at(input2_index).shape()),
                                       _ctx.at(input2_index).type(), _ctx.at(input2_index).scale(),
                                       _ctx.at(input2_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int output_index;
    int input1_index;
    int input2_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input1_index = input1_index.asInt();
  param.input2_index = input2_index.asInt();
  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input1_alloc = ctx.at(::internal::tflite::operand::Index{param.input1_index});
    auto input2_alloc = ctx.at(::internal::tflite::operand::Index{param.input2_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLComparison>();

      fn->configure(CAST_CL(input1_alloc), CAST_CL(input2_alloc), CAST_CL(output_alloc),
                    ::arm_compute::ComparisonOperation::Equal);

      builder.append("Equal", std::move(fn));
    }
    else
    {
      // TODO Add NEON support

      throw std::runtime_error("Not supported, yet");
    }
  };
  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::TransposeConv::Node &node)
{
  VERBOSE(TransposeConv) << "Configure TransposeConv operation" << std::endl;

  const ::internal::tflite::operand::Index op_shape_index{node.param().op_shape_index};
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index ker_index{node.param().ker_index};

  const ::internal::tflite::operand::Index padding_index{node.param().padding_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};
  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};

  // Only 4D tensors are supported
  assert(_ctx.at(ofm_index).shape().rank() == 4);
  assert(_ctx.at(ofm_index).shape().rank() == _ctx.at(ifm_index).shape().rank());
  assert(_ctx.at(ofm_index).shape().rank() == _ctx.at(ker_index).shape().rank());

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();
  const auto ker_shape = _ctx.at(ker_index).shape().asFeature();

  assert(_ctx.at(padding_index).hasData() == true);

  const int32_t vstride = _ctx.at(vstride_index).asScalar<int32_t>();
  const int32_t hstride = _ctx.at(hstride_index).asScalar<int32_t>();

  const PaddingCode padding_type =
      static_cast<PaddingCode>(_ctx.at(padding_index).asScalar<int32_t>());

  assert(vstride > 0);
  assert(hstride > 0);
  assert((ANEURALNETWORKS_PADDING_SAME == padding_type) ||
         (ANEURALNETWORKS_PADDING_VALID == padding_type));
  assert(ifm_shape.N == ofm_shape.N);
  assert(ifm_shape.C == ker_shape.C);
  assert(ker_shape.N == ofm_shape.C);

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ker_index, asTensorInfo(asTensorShape(_ctx.at(ker_index).shape()), _ctx.at(ker_index).type(),
                              _ctx.at(ker_index).scale(), _ctx.at(ker_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    int ker_index;
    Padding padding;
    Stride stride;
    uint32_t invalid_horizontal;
    uint32_t invalid_vertical;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.ker_index = ker_index.asInt();

  param.stride.horizontal = hstride;
  param.stride.vertical = vstride;

  param.padding = (padding_type == ANEURALNETWORKS_PADDING_SAME)
                      ? same_padding(ofm_shape, ifm_shape, param.stride, ker_shape.W, ker_shape.H)
                      : valid_padding();

  param.invalid_horizontal =
      (padding_type == ANEURALNETWORKS_PADDING_SAME)
          ? 0
          : ofm_shape.W - (1 + (ifm_shape.W - 1) * hstride) - (ker_shape.W - 1);
  param.invalid_vertical =
      (padding_type == ANEURALNETWORKS_PADDING_SAME)
          ? 0
          : ofm_shape.H - (1 + (ifm_shape.H - 1) * param.stride.vertical) - (ker_shape.H - 1);

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});
    auto ker_alloc = ctx.at(::internal::tflite::operand::Index{param.ker_index});

    // Only rank 4 is supported
    const int rank = 4;

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLTransposeConvLayer>();

      auto symmetric_tconv_info = asPadStrideInfo(param.padding, param.stride);

      // TODO Support WeightInfo in some cases in order to performance improvement
      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ker_alloc), nullptr, CAST_CL(ofm_alloc),
                    symmetric_tconv_info, param.invalid_horizontal, param.invalid_vertical);
      builder.append("TransposeConv", std::move(fn));
    }
    else
    {
      throw std::runtime_error("Not supported, yet");
    }
  };
  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::SquaredDifference::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index lhs_index{node.param().lhs_index};
  const ::internal::tflite::operand::Index rhs_index{node.param().rhs_index};

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));

  if (!(_ctx.at(lhs_index).shape() == _ctx.at(rhs_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(lhs_index).shape().rank(), _ctx.at(rhs_index).shape().rank());
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(lhs_index).shape())
        .extendRank(broadcast_rank);
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(rhs_index).shape())
        .extendRank(broadcast_rank);
  }
  _builder.addShapeConstr(
      lhs_index, asTensorInfo(asTensorShape(_ctx.at(lhs_index).shape()), _ctx.at(lhs_index).type(),
                              _ctx.at(lhs_index).scale(), _ctx.at(lhs_index).zeroPoint()));
  _builder.addShapeConstr(
      rhs_index, asTensorInfo(asTensorShape(_ctx.at(rhs_index).shape()), _ctx.at(rhs_index).type(),
                              _ctx.at(rhs_index).scale(), _ctx.at(rhs_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int lhs_index;
    int rhs_index;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.lhs_index = lhs_index.asInt();
  param.rhs_index = rhs_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto lhs_alloc = ctx.at(::internal::tflite::operand::Index{param.lhs_index});
    auto rhs_alloc = ctx.at(::internal::tflite::operand::Index{param.rhs_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLElementwiseSquaredDiff>();

      fn->configure(CAST_CL(lhs_alloc), CAST_CL(rhs_alloc), CAST_CL(ofm_alloc));
      builder.append("SquaredDifference", std::move(fn));
    }
    else
    {
      // TODO Enable NEON Support
      throw std::runtime_error("Not supported, yet");
    }

  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Split::Node &node)
{
  VERBOSE(Split) << "Configure Split operation" << std::endl;

  const ::internal::tflite::operand::Index axis_index{node.param().axis_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  const auto ifm_shape = _ctx.at(ifm_index).shape();
  int32_t axis = _ctx.at(axis_index).asScalar<int32_t>();

  // Handle negative axis
  if (axis < 0)
  {
    axis += ifm_shape.rank();
  }

  const int32_t num_split = node.param().ofm_indexes.size();
  const auto input_size = ifm_shape.dim(axis);
  assert(input_size % num_split == 0);
  const int32_t slice_size = input_size / num_split;

  // Set Shape Constraints and TensorInfo (for input)
  _builder.addShapeConstr(ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()),
                                                  _ctx.at(ifm_index).type()));

  // Set Shape Constraints and TensorInfo (for output)
  const auto rank = ifm_shape.rank();
  const uint32_t coord_index = ToARMComputeAxis(rank, axis).value();
  uint32_t depth = 0;

  ::arm_compute::Coordinates coordinates;
  coordinates.set_num_dimensions(rank);

  for (const auto &index : node.param().ofm_indexes)
  {
    const ::internal::tflite::operand::Index ofm_index{index};

    coordinates[coord_index] = depth;

    _builder.addSubsumptionConstr(ofm_index, ifm_index, coordinates,
                                  asTensorShape(_ctx.at(ofm_index).shape()), true);
    depth += slice_size;
  }

  // NOTE Split has no actual operation!
}

void Planner::visit(const ::internal::tflite::op::Pad::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index paddings_index{node.param().paddings_index};

  assert(_ctx.at(paddings_index).hasData() == true);

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(ifm_index,
                          asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape(), false),
                                       _ctx.at(ifm_index).type(), _ctx.at(ifm_index).scale(),
                                       _ctx.at(ifm_index).zeroPoint()));
  _builder.addShapeConstr(ofm_index,
                          asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape(), false),
                                       _ctx.at(ofm_index).type(), _ctx.at(ofm_index).scale(),
                                       _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      paddings_index, asTensorInfo(asTensorShape(_ctx.at(paddings_index).shape(), false),
                                   _ctx.at(paddings_index).type(), _ctx.at(paddings_index).scale(),
                                   _ctx.at(paddings_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    ::arm_compute::PixelValue pixel_value;
    ::arm_compute::PaddingList padding_list;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  // initializer for padding
  auto rank = _ctx.at(ifm_index).shape().rank();
  auto pad_type = _ctx.at(paddings_index).type();

  if (pad_type == ANEURALNETWORKS_TENSOR_INT32)
  {
    auto pad_base = _ctx.at(paddings_index).data().base();
    auto pad_shape = _ctx.at(paddings_index).shape();

    param.padding_list.resize(rank);
    for (int32_t n = 0; n < rank; ++n)
    {
      const int32_t *from = reinterpret_cast<const int32_t *>(pad_base) + (n * pad_shape.dim(1));
      auto axis = ToARMComputeAxis(rank, n).value();

      param.padding_list[axis] = ::arm_compute::PaddingInfo{from[0], from[1]};
    }
    auto data_type = asDataType(_ctx.at(ifm_index).type());
    auto quant_info =
        asQuantizationInfo(_ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint());
    param.pixel_value = ::arm_compute::PixelValue{0, data_type, quant_info};
  }
  else
  {
    throw std::runtime_error("Only Int32 datatype is supported for Pad values");
  }

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    {
      if (::internal::arm_compute::isGpuMode()) // GPU
      {
        auto fn = std::make_unique<::arm_compute::CLPadLayer>();

        fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), param.padding_list,
                      param.pixel_value);

        builder.append("PAD", std::move(fn));
      }
      else // NEON
      {
        // TODO Enable NEON Support
        throw std::runtime_error("Not supported, yet");
      }
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::SpaceToDepth::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};
  const ::internal::tflite::operand::Index block_size_index{node.param().block_size_index};

  const auto input_batch = _ctx.at(input_index).shape().dim(0);
  const auto output_batch = _ctx.at(output_index).shape().dim(0);
  const auto input_depth = _ctx.at(input_index).shape().dim(3);
  const auto output_depth = _ctx.at(output_index).shape().dim(3);
  const auto block_size = _ctx.at(block_size_index).asScalar<int32_t>();
  const auto input_height = _ctx.at(input_index).shape().dim(1);
  const auto input_width = _ctx.at(input_index).shape().dim(2);

  // All assertions as per NNAPI specification.
  assert(_ctx.at(input_index).shape().rank() == 4);
  assert(_ctx.at(output_index).shape().rank() == 4);
  assert((block_size >= 1) && (input_height % block_size == 0) && (input_width % block_size == 0));
  assert(input_batch == output_batch);
  assert(input_depth * block_size * block_size == output_depth);

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape(), false),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(input_index,
                          asTensorInfo(asTensorShape(_ctx.at(input_index).shape(), false),
                                       _ctx.at(input_index).type(), _ctx.at(input_index).scale(),
                                       _ctx.at(input_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int output_index;
    int input_index;
    int32_t block_size;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();
  param.block_size = block_size;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});

    {
      if (::internal::arm_compute::isGpuMode()) // GPU
      {
        auto fn = std::make_unique<::arm_compute::CLSpaceToDepth>();

        fn->configure(CAST_CL(input_alloc), CAST_CL(output_alloc), param.block_size);

        builder.append("SpaceToDepth", std::move(fn));
      }
      else // NEON
      {
        // TODO Enable NEON Support
        throw std::runtime_error("Not supported, yet");
      }
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::SpaceToBatchND::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};
  const ::internal::tflite::operand::Index block_size_index{node.param().block_size_index};
  const ::internal::tflite::operand::Index padding_size_index{node.param().padding_size_index};

  const auto &output_shape = _ctx.at(output_index).shape();
  const auto &input_shape = _ctx.at(input_index).shape();
  const auto &padding_size_shape = _ctx.at(padding_size_index).shape();
  auto block_size_base = reinterpret_cast<const int32_t *>(_ctx.at(block_size_index).data().base());
  auto padding_size_base =
      reinterpret_cast<const int32_t *>(_ctx.at(padding_size_index).data().base());

  { // New block for assertions
    const auto &block_size_shape = _ctx.at(block_size_index).shape();

    // Currently, only 4D NHWC input/output op_context are supported.
    // The 4D array need to have exactly 2 spatial dimensions.
    // TODO: Support arbitrary dimension in SpaceToBatchND.
    assert(input_shape.rank() == 4);
    assert(output_shape.rank() == 4);
    assert(block_size_shape.rank() == 1);
    assert(padding_size_shape.rank() == 2);

    assert(output_shape.dim(3) == input_shape.dim(3));
    assert(block_size_shape.dim(0) == 2);
    assert(padding_size_shape.dim(0) == 2);
    assert(padding_size_shape.dim(1) == 2);

    assert(_ctx.at(block_size_index).hasData() && _ctx.at(padding_size_index).hasData());
    assert(_ctx.at(block_size_index).type() == ANEURALNETWORKS_TENSOR_INT32);
    assert(_ctx.at(padding_size_index).type() == ANEURALNETWORKS_TENSOR_INT32);

    assert(block_size_base[0] > 0 && block_size_base[1] > 0);
    assert(output_shape.dim(0) == input_shape.dim(0) * block_size_base[0] * block_size_base[1]);
    assert(output_shape.dim(1) ==
           (input_shape.dim(1) + padding_size_base[0] + padding_size_base[1]) / block_size_base[0]);
    assert(output_shape.dim(2) ==
           (input_shape.dim(2) + padding_size_base[2] + padding_size_base[3]) / block_size_base[1]);
  }

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape(), false),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(input_index,
                          asTensorInfo(asTensorShape(_ctx.at(input_index).shape(), false),
                                       _ctx.at(input_index).type(), _ctx.at(input_index).scale(),
                                       _ctx.at(input_index).zeroPoint()));

  _builder.addShapeConstr(block_size_index,
                          asTensorInfo(asTensorShape(_ctx.at(block_size_index).shape()),
                                       _ctx.at(block_size_index).type(),
                                       _ctx.at(block_size_index).scale(),
                                       _ctx.at(block_size_index).zeroPoint()));

  _builder.addShapeConstr(padding_size_index,
                          asTensorInfo(asTensorShape(_ctx.at(padding_size_index).shape()),
                                       _ctx.at(padding_size_index).type(),
                                       _ctx.at(padding_size_index).scale(),
                                       _ctx.at(padding_size_index).zeroPoint()));

  { // Append block_size initializer
    auto initializer = [block_size_base](::arm_compute::ITensor &tensor) {
      const auto block_size_y = block_size_base[0];
      const auto block_size_x = block_size_base[1];

      auto into = reinterpret_cast<int32_t *>(tensor.ptr_to_element({0}));
      into[0] = block_size_x;
      into[1] = block_size_y;
    };
    _builder.addInitializer(block_size_index, initializer);
  }

  { // Append padding_size initializer
    auto initializer = [padding_size_base, padding_size_shape](::arm_compute::ITensor &tensor) {
      // If n == 0, then the axis is the height
      // If n == 1, then the axis is the width
      for (size_t n = 0; n < padding_size_shape.dim(0); ++n)
      {
        const auto from = padding_size_base + (n * padding_size_shape.dim(1));
        auto into = reinterpret_cast<int32_t *>(tensor.ptr_to_element({0, 1 - n}));
        into[0] = from[0];
        into[1] = from[1];
      }
    };
    _builder.addInitializer(padding_size_index, initializer);
  }

  // Construct operation parameters
  struct Param
  {
    int output_index;
    int input_index;
    int block_size_index;
    int padding_size_index;
    int32_t rank;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();
  param.block_size_index = block_size_index.asInt();
  param.padding_size_index = padding_size_index.asInt();
  param.rank = _ctx.at(input_index).shape().rank();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});
    auto block_size_alloc = ctx.at(::internal::tflite::operand::Index{param.block_size_index});
    auto padding_size_alloc = ctx.at(::internal::tflite::operand::Index{param.padding_size_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLSpaceToBatchND>();

      fn->configure(CAST_CL(input_alloc), CAST_CL(block_size_alloc), CAST_CL(padding_size_alloc),
                    CAST_CL(output_alloc));
      builder.append("SpaceToBatchND", std::move(fn));
    }
    else
    {
      // TODO Enable NEON Support
      throw std::runtime_error("Not supported, yet");
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::BatchToSpaceNd::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};
  const ::internal::tflite::operand::Index block_size_index{node.param().block_size_index};

  assert(_ctx.at(input_index).shape().rank() == 4);
  assert(_ctx.at(output_index).shape().rank() == 4);
  assert(_ctx.at(block_size_index).shape().rank() == 1);
  assert(_ctx.at(block_size_index).hasData() == true);

  const int32_t *block_size =
      reinterpret_cast<const int32_t *>(_ctx.at(block_size_index).data().base());

  const auto &output_shape = _ctx.at(output_index).shape();
  const auto &input_shape = _ctx.at(input_index).shape();
  const auto &block_size_shape = _ctx.at(block_size_index).shape();

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      output_index, asTensorInfo(asTensorShape(output_shape, false), _ctx.at(output_index).type(),
                                 _ctx.at(output_index).scale(), _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(
      input_index, asTensorInfo(asTensorShape(input_shape, false), _ctx.at(input_index).type(),
                                _ctx.at(input_index).scale(), _ctx.at(input_index).zeroPoint()));

  _builder.addShapeConstr(block_size_index, asTensorInfo(asTensorShape(block_size_shape),
                                                         _ctx.at(block_size_index).type(),
                                                         _ctx.at(block_size_index).scale(),
                                                         _ctx.at(block_size_index).zeroPoint()));

  // initializer for block_size
  {
    const auto block_size_base =
        reinterpret_cast<const int32_t *>(_ctx.at(block_size_index).data().base());

    assert(output_shape.dim(3) == input_shape.dim(3));
    assert(output_shape.dim(1) == input_shape.dim(1) * block_size_base[0]);
    assert(output_shape.dim(2) == input_shape.dim(2) * block_size_base[1]);
    assert(output_shape.dim(0) == input_shape.dim(0) / (block_size_base[0] * block_size_base[1]));
    assert(_ctx.at(block_size_index).type() == ANEURALNETWORKS_TENSOR_INT32);

    assert((_ctx.at(block_size_index).data().size() / sizeof(int32_t)) == 2 &&
           block_size_base[0] > 0 && block_size_base[1] > 0);

    auto initializer = [block_size_base](::arm_compute::ITensor &tensor) {
      const int32_t *from = reinterpret_cast<const int32_t *>(block_size_base);
      int32_t *into = reinterpret_cast<int32_t *>(tensor.ptr_to_element({0}));
      into[0] = from[1];
      into[1] = from[0];
    };
    _builder.addInitializer(block_size_index, initializer);
  }

  // Construct operation parameters
  struct Param
  {
    int output_index;
    int input_index;
    int block_size_index;
    const int32_t *block_size;
    int32_t rank;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();
  param.block_size_index = block_size_index.asInt();
  param.block_size = block_size;
  param.rank = _ctx.at(input_index).shape().rank();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});
    auto block_size_alloc = ctx.at(::internal::tflite::operand::Index{param.block_size_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLBatchToSpaceLayer>();

      fn->configure(CAST_CL(input_alloc), CAST_CL(block_size_alloc), CAST_CL(output_alloc));
      builder.append("BatchToSpaceND", std::move(fn));
    }
    else
    {
      // TODO Enable NEON Support
      throw std::runtime_error("Not supported, yet");
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::L2Normalization::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));

  // {CL|Neon}L2Normalization performs the reduction only along dimension 0
  // L2 Normalization always performs the reduction along the depth axis
  // Thus, we repurpose {CL|Neon}NormalizationLayers to act as depthwise L2 normalizations by
  // choosing normalization parameters as below

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    int32_t radius;
    float alpha;
    float beta;
    float bias;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.radius = 2 * _ctx.at(ifm_index).shape().dim(3) + 1; // normSize = depth * 2 + 1
  param.alpha = 1.0f; // In the implementation to make alpha_ become 1
  param.beta = 0.5f;  // pow(reduction, -0.5) = 1 / sqrt(reduction)
  param.bias = 0.0f;  // Don't offset the reduction.

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    const auto norm_info =
        ::arm_compute::NormalizationLayerInfo(::arm_compute::NormType::CROSS_MAP, param.radius,
                                              param.alpha, param.beta, param.bias, false);

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLNormalizationLayer>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), norm_info);

      builder.append("L2Normalize", std::move(fn));
    }
    else
    {
      auto fn = std::make_unique<::arm_compute::NENormalizationLayer>();

      fn->configure(ifm_alloc, ofm_alloc, norm_info);

      builder.append("L2Normalize", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::L2Pool2D::Implicit::Node &node)

{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  const ::internal::tflite::operand::Index kh_index{node.param().kh_index};
  const ::internal::tflite::operand::Index kw_index{node.param().kw_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_index{node.param().padding_index};
  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();

  const int32_t kh = _ctx.at(kh_index).asScalar<int32_t>();
  const int32_t kw = _ctx.at(kw_index).asScalar<int32_t>();

  const int32_t vstride = _ctx.at(vstride_index).asScalar<int32_t>();
  const int32_t hstride = _ctx.at(hstride_index).asScalar<int32_t>();

  const PaddingCode padding_type =
      static_cast<PaddingCode>(_ctx.at(padding_index).asScalar<int32_t>());

  assert((ANEURALNETWORKS_PADDING_SAME == padding_type) ||
         (ANEURALNETWORKS_PADDING_VALID == padding_type));

  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  struct Param
  {
    int ofm_index;
    int ifm_index;

    uint32_t kw;
    uint32_t kh;

    Padding padding;
    Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.kh = kh;
  param.kw = kw;

  param.stride.vertical = vstride;
  param.stride.horizontal = hstride;

  param.padding = (padding_type == ANEURALNETWORKS_PADDING_SAME)
                      ? same_padding(ifm_shape, ofm_shape, param.stride, kw, kh)
                      : valid_padding();
  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    ::arm_compute::PoolingLayerInfo info{::arm_compute::PoolingType::L2,
                                         ::arm_compute::Size2D{param.kw, param.kh},
                                         asPadStrideInfo(param.padding, param.stride)};

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::CLPoolingLayer> fn{new ::arm_compute::CLPoolingLayer};

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), info);

      builder.append("L2Pool2D", std::move(fn));
    }
    else
    {
      std::unique_ptr<::arm_compute::NEPoolingLayer> fn{new ::arm_compute::NEPoolingLayer};

      fn->configure(ifm_alloc, ofm_alloc, info);

      builder.append("L2Pool2D", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::L2Pool2D::Explicit::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  const ::internal::tflite::operand::Index kh_index{node.param().kh_index};
  const ::internal::tflite::operand::Index kw_index{node.param().kw_index};

  const ::internal::tflite::operand::Index vstride_index{node.param().vstride_index};
  const ::internal::tflite::operand::Index hstride_index{node.param().hstride_index};

  const ::internal::tflite::operand::Index padding_left_index{node.param().padding_left_index};
  const ::internal::tflite::operand::Index padding_right_index{node.param().padding_right_index};
  const ::internal::tflite::operand::Index padding_top_index{node.param().padding_top_index};
  const ::internal::tflite::operand::Index padding_bottom_index{node.param().padding_bottom_index};

  const ::internal::tflite::operand::Index activation_index{node.param().activation_index};

  const int32_t kh = _ctx.at(kh_index).asScalar<int32_t>();
  const int32_t kw = _ctx.at(kw_index).asScalar<int32_t>();

  const int32_t vstride = _ctx.at(vstride_index).asScalar<int32_t>();
  const int32_t hstride = _ctx.at(hstride_index).asScalar<int32_t>();

  const int32_t padding_left = _ctx.at(padding_left_index).asScalar<int32_t>();
  const int32_t padding_right = _ctx.at(padding_right_index).asScalar<int32_t>();
  const int32_t padding_top = _ctx.at(padding_top_index).asScalar<int32_t>();
  const int32_t padding_bottom = _ctx.at(padding_bottom_index).asScalar<int32_t>();

  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;

    uint32_t kw;
    uint32_t kh;

    Padding padding;
    Stride stride;

    FuseCode activation;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.kh = kh;
  param.kw = kw;

  param.stride.vertical = vstride;
  param.stride.horizontal = hstride;

  param.padding.left = padding_left;
  param.padding.right = padding_right;
  param.padding.top = padding_top;
  param.padding.bottom = padding_bottom;

  param.activation = static_cast<FuseCode>(_ctx.at(activation_index).asScalar<int32_t>());

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    ::arm_compute::PoolingLayerInfo info{::arm_compute::PoolingType::L2,
                                         ::arm_compute::Size2D{param.kw, param.kh},
                                         asPadStrideInfo(param.padding, param.stride)};

    if (::internal::arm_compute::isGpuMode())
    {
      std::unique_ptr<::arm_compute::CLPoolingLayer> fn{new ::arm_compute::CLPoolingLayer};

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), info);

      builder.append("L2Pool2D", std::move(fn));
    }
    else
    {
      std::unique_ptr<::arm_compute::NEPoolingLayer> fn{new ::arm_compute::NEPoolingLayer};

      fn->configure(ifm_alloc, ofm_alloc, info);

      builder.append("L2Pool2D", std::move(fn));
    }

    ActivationBuilder{builder}.append(param.activation, ofm_alloc);
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::EmbeddingLookup::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index lookups_index{node.param().lookups_index};
  const ::internal::tflite::operand::Index values_index{node.param().values_index};

  const auto &output_obj = _ctx.at(output_index);
  const auto &lookups_obj = _ctx.at(lookups_index);
  const auto &values_obj = _ctx.at(values_index);

  // Verify operand here, not at configure() to avoid acl's modifying
  // TensorShape sometimes(Issue: https://github.sec.samsung.net/STAR/nnfw/issues/729)
  {
    assert(lookups_obj.type() == ANEURALNETWORKS_TENSOR_INT32);

    const auto &output_shape = output_obj.shape();
    const auto &lookups_shape = lookups_obj.shape();
    const auto &values_shape = values_obj.shape();

    assert(lookups_shape.rank() == 1);
    assert(values_shape.rank() >= 2);

    // output should be a n-D tensor with the same rank and shape as the values tensor, except for
    // the first dimension which has the same size as lookups' only dimension.
    assert(output_shape.rank() == values_shape.rank());
    assert(output_shape.dim(0) == lookups_shape.dim(0));
    for (size_t n = 1; n < output_shape.rank(); ++n)
    {
      assert(output_shape.dim(n) == values_shape.dim(n));
    }
  }

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(output_obj.shape(), false), output_obj.type(),
                                       output_obj.scale(), output_obj.zeroPoint()));
  _builder.addShapeConstr(lookups_index,
                          asTensorInfo(asTensorShape(lookups_obj.shape()), lookups_obj.type(),
                                       lookups_obj.scale(), lookups_obj.zeroPoint()));
  _builder.addShapeConstr(values_index,
                          asTensorInfo(asTensorShape(values_obj.shape(), false), values_obj.type(),
                                       values_obj.scale(), values_obj.zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int32_t output_index;
    int32_t lookups_index;
    int32_t values_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.lookups_index = lookups_index.asInt();
  param.values_index = values_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto lookups_alloc = ctx.at(::internal::tflite::operand::Index{param.lookups_index});
    auto values_alloc = ctx.at(::internal::tflite::operand::Index{param.values_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLEmbeddingLookup>();

      fn->configure(CAST_CL(values_alloc), CAST_CL(output_alloc), CAST_CL(lookups_alloc));

      builder.append("EmbeddingLookup", std::move(fn));
    }
    else
    {
      // TODO Enable NEON Support
      throw std::runtime_error("Not supported, yet");
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::HashtableLookup::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index hits_index{node.param().hits_index};
  const ::internal::tflite::operand::Index lookups_index{node.param().lookups_index};
  const ::internal::tflite::operand::Index values_index{node.param().values_index};
  const ::internal::tflite::operand::Index keys_index{node.param().keys_index};

  const auto &lookups_obj = _ctx.at(lookups_index);
  const auto &keys_obj = _ctx.at(keys_index);
  const auto &hits_obj = _ctx.at(hits_index);
  const auto &values_obj = _ctx.at(values_index);
  const auto &output_obj = _ctx.at(output_index);

  assert(lookups_obj.type() == ANEURALNETWORKS_TENSOR_INT32);
  assert(keys_obj.type() == ANEURALNETWORKS_TENSOR_INT32);
  assert(hits_obj.type() == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM);

  const auto &lookups_shape = lookups_obj.shape();
  const auto &keys_shape = keys_obj.shape();
  const auto &hits_shape = hits_obj.shape();
  const auto &values_shape = values_obj.shape();
  const auto &output_shape = output_obj.shape();

  assert(values_shape.rank() == output_shape.rank());

  assert(lookups_shape.rank() == 1);
  assert(keys_shape.rank() == 1);
  assert(values_shape.dim(0) == keys_shape.dim(0));
  assert(lookups_shape.dim(0) == output_shape.dim(0));

  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(hits_index,
                          asTensorInfo(asTensorShape(_ctx.at(hits_index).shape()),
                                       _ctx.at(hits_index).type(), _ctx.at(hits_index).type(),
                                       _ctx.at(hits_index).zeroPoint()));

  _builder.addShapeConstr(lookups_index, asTensorInfo(asTensorShape(_ctx.at(lookups_index).shape()),
                                                      _ctx.at(lookups_index).type(),
                                                      _ctx.at(lookups_index).scale(),
                                                      _ctx.at(lookups_index).zeroPoint()));
  _builder.addShapeConstr(values_index,
                          asTensorInfo(asTensorShape(_ctx.at(values_index).shape()),
                                       _ctx.at(values_index).type(), _ctx.at(values_index).scale(),
                                       _ctx.at(values_index).zeroPoint()));
  _builder.addShapeConstr(keys_index,
                          asTensorInfo(asTensorShape(_ctx.at(keys_index).shape()),
                                       _ctx.at(keys_index).type(), _ctx.at(keys_index).scale(),
                                       _ctx.at(keys_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int32_t output_index;
    int32_t hits_index;
    int32_t lookups_index;
    int32_t values_index;
    int32_t keys_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.hits_index = hits_index.asInt();
  param.lookups_index = lookups_index.asInt();
  param.values_index = values_index.asInt();
  param.keys_index = keys_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto hits_alloc = ctx.at(::internal::tflite::operand::Index{param.hits_index});
    auto lookups_alloc = ctx.at(::internal::tflite::operand::Index{param.lookups_index});
    auto values_alloc = ctx.at(::internal::tflite::operand::Index{param.values_index});
    auto keys_alloc = ctx.at(::internal::tflite::operand::Index{param.keys_index});

    if (::internal::arm_compute::isGpuMode()) // GPU
    {
      auto fn = std::make_unique<::arm_compute::CLHashtableLookup>();

      fn->configure(CAST_CL(lookups_alloc), CAST_CL(keys_alloc), CAST_CL(values_alloc),
                    CAST_CL(output_alloc), CAST_CL(hits_alloc));

      builder.append("HashtableLookup", std::move(fn));
    }
    else // NEON
    {
      // TODO Enable NEON Support
      throw std::runtime_error("Not supported, yet");
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::LocalResponseNormalization::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index radius_index{node.param().radius_index};
  const ::internal::tflite::operand::Index bias_index{node.param().bias_index};
  const ::internal::tflite::operand::Index alpha_index{node.param().alpha_index};
  const ::internal::tflite::operand::Index beta_index{node.param().beta_index};

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
    int32_t radius;
    float bias;
    float alpha;
    float beta;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  param.radius = _ctx.at(radius_index).asScalar<int32_t>();
  param.alpha = _ctx.at(alpha_index).asScalar<float>();
  param.beta = _ctx.at(beta_index).asScalar<float>();
  param.bias = _ctx.at(bias_index).asScalar<float>();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    const auto norm_info = ::arm_compute::NormalizationLayerInfo(::arm_compute::NormType::CROSS_MAP,
                                                                 param.radius * 2 + 1, param.alpha,
                                                                 param.beta, param.bias, false);
    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLNormalizationLayer>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), norm_info);

      builder.append("LocalResponseNormalization", std::move(fn));
    }
    else
    {
      auto fn = std::make_unique<::arm_compute::NENormalizationLayer>();

      fn->configure(ifm_alloc, ofm_alloc, norm_info);

      builder.append("LocalResponseNormalization", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::DepthToSpace::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};
  const ::internal::tflite::operand::Index block_size_index{node.param().block_size_index};

  assert(_ctx.at(input_index).shape().rank() == 4);
  assert(_ctx.at(output_index).shape().rank() == 4);

  int32_t block_size = _ctx.at(block_size_index).asScalar<int32_t>();
  assert(block_size > 0);

  { // assertions block
    const auto output_shape = _ctx.at(output_index).shape();
    const auto input_shape = _ctx.at(input_index).shape();
    assert(output_shape.dim(0) == input_shape.dim(0));
    assert(output_shape.dim(1) == input_shape.dim(1) * block_size);
    assert(output_shape.dim(2) == input_shape.dim(2) * block_size);
    assert(input_shape.dim(3) % (block_size * block_size) == 0);
    assert(output_shape.dim(3) == input_shape.dim(3) / (block_size * block_size));
  }

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape(), false),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(input_index,
                          asTensorInfo(asTensorShape(_ctx.at(input_index).shape(), false),
                                       _ctx.at(input_index).type(), _ctx.at(input_index).scale(),
                                       _ctx.at(input_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int output_index;
    int input_index;
    int32_t block_size;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();
  param.block_size = block_size;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});

    {
      if (::internal::arm_compute::isGpuMode()) // GPU
      {
        auto fn = std::make_unique<::arm_compute::CLDepthToSpace>();

        fn->configure(CAST_CL(input_alloc), CAST_CL(output_alloc), param.block_size);

        builder.append("DepthToSpace", std::move(fn));
      }
      else // NEON
      {
        // TODO Enable NEON Support
        throw std::runtime_error("Not supported, yet");
      }
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Unpack::Node &node)
{
  VERBOSE(Unpack) << "Configure Unpack operation" << std::endl;
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  uint32_t input_rank = _ctx.at(ifm_index).shape().rank();

  assert(input_rank == 4 || input_rank == 3 || input_rank == 2);
  _builder.addShapeConstr(ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()),
                                                  _ctx.at(ifm_index).type()));

  int32_t axis =
      _ctx.at(::internal::tflite::operand::Index{node.param().axis_index}).asScalar<int32_t>();
  // Negatige axis is supported, -1 implies R-1 axis where R is input rank
  if (axis < 0)
  {
    axis += input_rank;
    assert(axis >= 0);
  }
  uint32_t axis_uint = ToARMComputeAxis(input_rank, axis).value();
  // int32_t num_split =
  // _ctx.at(::internal::tflite::operand::Index{node.param().num_split_index}).asScalar<int32_t>();

  for (const auto &index : node.param().ofm_indexes)
  {
    const ::internal::tflite::operand::Index ofm_index{index};
    _builder.addShapeConstr(ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()),
                                                    _ctx.at(ofm_index).type()));
  }

  struct Param
  {
    std::vector<int32_t> ofm_indexes;
    int ifm_index;
    uint32_t axis;
  };

  if (input_rank == 4)
  {
    // TODO: generate test case for this and generalize 4D method all cases.
    throw std::runtime_error("UNPACK_4D not implemented");
  }
  else if (input_rank == 3)
  {
    Param param;
    param.ifm_index = ifm_index.asInt();
    param.axis = axis_uint;
    for (const auto &index : node.param().ofm_indexes)
    {
      param.ofm_indexes.push_back(index);
    }

    auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
      auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

      if (::internal::arm_compute::isGpuMode())
      {
        auto fn = std::make_unique<::arm_compute::CLUnstack>();
        std::vector<::arm_compute::ICLTensor *> outputs;
        for (const auto &index : param.ofm_indexes)
        {
          auto output_alloc = ctx.at(::internal::tflite::operand::Index{index});
          outputs.push_back(CAST_CL(output_alloc));
        }
        fn->configure(CAST_CL(input_alloc), outputs, param.axis);

        builder.append("Unpack", std::move(fn));
      }
      else
        throw std::runtime_error("Not supported, yet");
    };

    _builder.addStage(stage);
  }
  else if (input_rank == 2)
  {
    throw std::runtime_error("UNPACK_2D not implemented");
  }
  else
  {
    throw std::runtime_error("UNPACK axis is not valid");
  }
}

void Planner::visit(const ::internal::tflite::op::Pack::Node &node)
{
  VERBOSE(Pack) << "Configure Pack operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const uint32_t output_rank = _ctx.at(ofm_index).shape().rank();
  const uint32_t input_rank = output_rank - 1;

  assert(output_rank == 4 || output_rank == 3 || output_rank == 2);

  for (const auto &index : node.param().ifm_indexes)
  {
    const ::internal::tflite::operand::Index ifm_index{index};
    assert(_ctx.at(ifm_index).shape().rank() == input_rank);
    _builder.addShapeConstr(ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()),
                                                    _ctx.at(ifm_index).type()));
  }

  _builder.addShapeConstr(ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()),
                                                  _ctx.at(ofm_index).type()));

  int32_t axis =
      _ctx.at(::internal::tflite::operand::Index{node.param().axis_index}).asScalar<int32_t>();
  // A negative axis implies axis from the end.
  // For example, axis = -1 implies the first axis from the end, i.e. axis = Rank - 1.
  // Similarly, axis = -2 imples second axis from the end, i.e. axis = Rank - 2.
  if (axis < 0)
  {
    axis += output_rank;
    assert(axis >= 0);
  }
  uint32_t axis_uint = ToARMComputeAxis(output_rank, axis).value();

  struct Param
  {
    std::vector<int32_t> ifm_indexes;
    int ofm_index;
    uint32_t axis;
  };

  if (input_rank == 3)
  {
    // TODO: generate test case for this and generalize 4D method all cases.
    throw std::runtime_error("PACK_3D not implemented");
  }
  else if (input_rank == 2)
  {
    Param param;
    param.ofm_index = ofm_index.asInt();
    param.axis = axis_uint;

    for (const auto &index : node.param().ifm_indexes)
    {
      param.ifm_indexes.push_back(index);
    }

    auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
      auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});

      if (::internal::arm_compute::isGpuMode())
      {
        auto fn = std::make_unique<::arm_compute::CLStackLayer>();
        std::vector<::arm_compute::ICLTensor *> inputs;
        for (const auto &index : param.ifm_indexes)
        {
          auto input_alloc = ctx.at(::internal::tflite::operand::Index{index});
          inputs.push_back(CAST_CL(input_alloc));
        }
        fn->configure(inputs, param.axis, CAST_CL(output_alloc));

        builder.append("Pack", std::move(fn));
      }
      else
        throw std::runtime_error("Not supported, yet");
    };

    _builder.addStage(stage);
  }
  else if (input_rank == 1)
  {
    throw std::runtime_error("PACK_1D not implemented");
  }
  else
  {
    throw std::runtime_error("PACK axis is not valid");
  }
}

void Planner::visit(const ::internal::tflite::op::Neg::Node &node)
{
  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int ofm_index;
    int ifm_index;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLNeg>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc));
      builder.append("Neg", std::move(fn));
    }
    else
    {
      // TODO Enable NEON Support
      throw std::runtime_error("Not supported, yet");
    }

  };
  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Exp::Node &node)
{
  VERBOSE(Exp) << "Configure Exp operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};

  // Set shape constraints
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  struct Param
  {
    int ofm_index;
    int ifm_index;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLExpLayer>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc));

      builder.append("Exp", std::move(fn));
    }
    else
    {
      throw std::runtime_error("Not supported");
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::ReduceSum::Node &node)
{
  VERBOSE(ReduceSum) << "Configure ReduceSum operation" << std::endl;

  const ::internal::tflite::operand::Index ofm_index{node.param().ofm_index};
  const ::internal::tflite::operand::Index ifm_index{node.param().ifm_index};
  const ::internal::tflite::operand::Index axis_index{node.param().axis_index};

  const auto ifm_shape = _ctx.at(ifm_index).shape();
  const auto ofm_shape = _ctx.at(ofm_index).shape();
  const auto axis_shape = _ctx.at(axis_index).shape();

  assert(ifm_shape.rank() <= 4);
  assert(ofm_shape.rank() <= ifm_shape.rank());
  assert(_ctx.at(axis_index).hasData());
  assert(axis_shape.rank() == 0 || axis_shape.rank() == 1);

  // NOTE For the 4-dimensions, if the rank of input and output are different, this runtime only
  // supports cases reducing height and width or reducing depth.
  // TODO We have to support all cases of dimensions up to 4.
  // For correct permuting, we have to set output's shape to be equal in dimension position of the
  // input. But the positions of the same dimensions in the input and output may be set differently.
  // For example {2,3,4,5}(input's shape) can be reduced to {3,5}(output's shape). The original
  // output shape should be {1,3,1,5}, but real output shape may be {3,5}. If you simply try to
  // extend it in 4 dimensions, it should be {1,1,3,5}.
  // Even if output shape is changed to {1,3,1,5}, there is another problem. It is that shape of
  // output tensor used at next operation is changed to {1,3,1,5} after this operation even if the
  // next operation is not desired.
  if (ifm_shape.rank() == 4 && ifm_shape.rank() != ofm_shape.rank())
  {
    if (ofm_shape.rank() == 2)
    {
      // Reducing HW
      assert(ifm_shape.dim(0) == ofm_shape.dim(0) && ifm_shape.dim(3) == ofm_shape.dim(1));
    }
    else if (ofm_shape.rank() == 3)
    {
      // Reducing C or
      // (Reducing H and C(ifm and ofm) == 1) or (Reducing W and C(ifm and ofm) == 1)
      assert(ifm_shape.dim(0) == ofm_shape.dim(0) && ifm_shape.dim(1) == ofm_shape.dim(1) &&
                 ifm_shape.dim(2) == ofm_shape.dim(2) ||
             (ifm_shape.dim(0) == ofm_shape.dim(0) &&
              (ifm_shape.dim(1) == ofm_shape.dim(1) || ifm_shape.dim(2) == ofm_shape.dim(1)) &&
              ifm_shape.dim(3) == 1 && ofm_shape.dim(2) == 1));
    }
  }

  // Set shape constraints
  _builder.addShapeConstr(
      ofm_index, asTensorInfo(asTensorShape(_ctx.at(ofm_index).shape()), _ctx.at(ofm_index).type(),
                              _ctx.at(ofm_index).scale(), _ctx.at(ofm_index).zeroPoint()));
  _builder.addShapeConstr(
      ifm_index, asTensorInfo(asTensorShape(_ctx.at(ifm_index).shape()), _ctx.at(ifm_index).type(),
                              _ctx.at(ifm_index).scale(), _ctx.at(ifm_index).zeroPoint()));

  uint32_t input_rank = ifm_shape.rank();
  std::set<uint32_t> axis;
  int32_t axis_rank = axis_shape.rank();

  if (axis_rank == 0)
  {
    int32_t axis_value = _ctx.at(axis_index).asScalar<int32_t>();
    if (axis_value < 0)
    {
      axis_value += input_rank;
    }
    axis.insert(ToARMComputeAxis(input_rank, axis_value).value());
  }
  else if (axis_rank == 1)
  {
    const auto axis_base = _ctx.at(axis_index).data().base();
    const auto axis_size = _ctx.at(axis_index).shape().asVector();

    // If axis's data does not exist as constant values and can be gotten as input data, we have to
    // find a way to infer output shape when sinking output.
    assert(axis_base != nullptr);
    for (uint32_t n = 0; n < axis_size; ++n)
    {
      int32_t axis_value = *(reinterpret_cast<const int32_t *>(axis_base) + n);
      if (axis_value < 0)
      {
        axis_value += input_rank;
      }
      axis.insert(ToARMComputeAxis(input_rank, axis_value).value());
    }
  }
  else
  {
    throw std::runtime_error("Not supported axis");
  }

  struct Param
  {
    int ofm_index;
    int ifm_index;
    std::set<uint32_t> axis;
  };

  Param param;

  param.ofm_index = ofm_index.asInt();
  param.ifm_index = ifm_index.asInt();
  param.axis = axis;

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto ofm_alloc = ctx.at(::internal::tflite::operand::Index{param.ofm_index});
    auto ifm_alloc = ctx.at(::internal::tflite::operand::Index{param.ifm_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLReduceOperation>();

      fn->configure(CAST_CL(ifm_alloc), CAST_CL(ofm_alloc), param.axis,
                    ::arm_compute::ReduceOperation::SUM);

      builder.append("ReduceSum", std::move(fn));
    }
    else
      throw std::runtime_error("Not supported, yet");
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::Abs::Node &node)
{
  VERBOSE(Tanh) << "Configure Abs operation" << std::endl;

  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};

  // Set shape constraints
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));
  _builder.addShapeConstr(input_index,
                          asTensorInfo(asTensorShape(_ctx.at(input_index).shape()),
                                       _ctx.at(input_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));

  struct Param
  {
    int output_index;
    int input_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();

  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});

    const ::arm_compute::ActivationLayerInfo act_info{
        ::arm_compute::ActivationLayerInfo::ActivationFunction::ABS};

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLActivationLayer>();

      fn->configure(CAST_CL(input_alloc), CAST_CL(output_alloc), act_info);

      builder.append("Abs", std::move(fn));
    }
    else
    {
      auto fn = std::make_unique<::arm_compute::NEActivationLayer>();

      fn->configure(input_alloc, output_alloc, act_info);

      builder.append("Abs", std::move(fn));
    }
  };

  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::NotEqual::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input1_index{node.param().input1_index};
  const ::internal::tflite::operand::Index input2_index{node.param().input2_index};

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));

  if (!(_ctx.at(input1_index).shape() == _ctx.at(input2_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(input1_index).shape().rank(), _ctx.at(input2_index).shape().rank());
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(input1_index).shape())
        .extendRank(broadcast_rank);
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(input2_index).shape())
        .extendRank(broadcast_rank);
  }
  _builder.addShapeConstr(input1_index,
                          asTensorInfo(asTensorShape(_ctx.at(input1_index).shape()),
                                       _ctx.at(input1_index).type(), _ctx.at(input1_index).scale(),
                                       _ctx.at(input1_index).zeroPoint()));
  _builder.addShapeConstr(input2_index,
                          asTensorInfo(asTensorShape(_ctx.at(input2_index).shape()),
                                       _ctx.at(input2_index).type(), _ctx.at(input2_index).scale(),
                                       _ctx.at(input2_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int output_index;
    int input1_index;
    int input2_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input1_index = input1_index.asInt();
  param.input2_index = input2_index.asInt();
  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input1_alloc = ctx.at(::internal::tflite::operand::Index{param.input1_index});
    auto input2_alloc = ctx.at(::internal::tflite::operand::Index{param.input2_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLComparison>();

      fn->configure(CAST_CL(input1_alloc), CAST_CL(input2_alloc), CAST_CL(output_alloc),
                    ::arm_compute::ComparisonOperation::NotEqual);

      builder.append("NotEqual", std::move(fn));
    }
    else
    {
      // TODO Add NEON support

      throw std::runtime_error("Not supported yet");
    }
  };
  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::LogicalAnd::Node &node)
{
  VERBOSE(Logical_AND) << "Configure Logical_AND operation" << std::endl;

  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input1_index{node.param().input1_index};
  const ::internal::tflite::operand::Index input2_index{node.param().input2_index};

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));

  if (!(_ctx.at(input1_index).shape() == _ctx.at(input2_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(input1_index).shape().rank(), _ctx.at(input2_index).shape().rank());
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(input1_index).shape())
        .extendRank(broadcast_rank);
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(input2_index).shape())
        .extendRank(broadcast_rank);
  }
  _builder.addShapeConstr(input1_index,
                          asTensorInfo(asTensorShape(_ctx.at(input1_index).shape()),
                                       _ctx.at(input1_index).type(), _ctx.at(input1_index).scale(),
                                       _ctx.at(input1_index).zeroPoint()));
  _builder.addShapeConstr(input2_index,
                          asTensorInfo(asTensorShape(_ctx.at(input2_index).shape()),
                                       _ctx.at(input2_index).type(), _ctx.at(input2_index).scale(),
                                       _ctx.at(input2_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int output_index;
    int input1_index;
    int input2_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input1_index = input1_index.asInt();
  param.input2_index = input2_index.asInt();
  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input1_alloc = ctx.at(::internal::tflite::operand::Index{param.input1_index});
    auto input2_alloc = ctx.at(::internal::tflite::operand::Index{param.input2_index});

    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLBinaryLogicalOp>();

      fn->configure(CAST_CL(input1_alloc), CAST_CL(input2_alloc), CAST_CL(output_alloc),
                    ::arm_compute::BinaryLogicalOperation::AND);

      builder.append("LogicalAnd", std::move(fn));
    }
    else
    {
      // TODO Add NEON support

      throw std::runtime_error("Not supported yet");
    }
  };
  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::LogicalNot::Node &node)
{
  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input_index{node.param().input_index};

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       ::arm_compute::DataType::U8, _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));

  _builder.addShapeConstr(input_index,
                          asTensorInfo(asTensorShape(_ctx.at(input_index).shape()),
                                       ::arm_compute::DataType::U8, _ctx.at(input_index).scale(),
                                       _ctx.at(input_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int output_index;
    int input_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input_index = input_index.asInt();
  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input_alloc = ctx.at(::internal::tflite::operand::Index{param.input_index});
    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLBitwiseNot>();

      fn->configure(CAST_CL(input_alloc), CAST_CL(output_alloc));

      builder.append("LogicalNot", std::move(fn));
    }
    else
    {
      // TODO Add NEON support

      throw std::runtime_error("Not supported yet");
    }
  };
  _builder.addStage(stage);
}

void Planner::visit(const ::internal::tflite::op::LogicalOr::Node &node)
{
  VERBOSE(LogicalOr) << "Configure LogicalOr operation" << std::endl;

  const ::internal::tflite::operand::Index output_index{node.param().output_index};
  const ::internal::tflite::operand::Index input1_index{node.param().input1_index};
  const ::internal::tflite::operand::Index input2_index{node.param().input2_index};

  // Set Shape Constraints and TensorInfo
  _builder.addShapeConstr(output_index,
                          asTensorInfo(asTensorShape(_ctx.at(output_index).shape()),
                                       _ctx.at(output_index).type(), _ctx.at(output_index).scale(),
                                       _ctx.at(output_index).zeroPoint()));

  if (!(_ctx.at(input1_index).shape() == _ctx.at(input2_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(input1_index).shape().rank(), _ctx.at(input2_index).shape().rank());
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(input1_index).shape())
        .extendRank(broadcast_rank);
    const_cast<::internal::tflite::operand::Shape &>(_ctx.at(input2_index).shape())
        .extendRank(broadcast_rank);
  }

  _builder.addShapeConstr(input1_index,
                          asTensorInfo(asTensorShape(_ctx.at(input1_index).shape()),
                                       _ctx.at(input1_index).type(), _ctx.at(input1_index).scale(),
                                       _ctx.at(input1_index).zeroPoint()));
  _builder.addShapeConstr(input2_index,
                          asTensorInfo(asTensorShape(_ctx.at(input2_index).shape()),
                                       _ctx.at(input2_index).type(), _ctx.at(input2_index).scale(),
                                       _ctx.at(input2_index).zeroPoint()));

  // Construct operation parameters
  struct Param
  {
    int output_index;
    int input1_index;
    int input2_index;
  };

  Param param;

  param.output_index = output_index.asInt();
  param.input1_index = input1_index.asInt();
  param.input2_index = input2_index.asInt();
  auto stage = [param](const IAllocationContext &ctx, IExecutionBuilder &builder) {
    auto output_alloc = ctx.at(::internal::tflite::operand::Index{param.output_index});
    auto input1_alloc = ctx.at(::internal::tflite::operand::Index{param.input1_index});
    auto input2_alloc = ctx.at(::internal::tflite::operand::Index{param.input2_index});
    if (::internal::arm_compute::isGpuMode())
    {
      auto fn = std::make_unique<::arm_compute::CLBinaryLogicalOp>();

      fn->configure(CAST_CL(input1_alloc), CAST_CL(input2_alloc), CAST_CL(output_alloc),
                    ::arm_compute::BinaryLogicalOperation::OR);

      builder.append("LogicalOr", std::move(fn));
    }
    else
    {
      // TODO Add NEON support

      throw std::runtime_error("Not supported yet");
    }
  };
  _builder.addStage(stage);
}

class AllocationContext final : public IAllocationContext
{
public:
  AllocationContext(::internal::arm_compute::Plan &plan) : _plan{plan}
  {
    // DO NOTHING
  }

public:
  ::arm_compute::ITensor *at(const ::internal::tflite::operand::Index &ind) const override
  {
    return _plan.operands().at(ind).ptr();
  }

private:
  ::internal::arm_compute::Plan &_plan;
};

class ExecutionBuilder final : public IExecutionBuilder
{
public:
  ExecutionBuilder(::internal::arm_compute::Plan &plan) : _plan{plan}
  {
    // DO NOTHING
  }

public:
  void append(const std::string &name, std::unique_ptr<::arm_compute::IFunction> &&f) override
  {
    _plan.operations().append(std::move(f));
    _plan.operations().at(_plan.operations().size() - 1).name() = name;
  }

#ifdef TFLITE_PROFILING_ENABLED
public:
  int plan_op_size() const { return _plan.operations().size(); }
  void addOpIndexToSteps(int from, int to, int op_idx)
  {
    for (int i = from; i < to; ++i)
      _plan.operations().at(i).op_idx() = op_idx;
  }
#endif

private:
  ::internal::arm_compute::Plan &_plan;
};

/**
 * @brief Class to provide methods of compilation plan builder
 */
class PlanBuilder final : public IPlanBuilder
{
public:
  /**
   * @brief Construct a new PlanBuilder object with Plan
   * @param [in] plan  The Plan object
   */
  PlanBuilder(::internal::arm_compute::Plan &plan) : _plan{plan}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief  Add TensorInfo with Shape Constraints
   * @param [in] ind   Index of operand
   * @param [in] info  TensorInfo value to set to index of operand
   * @return  N/A
   */
  void addShapeConstr(const ::internal::tflite::operand::Index &ind,
                      const ::arm_compute::TensorInfo &info) override;

public:
  /**
   * @brief  Add Subsumption constraints
   * @param [in] ind  Index of operand
   * @param [in] base  Index of base operand of Subsumption
   * @param [in] offset  Offset of Subsumption
   * @param [in] shape  Shape of Subsumption
   * @param [in] extend_parent  extend_parent value of Subsumption
   * @return  N/A
   */
  void addSubsumptionConstr(const ::internal::tflite::operand::Index &ind,
                            const ::internal::tflite::operand::Index &base,
                            const ::arm_compute::Coordinates &offset,
                            const ::arm_compute::TensorShape &shape, bool extend_parent) override;

public:
  /**
   * @brief  Add Initializer lambda with ITensor param
   * @param [in] ind  Index of operand
   * @param [in] initializer  Initializer to add
   * @return  N/A
   */
  void addInitializer(const ::internal::tflite::operand::Index &ind,
                      const Initializer &initializer) override;

public:
  /**
   * @brief  Add Stage lambda with IAllocationContext and IExecutionBuilder params
   * @param [in] stage  Stage to add
   * @return  N/A
   */
  void addStage(const Stage &stage) override;

public:
  /**
   * @brief  Finilize(build) the Plan
   * @return  N/A
   */
  void finalize(void) const;

private:
  ::internal::arm_compute::Plan &_plan;

private:
  struct Subsumption
  {
  public:
    Subsumption(const ::internal::tflite::operand::Index &base,
                const ::arm_compute::Coordinates &offset, const ::arm_compute::TensorShape &shape,
                bool extend_parent)
        : _base{base}, _offset{offset}, _shape{shape}, _extend_parent{extend_parent}
    {
      // DO NOTHING
    }

  public:
    const ::internal::tflite::operand::Index &base(void) const { return _base; }
    const ::arm_compute::Coordinates &offset(void) const { return _offset; }
    const ::arm_compute::TensorShape &shape(void) const { return _shape; }
    const bool extend_parent(void) const { return _extend_parent; }

  private:
    const ::internal::tflite::operand::Index _base;
    const ::arm_compute::Coordinates _offset;
    const ::arm_compute::TensorShape _shape;
    const bool _extend_parent;
  };

private:
  std::map<int, ::arm_compute::TensorInfo> _tensor_info_ctx;
  std::map<int, std::shared_ptr<Subsumption>> _subsumption_ctx;
  std::map<int, Initializer> _initializer_ctx;
  std::vector<Stage> _stages;
};

void PlanBuilder::addShapeConstr(const ::internal::tflite::operand::Index &ind,
                                 const ::arm_compute::TensorInfo &info)
{
  _tensor_info_ctx[ind.asInt()] = info;
}

void PlanBuilder::addSubsumptionConstr(const ::internal::tflite::operand::Index &ind,
                                       const ::internal::tflite::operand::Index &base,
                                       const ::arm_compute::Coordinates &offset,
                                       const ::arm_compute::TensorShape &shape, bool extend_parent)
{
  _subsumption_ctx[ind.asInt()] = std::make_shared<Subsumption>(base, offset, shape, extend_parent);
}

void PlanBuilder::addInitializer(const ::internal::tflite::operand::Index &ind,
                                 const Initializer &initializer)
{
  _initializer_ctx[ind.asInt()] = initializer;
}

void PlanBuilder::addStage(const Stage &stage) { _stages.emplace_back(stage); }

#include <stack>

void PlanBuilder::finalize(void) const
{
  // ITensor objects to be initialized later
  std::vector<std::shared_ptr<::arm_compute::ITensor>> tensors;

  // Create Tensor & CLSubTensor
  auto isAllocated = [this](int ind) {
    const ::internal::tflite::operand::Index operand_index{ind};
    return _plan.operands().exist(operand_index);
  };

  auto setCLTensor = [&](int ind) {
    auto tensor = std::make_shared<::arm_compute::CLTensor>();

    tensor->allocator()->init(_tensor_info_ctx.at(ind));

    // NOTE Do NOT allocate here. allocate should be invoked after configure functions
    _plan.operands().set(::internal::tflite::operand::Index{ind}, tensor);
    tensors.emplace_back(tensor);
  };

  auto setCLSubTensor = [&](int curr) {
    const auto &sub_info = *(_subsumption_ctx.find(curr)->second);

    auto base_tensor = _plan.operands().at(sub_info.base()).ptr();

    assert(base_tensor != nullptr);

    auto curr_tensor = std::make_shared<::arm_compute::CLSubTensor>(
        CAST_CL(base_tensor), sub_info.shape(), sub_info.offset(), sub_info.extend_parent());

    _plan.operands().set(::internal::tflite::operand::Index{curr}, curr_tensor);
  };

  auto setNETensor = [&](int ind) {
    auto tensor = std::make_shared<::arm_compute::Tensor>();

    tensor->allocator()->init(_tensor_info_ctx.at(ind));

    // NOTE Do NOT allocate here. allocate should be invoked after configure functions
    _plan.operands().set(::internal::tflite::operand::Index{ind}, tensor);
    tensors.emplace_back(tensor);
  };

  auto setNESubTensor = [&](int curr) {
    const auto &sub_info = *(_subsumption_ctx.find(curr)->second);

    auto base_tensor = _plan.operands().at(sub_info.base()).ptr();

    assert(base_tensor != nullptr);

    auto curr_tensor = std::make_shared<::arm_compute::SubTensor>(base_tensor, sub_info.shape(),
                                                                  sub_info.offset());

    _plan.operands().set(::internal::tflite::operand::Index{curr}, curr_tensor);
  };

  for (auto it = _subsumption_ctx.begin(); it != _subsumption_ctx.end(); ++it)
  {
    std::stack<int> stack;

    stack.push(it->first);

    while (!stack.empty())
    {
      const auto curr = stack.top();

      if (isAllocated(curr))
      {
        // Skip if already allocated
        stack.pop();
        continue;
      }

      auto it_s = _subsumption_ctx.find(curr);

      if (it_s == _subsumption_ctx.end())
      {
        if (::internal::arm_compute::isGpuMode())
          setCLTensor(curr);
        else
          setNETensor(curr);
        stack.pop();
        continue;
      }

      const auto &sub_info = *(it_s->second);

      if (isAllocated(sub_info.base().asInt()))
      {
        if (::internal::arm_compute::isGpuMode())
          setCLSubTensor(curr);
        else
          setNESubTensor(curr);
        stack.pop();
      }
      else
      {
        // Allocate base tensor first
        stack.push(sub_info.base().asInt());
      }
    }
  }

  for (auto it = _tensor_info_ctx.begin(); it != _tensor_info_ctx.end(); ++it)
  {
    if (isAllocated(it->first))
    {
      // Skip if already allocated
      continue;
    }

    if (::internal::arm_compute::isGpuMode())
      setCLTensor(it->first);
    else
      setNETensor(it->first);
  }

  // Process Stage
  AllocationContext allocation_context{_plan};
  ExecutionBuilder execution_builder{_plan};

  for (int idx = 0; idx < _stages.size(); idx++)
  {
    const auto &stage = _stages[idx];
#ifdef TFLITE_PROFILING_ENABLED
    int from = execution_builder.plan_op_size();
#endif
    stage(allocation_context, execution_builder);
#ifdef TFLITE_PROFILING_ENABLED
    int to = execution_builder.plan_op_size();
    execution_builder.addOpIndexToSteps(from, to, idx);
#endif
  }

  // Allocate Tensor Memory
  for (const auto &tensor : tensors)
  {
    if (::internal::arm_compute::isGpuMode())
    {
      auto cl_tensor = CAST_CL(tensor.get());
      cl_tensor->allocator()->allocate();
    }
    else
    {
      auto ne_tensor = CAST_NE(tensor.get());
      ne_tensor->allocator()->allocate();
    }
  }

  // Fill weight/bias
  for (auto it = _initializer_ctx.begin(); it != _initializer_ctx.end(); ++it)
  {
    const ::internal::tflite::operand::Index operand_index{it->first};
    _plan.operands().at(operand_index).access(it->second);
  }

  // Initialize CLTensors that have data in their corresponding NNAPI operand but are not
  // initialized yet
  const auto &operands = _plan.model().operands();
  for (int idx = 0; idx < operands.size(); ++idx)
  {
    const ::internal::tflite::operand::Index operand_idx{idx};
    if (isAllocated(idx) && operands.at(operand_idx).hasData() &&
        _initializer_ctx.find(idx) == _initializer_ctx.end())
    {
      auto rank = operands.at(operand_idx).shape().rank();
      auto base = operands.at(operand_idx).data().base();
      auto type = operands.at(operand_idx).type();
      auto shape = operands.at(operand_idx).shape();

      // Need to support scalar types (ANEURALNETWORKS_FLOAT32 and ANEURALNETWORKS_INT32)
      // for rank > 1 tensor, because it can be operand of broadcast operation
      switch (rank)
      {
        case 0: // scalar
        {
          switch (type)
          {
            case ANEURALNETWORKS_FLOAT32:
            case ANEURALNETWORKS_TENSOR_FLOAT32:
            {
              auto initializer = std::bind(initVectorTensor<float>, _1, base, 1);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_INT32:
            case ANEURALNETWORKS_TENSOR_INT32:
            {
              auto initializer = std::bind(initVectorTensor<int32_t>, _1, base, 1);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_UINT32:
            {
              auto initializer = std::bind(initVectorTensor<uint32_t>, _1, base, 1);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
            {
              auto initializer = std::bind(initVectorTensor<uint8_t>, _1, base, 1);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            default:
              throw std::runtime_error("Unknown scalar type, type : " + std::to_string(type));
              break;
          }
          break;
        }
        case 1: // vector
        {
          auto size = shape.asVector();
          switch (type)
          {
            case ANEURALNETWORKS_FLOAT32:
            case ANEURALNETWORKS_TENSOR_FLOAT32:
            {
              auto initializer = std::bind(initVectorTensor<float>, _1, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_INT32:
            case ANEURALNETWORKS_TENSOR_INT32:
            {
              auto initializer = std::bind(initVectorTensor<int32_t>, _1, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
            {
              auto initializer = std::bind(initVectorTensor<uint8_t>, _1, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            default:
              throw std::runtime_error("Unknown tensor type, type : " + std::to_string(type));
              break;
          }
          break;
        }
        case 2: // matrix
        {
          const auto matrix_shape = shape.asMatrix();
          auto size = operands.at(operand_idx).data().size();
          switch (type)
          {
            case ANEURALNETWORKS_FLOAT32:
            case ANEURALNETWORKS_TENSOR_FLOAT32:
            {
              auto initializer = std::bind(initMatrixTensor<float>, _1, matrix_shape, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_INT32:
            case ANEURALNETWORKS_TENSOR_INT32:
            {
              auto initializer = std::bind(initMatrixTensor<int32_t>, _1, matrix_shape, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
            {
              auto initializer = std::bind(initMatrixTensor<uint8_t>, _1, matrix_shape, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            default:
              throw std::runtime_error("Unknown tensor type, type : " + std::to_string(type));
              break;
          }
          break;
        }
        case 3: // 3D tensor
        {
          const auto tensor_shape = shape.asTensor();
          auto size = operands.at(operand_idx).data().size();
          switch (type)
          {
            case ANEURALNETWORKS_FLOAT32:
            case ANEURALNETWORKS_TENSOR_FLOAT32:
            {
              auto initializer = std::bind(initTensor3D<float>, _1, tensor_shape, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_INT32:
            case ANEURALNETWORKS_TENSOR_INT32:
            {
              auto initializer = std::bind(initTensor3D<int32_t>, _1, tensor_shape, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
            {
              auto initializer = std::bind(initTensor3D<uint8_t>, _1, tensor_shape, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            default:
              throw std::runtime_error("Unknown tensor type, type : " + std::to_string(type));
              break;
          }
          break;
        }
        case 4: // feature
        {
          const auto feature_shape = shape.asFeature();
          auto size = operands.at(operand_idx).data().size();
          switch (type)
          {
            case ANEURALNETWORKS_FLOAT32:
            case ANEURALNETWORKS_TENSOR_FLOAT32:
            {
              auto initializer = std::bind(initFeatureTensor<float>, _1, feature_shape, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_INT32:
            case ANEURALNETWORKS_TENSOR_INT32:
            {
              auto initializer =
                  std::bind(initFeatureTensor<int32_t>, _1, feature_shape, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
            {
              auto initializer =
                  std::bind(initFeatureTensor<uint8_t>, _1, feature_shape, base, size);
              _plan.operands().at(operand_idx).access(initializer);
              break;
            }
            default:
              throw std::runtime_error("Unknown tensor type, type : " + std::to_string(type));
              break;
          }
          break;
        }
        default:
          throw std::runtime_error("Not supported, yet");
          break;
      }
    }
  }
}

//
// NNAPI Implementation
//
int ANeuralNetworksCompilation_create(ANeuralNetworksModel *model,
                                      ANeuralNetworksCompilation **compilation)
{
  if ((model == nullptr) || (compilation == nullptr))
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (!model->isFinished())
  {
    return ANEURALNETWORKS_BAD_STATE;
  }

  std::shared_ptr<const internal::tflite::Model> internal;

  model->release(internal);

  ANeuralNetworksCompilation *compilation_ptr = new ANeuralNetworksCompilation(internal);
  if (compilation_ptr == nullptr)
  {
    return ANEURALNETWORKS_OUT_OF_MEMORY;
  }
  *compilation = compilation_ptr;

  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_setPreference(ANeuralNetworksCompilation *compilation,
                                             int32_t preference)
{
  if (compilation == nullptr)
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  // NOTE Pure CL runimte currently ignores this API call
  // TODO Use preference
  return ANEURALNETWORKS_NO_ERROR;
}

int ANeuralNetworksCompilation_finish(ANeuralNetworksCompilation *compilation)
{
  if (compilation == nullptr)
  {
    return ANEURALNETWORKS_UNEXPECTED_NULL;
  }

  if (::internal::arm_compute::isGpuMode())
  {
    arm_compute::CLScheduler::get().default_init();
    // NOTE CLKernelLibraryEx must use the same context as CLScheduler
    // It did not check whether another device is available.
    arm_compute::CLKernelLibraryEx::get().init(
        "./cl_kernels/", arm_compute::CLScheduler::get().context(), cl::Device::getDefault());
  }

  const auto &operands = compilation->plan().model().operands();
  const auto &operations = compilation->plan().model().operations();

  PlanBuilder plan_builder{compilation->plan()};

  for (uint32_t n = 0; n < operations.size(); ++n)
  {
    operations.at(n).accept(Planner{operands, plan_builder});
  }

  plan_builder.finalize();

  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation *compilation)
{
  delete compilation;
}
