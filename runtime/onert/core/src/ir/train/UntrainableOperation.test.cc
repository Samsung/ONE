/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ir/train/operation/UntrainableOperation.h"

#include "ir/Operations.Include.h"

#include <gtest/gtest.h>

using namespace ::onert::ir;

operation::AddN generateAddN()
{
  return operation::AddN{OperandIndexSequence{1, 2}, OperandIndexSequence{0}};
}

operation::ArgMinMax generateArgMinMax()
{
  operation::ArgMinMax::Param param;
  param.output_type = DataType::FLOAT32;

  return operation::ArgMinMax{OperandIndexSequence{1, 2}, OperandIndexSequence{0}, param};
}

operation::BatchMatMul generateBatchMatMul()
{
  operation::BatchMatMul::Param param;
  param.adj_x = true;
  param.adj_y = true;

  return operation::BatchMatMul{OperandIndexSequence{1, 2}, OperandIndexSequence{0}, param};
}

operation::BatchToSpaceND generateBatchToSpaceND()
{
  return operation::BatchToSpaceND{OperandIndexSequence{1, 2}, OperandIndexSequence{0}};
}

operation::BCQFullyConnected generateBCQFullyConnected()
{
  operation::BCQFullyConnected::Param param;
  param.activation = Activation::NONE;
  param.weights_hidden_size = 1;

  return operation::BCQFullyConnected{OperandIndexSequence{1, 2, 3, 4, 5}, OperandIndexSequence{0},
                                      param};
}

operation::BCQGather generateBCQGather()
{
  operation::BCQGather::Param param;
  param.axis = 0;
  param.input_hidden_size = 1;

  return operation::BCQGather{OperandIndexSequence{1, 2, 3, 4}, OperandIndexSequence{0}, param};
}

operation::BinaryArithmetic generateBinaryArithmetic()
{
  operation::BinaryArithmetic::Param param;
  param.activation = Activation::NONE;
  param.arithmetic_type = operation::BinaryArithmetic::ArithmeticType::ADD;

  return operation::BinaryArithmetic{OperandIndexSequence{1, 2}, OperandIndexSequence{0}, param};
}

operation::BroadcastTo generateBroadcastTo()
{
  return operation::BroadcastTo{OperandIndexSequence{1, 2}, OperandIndexSequence{0}};
}

operation::Bulk generateBulk()
{
  operation::Bulk::Param param;
  param.binary_path = "";
  param.origin_input_shapes = std::vector<onert::ir::Shape>{};
  param.origin_output_shapes = std::vector<onert::ir::Shape>{};

  return operation::Bulk{OperandIndexSequence{1, 2}, OperandIndexSequence{0}, param};
}

operation::Comparison generateComparison()
{
  operation::Comparison::Param param;
  param.comparison_type = operation::Comparison::ComparisonType::Equal;

  return operation::Comparison{OperandIndexSequence{1, 2}, OperandIndexSequence{0}, param};
}

operation::Concat generateConcat()
{
  operation::Concat::Param param;
  param.axis = 0;

  return operation::Concat{OperandIndexSequence{1, 2, 3}, OperandIndexSequence{0}, param};
}

operation::Conv2D generateConv2D()
{
  operation::Conv2D::Param param;
  param.activation = Activation::NONE;
  param.dilation = Dilation{};
  param.padding = Padding{};
  param.stride = Stride{};

  return operation::Conv2D{OperandIndexSequence{1, 2, 3}, OperandIndexSequence{0}, param};
}

operation::ConvertFp16ToFp32 generateConvertFp16ToFp32()
{
  return operation::ConvertFp16ToFp32{OperandIndexSequence{1}, OperandIndexSequence{0}};
}

operation::ConvertFp32ToFp16 generateConvertFp32ToFp16()
{
  return operation::ConvertFp32ToFp16{OperandIndexSequence{1}, OperandIndexSequence{0}};
}

operation::Custom generateCustom()
{
  return operation::Custom{OperandConstraint::createExact(1u), OperandIndexSequence{1},
                           OperandIndexSequence{0}, std::string("id"),
                           operation::Custom::Userdata{}};
}

operation::DepthToSpace generateDepthToSpace()
{
  operation::DepthToSpace::Param param;
  param.block_size = 1;

  return operation::DepthToSpace{OperandIndexSequence{1}, OperandIndexSequence{0}, param};
}

operation::DepthwiseConv2D generateDepthwiseConv2D()
{
  operation::DepthwiseConv2D::Param param;
  param.activation = Activation::NONE;
  param.dilation = Dilation{};
  param.multiplier = 1u;
  param.padding = Padding{};
  param.stride = Stride{};

  return operation::DepthwiseConv2D{OperandIndexSequence{1, 2, 3}, OperandIndexSequence{0}, param};
}

operation::DetectionPostProcess generateDetectionPostProcess()
{
  operation::DetectionPostProcess::Param param;

  return operation::DetectionPostProcess{OperandIndexSequence{1, 2, 3}, OperandIndexSequence{0},
                                         param};
}

operation::Einsum generateEinsum()
{
  operation::Einsum::Param param;
  param.equation = "";

  return operation::Einsum{OperandIndexSequence{1}, OperandIndexSequence{0}, param};
}

operation::ElementwiseActivation generateElementwiseActivation()
{
  operation::ElementwiseActivation::Param param;
  param.alpha = 0.f;
  param.beta = 0.f;
  param.op_type = operation::ElementwiseActivation::Type::ELU;

  return operation::ElementwiseActivation{OperandIndexSequence{1}, OperandIndexSequence{0}, param};
}

operation::ElementwiseBinary generateElementwiseBinary()
{
  operation::ElementwiseBinary::Param param;
  param.op_type = operation::ElementwiseBinary::ElementwiseBinaryType::FLOOR_DIV;

  return operation::ElementwiseBinary{OperandIndexSequence{1, 2}, OperandIndexSequence{0}, param};
}

operation::ElementwiseUnary generateElementwiseUnary()
{
  operation::ElementwiseUnary::Param param;
  param.op_type = operation::ElementwiseUnary::Type::ABS;

  return operation::ElementwiseUnary{OperandIndexSequence{1}, OperandIndexSequence{0}, param};
}

operation::EmbeddingLookup generateEmbeddingLookup()
{
  return operation::EmbeddingLookup{OperandIndexSequence{1, 2}, OperandIndexSequence{0}};
}

operation::ExpandDims generateExpandDims()
{
  return operation::ExpandDims{OperandIndexSequence{1, 2}, OperandIndexSequence{0}};
}

operation::Fill generateFill()
{
  return operation::Fill{OperandIndexSequence{1, 2}, OperandIndexSequence{0}};
}

operation::FullyConnected generateFullyConnected()
{
  operation::FullyConnected::Param param;
  param.activation = Activation::NONE;
  param.weights_format = FullyConnectedWeightsFormat::Default;

  return operation::FullyConnected{OperandIndexSequence{1, 2, 3}, OperandIndexSequence{0}, param};
}

operation::FusedBatchNorm generateFusedBatchNorm()
{
  operation::FusedBatchNorm::Param param;
  param.is_training = false;
  param.epsilon = 0.f;
  param.data_format = "";

  return operation::FusedBatchNorm{OperandIndexSequence{1, 2, 3, 4, 5}, OperandIndexSequence{0},
                                   param};
}

operation::Gather generateGather()
{
  operation::Gather::Param param;
  param.axis = 0;

  return operation::Gather{OperandIndexSequence{1, 2}, OperandIndexSequence{0}, param};
}

operation::HashtableLookup generateHashtableLookup()
{
  return operation::HashtableLookup{OperandIndexSequence{2, 3, 4}, OperandIndexSequence{0, 1}};
}

operation::If generateIf()
{
  operation::If::Param param;
  param.else_subg_index = 1;
  param.then_subg_index = 2;

  return operation::If{OperandIndexSequence{1, 2, 3}, OperandIndexSequence{0}, param};
}

operation::InstanceNorm generateInstanceNorm()
{
  operation::InstanceNorm::Param param;
  param.activation = Activation::NONE;
  param.epsilon = 0.f;

  return operation::InstanceNorm{OperandIndexSequence{1, 2, 3}, OperandIndexSequence{0}, param};
}

operation::L2Normalization generateL2Normalization()
{
  return operation::L2Normalization{OperandIndexSequence{1}, OperandIndexSequence{0}};
}

operation::LocalResponseNormalization generateLocalResponseNormalization()
{
  operation::LocalResponseNormalization::Param param;
  param.alpha = 0.f;
  param.beta = 0.f;
  param.bias = 0.f;
  param.radius = 1;

  return operation::LocalResponseNormalization{OperandIndexSequence{1}, OperandIndexSequence{0},
                                               param};
}

operation::LogSoftmax generateLogSoftmax()
{
  operation::LogSoftmax::Param param;
  param.axis = 0;
  param.beta = 0.f;

  return operation::LogSoftmax{OperandIndexSequence{1}, OperandIndexSequence{0}, param};
}

operation::LSTM generateLSTM()
{
  operation::LSTM::Param param;
  param.activation = Activation::NONE;
  param.cell_threshold = 1.f;
  param.projection_threshold = 1.f;
  param.time_major = true;

  return operation::LSTM{
    OperandIndexSequence{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
    OperandIndexSequence{0}, param};
}

operation::MatrixBandPart generateMatrixBandPart()
{
  return operation::MatrixBandPart{OperandIndexSequence{1, 2, 3}, OperandIndexSequence{0}};
}

operation::OneHot generateOneHot()
{
  operation::OneHot::Param param;
  param.axis = 0;

  return operation::OneHot{OperandIndexSequence{1, 2, 3, 4}, OperandIndexSequence{0}, param};
}

operation::Pack generatePack()
{
  operation::Pack::Param param;
  param.axis = 0;
  param.num = 1;

  return operation::Pack{OperandIndexSequence{1}, OperandIndexSequence{0}, param};
}

operation::Pad generatePad()
{
  return operation::Pad{OperandIndexSequence{1, 2, 3}, OperandIndexSequence{0}};
}

operation::Permute generatePermute()
{
  return operation::Permute{OperandIndex{1}, OperandIndex{0}, operation::Permute::Type::COPY};
}

operation::Pool2D generatePool2D()
{
  operation::Pool2D::Param param;
  param.activation = Activation::NONE;
  param.kh = 1;
  param.kw = 1;
  param.op_type = operation::Pool2D::PoolType::AVG;
  param.padding = Padding{};
  param.stride = Stride{};

  return operation::Pool2D{OperandIndexSequence{1}, OperandIndexSequence{0}, param};
}

operation::Pow generatePow()
{
  return operation::Pow{OperandIndexSequence{1, 2}, OperandIndexSequence{0}};
}

operation::PReLU generatePReLU()
{
  return operation::PReLU{OperandIndexSequence{1, 2}, OperandIndexSequence{0}};
}

operation::Range generateRange()
{
  return operation::Range{OperandIndexSequence{1, 2, 3}, OperandIndexSequence{0}};
}

operation::Rank generateRank()
{
  return operation::Rank{OperandIndexSequence{1}, OperandIndexSequence{0}};
}

operation::Reduce generateReduce()
{
  operation::Reduce::Param param;
  param.keep_dims = true;
  param.reduce_type = operation::Reduce::ReduceType::ALL;

  return operation::Reduce{OperandIndexSequence{1, 2}, OperandIndexSequence{0}, param};
}

operation::Reshape generateReshape()
{
  operation::Reshape::Param param;
  param.new_shape = std::vector<int32_t>{1};

  return operation::Reshape{OperandIndexSequence{1, 2}, OperandIndexSequence{0}, param};
}

operation::ResizeBilinear generateResizeBilinear()
{
  operation::ResizeBilinear::Param param;
  param.align_corners = true;
  param.half_pixel_centers = true;
  param.height_out = 1;
  param.width_out = 1;

  return operation::ResizeBilinear{OperandIndexSequence{1, 2}, OperandIndexSequence{0}, param};
}

operation::ResizeNearestNeighbor generateResizeNearestNeighbor()
{
  operation::ResizeNearestNeighbor::Param param;
  param.align_corners = true;
  param.height_out = 1;
  param.width_out = 1;

  return operation::ResizeNearestNeighbor{OperandIndexSequence{1, 2}, OperandIndexSequence{0},
                                          param};
}

operation::Reverse generateReverse()
{
  return operation::Reverse{OperandIndexSequence{1, 2}, OperandIndexSequence{0}};
}

operation::RNN generateRNN()
{
  operation::RNN::Param param;
  param.activation = Activation::NONE;

  return operation::RNN{OperandIndexSequence{1, 2, 3, 4, 5}, OperandIndexSequence{0}, param};
}

operation::Select generateSelect()
{
  return operation::Select{OperandIndexSequence{1, 2, 3}, OperandIndexSequence{0}};
}

operation::Shape generateShape()
{
  return operation::Shape{OperandIndexSequence{1}, OperandIndexSequence{0}};
}

operation::Slice generateSlice()
{
  return operation::Slice{OperandIndexSequence{1, 2, 3}, OperandIndexSequence{0}};
}

operation::Softmax generateSoftmax()
{
  operation::Softmax::Param param;
  param.beta = 0.1f;

  return operation::Softmax{OperandIndexSequence{1}, OperandIndexSequence{0}, param};
}

operation::SpaceToBatchND generateSpaceToBatchND()
{
  return operation::SpaceToBatchND{OperandIndexSequence{1, 2, 3}, OperandIndexSequence{0}};
}

operation::SpaceToDepth generateSpaceToDepth()
{
  operation::SpaceToDepth::Param param;
  param.block_size = 1;

  return operation::SpaceToDepth{OperandIndexSequence{1}, OperandIndexSequence{0}, param};
}

operation::Split generateSplit()
{
  operation::Split::Param param;
  param.num_splits = 1;

  return operation::Split{OperandIndexSequence{1, 2}, OperandIndexSequence{0}, param};
}

operation::SplitV generateSplitV()
{
  operation::SplitV::Param param;
  param.num_splits = 1;

  return operation::SplitV{OperandIndexSequence{1, 2, 3}, OperandIndexSequence{0}, param};
}

operation::SquaredDifference generateSquaredDifference()
{
  return operation::SquaredDifference{OperandIndexSequence{1, 2}, OperandIndexSequence{0}};
}

operation::Squeeze generateSqueeze()
{
  operation::Squeeze::Param param;
  param.dims[0] = 1;
  param.ndim = 1;

  return operation::Squeeze{OperandIndexSequence{1}, OperandIndexSequence{0}, param};
}

operation::StatelessRandomUniform generateStatelessRandomUniform()
{
  return operation::StatelessRandomUniform{OperandIndexSequence{1, 2}, OperandIndexSequence{0}};
}

operation::StridedSlice generateStridedSlice()
{
  operation::StridedSlice::Param param;
  param.begin_mask = 1;
  param.end_mask = 1;
  param.shrink_axis_mask = 1;

  return operation::StridedSlice{OperandIndexSequence{1, 2, 3, 4}, OperandIndexSequence{0}, param};
}

operation::Tile generateTile()
{
  return operation::Tile{OperandIndexSequence{1, 2}, OperandIndexSequence{0}};
}

operation::TopKV2 generateTopKV2()
{
  operation::TopKV2::Param param;
  param.k = 1;

  return operation::TopKV2{OperandIndexSequence{1}, OperandIndexSequence{0}, param};
}

operation::Transpose generateTranspose()
{
  return operation::Transpose{OperandIndexSequence{1, 2}, OperandIndexSequence{0}};
}

operation::TransposeConv generateTransposeConv()
{
  operation::TransposeConv::Param param;
  param.padding = Padding();
  param.stride = Stride();

  return operation::TransposeConv{OperandIndexSequence{1, 2, 3}, OperandIndexSequence{0}, param};
}

operation::Unpack generateUnpack()
{
  operation::Unpack::Param param;
  param.axis = 0;
  param.num = 1;

  return operation::Unpack{OperandIndexSequence{1}, OperandIndexSequence{0}, param};
}

operation::While generateWhile()
{
  operation::While::Param param;
  param.cond_subg_index = 1;
  param.body_subg_index = 2;

  return operation::While{OperandIndexSequence{1, 2, 3}, OperandIndexSequence{0}, param};
}

class MockOperationVisitor : public OperationVisitor
{
public:
  void invoke(Operation &op) { op.accept(*this); }

#define OP(InternalName) \
  virtual void visit(const operation::InternalName &) override { visit_flag = true; }
#include "ir/Operations.lst"
#undef OP

public:
  // TODO Replace this flag with using GMOCK if necessary
  bool visit_flag{false};
};

template <typename OperationType> auto generateUntrainableOperation(const OperationType &op)
{
  return std::make_unique<train::operation::UntrainableOperation<OperationType>>(op);
}

template <typename OperationType> void verifyOp(const OperationType &op)
{
  // auto untrainable = std::make_unique<train::operation::UntrainableOperation<OperationType>>(op);
  auto untrainable = generateUntrainableOperation(op);

  // Check clone
  auto clone = untrainable->clone();
  EXPECT_TRUE(clone != nullptr);
  EXPECT_EQ(clone->hasTrainableParameter(), untrainable->hasTrainableParameter());

  // Check downcast
  const auto derived =
    dynamic_cast<train::operation::UntrainableOperation<OperationType> *>(clone.get());
  EXPECT_TRUE(derived != nullptr);
  EXPECT_EQ(derived->opcode(), op.opcode());
  EXPECT_EQ(derived->getInputs(), op.getInputs());
  EXPECT_EQ(derived->getOutputs(), op.getOutputs());

  // Check visitor
  MockOperationVisitor visitor;

  visitor.visit_flag = false;
  visitor.invoke(const_cast<OperationType &>(op));
  EXPECT_TRUE(visitor.visit_flag);
}

TEST(UntrainableOperation, testAllOps)
{
  const auto addn = generateAddN();
  verifyOp(addn);

  const auto argminmax = generateArgMinMax();
  verifyOp(argminmax);

  const auto batch_matmul = generateBatchMatMul();
  verifyOp(batch_matmul);

  const auto batch_to_spacend = generateBatchToSpaceND();
  verifyOp(batch_to_spacend);

  const auto bcq_fc = generateBCQFullyConnected();
  verifyOp(bcq_fc);

  const auto bcq_gather = generateBCQGather();
  verifyOp(bcq_gather);

  const auto binary_arithmetic = generateBinaryArithmetic();
  verifyOp(binary_arithmetic);

  const auto broadcast = generateBroadcastTo();
  verifyOp(broadcast);

  const auto bulk = generateBulk();
  verifyOp(bulk);

  const auto comparison = generateComparison();
  verifyOp(comparison);

  const auto concat = generateConcat();
  verifyOp(concat);

  const auto conv2d = generateConv2D();
  verifyOp(conv2d);

  const auto fp16_to_fp32 = generateConvertFp16ToFp32();
  verifyOp(fp16_to_fp32);

  const auto fp32_to_fp16 = generateConvertFp32ToFp16();
  verifyOp(fp32_to_fp16);

  const auto custom = generateCustom();
  verifyOp(custom);

  const auto depth_to_space = generateDepthToSpace();
  verifyOp(depth_to_space);

  const auto depthwise_conv2d = generateDepthwiseConv2D();
  verifyOp(depthwise_conv2d);

  const auto detection = generateDetectionPostProcess();
  verifyOp(detection);

  const auto einsum = generateEinsum();
  verifyOp(einsum);

  const auto activation = generateElementwiseActivation();
  verifyOp(activation);

  const auto binary = generateElementwiseBinary();
  verifyOp(binary);

  const auto unary = generateElementwiseUnary();
  verifyOp(unary);

  const auto embed = generateEmbeddingLookup();
  verifyOp(embed);

  const auto expand_dims = generateExpandDims();
  verifyOp(expand_dims);

  const auto fill = generateFill();
  verifyOp(fill);

  const auto fc = generateFullyConnected();
  verifyOp(fc);

  const auto fused_batch_norm = generateFusedBatchNorm();
  verifyOp(fused_batch_norm);

  const auto gather = generateGather();
  verifyOp(gather);

  const auto hashtable = generateHashtableLookup();
  verifyOp(hashtable);

  const auto if_op = generateIf();
  verifyOp(if_op);

  const auto in_norm = generateInstanceNorm();
  verifyOp(in_norm);

  const auto l2_norm = generateL2Normalization();
  verifyOp(l2_norm);

  const auto local_norm = generateLocalResponseNormalization();
  verifyOp(local_norm);

  const auto log_softmax = generateLogSoftmax();
  verifyOp(log_softmax);

  const auto lstm = generateLSTM();
  verifyOp(lstm);

  const auto maxrix_band_part = generateMatrixBandPart();
  verifyOp(maxrix_band_part);

  const auto one_hot = generateOneHot();
  verifyOp(one_hot);

  const auto pack = generatePack();
  verifyOp(pack);

  const auto pad = generatePad();
  verifyOp(pad);

  const auto permute = generatePermute();
  verifyOp(permute);

  const auto pool2d = generatePool2D();
  verifyOp(pool2d);

  const auto pow = generatePow();
  verifyOp(pow);

  const auto prelu = generatePReLU();
  verifyOp(prelu);

  const auto range = generateRange();
  verifyOp(range);

  const auto rank = generateRank();
  verifyOp(rank);

  const auto reduce = generateReduce();
  verifyOp(reduce);

  const auto reshape = generateReshape();
  verifyOp(reshape);

  const auto resize_bilinear = generateResizeBilinear();
  verifyOp(resize_bilinear);

  const auto resize_nearest_neighbor = generateResizeNearestNeighbor();
  verifyOp(resize_nearest_neighbor);

  const auto reverse = generateReverse();
  verifyOp(reverse);

  const auto rnn = generateRNN();
  verifyOp(rnn);

  const auto select = generateSelect();
  verifyOp(select);

  const auto shape = generateShape();
  verifyOp(shape);

  const auto slice = generateSlice();
  verifyOp(slice);

  const auto softmax = generateSoftmax();
  verifyOp(softmax);

  const auto space_to_batchnd = generateSpaceToBatchND();
  verifyOp(space_to_batchnd);

  const auto space_to_depth = generateSpaceToDepth();
  verifyOp(space_to_depth);

  const auto split = generateSplit();
  verifyOp(split);

  const auto splitv = generateSplitV();
  verifyOp(splitv);

  const auto squared_diff = generateSquaredDifference();
  verifyOp(squared_diff);

  const auto squeeze = generateSqueeze();
  verifyOp(squeeze);

  const auto stateless_random_uniform = generateStatelessRandomUniform();
  verifyOp(stateless_random_uniform);

  const auto strided_slice = generateStridedSlice();
  verifyOp(strided_slice);

  const auto tile = generateTile();
  verifyOp(tile);

  const auto topkv2 = generateTopKV2();
  verifyOp(topkv2);

  const auto transpose = generateTranspose();
  verifyOp(transpose);

  const auto transpose_conv = generateTransposeConv();
  verifyOp(transpose_conv);

  const auto unpack = generateUnpack();
  verifyOp(unpack);

  const auto while_op = generateWhile();
  verifyOp(while_op);
}

class MockTrainableOperationVisitor : public train::TrainableOperationVisitor
{
public:
  void invoke(train::ITrainableOperation &op) { op.accept(*this); }

#define OP(InternalName) \
  virtual void visit(const train::operation::InternalName &) override {}
#include "ir/train/ITrainableOperation.h"
#undef OP
};

TEST(UntrainableOperation, neg_TrainableOperationVisitor)
{
  MockTrainableOperationVisitor visitor;

  {
    const auto addn = generateAddN();
    auto untrainable = generateUntrainableOperation(addn);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    auto argminmax = generateArgMinMax();
    auto untrainable = generateUntrainableOperation(argminmax);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto batch_matmul = generateBatchMatMul();
    auto untrainable = generateUntrainableOperation(batch_matmul);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto batch_to_spacend = generateBatchToSpaceND();
    auto untrainable = generateUntrainableOperation(batch_to_spacend);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto bcq_fc = generateBCQFullyConnected();
    auto untrainable = generateUntrainableOperation(bcq_fc);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto bcq_gather = generateBCQGather();
    auto untrainable = generateUntrainableOperation(bcq_gather);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto binary_arithmetic = generateBinaryArithmetic();
    auto untrainable = generateUntrainableOperation(binary_arithmetic);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto broadcast = generateBroadcastTo();
    auto untrainable = generateUntrainableOperation(broadcast);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto bulk = generateBulk();
    auto untrainable = generateUntrainableOperation(bulk);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto comparison = generateComparison();
    auto untrainable = generateUntrainableOperation(comparison);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto concat = generateConcat();
    auto untrainable = generateUntrainableOperation(concat);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto conv2d = generateConv2D();
    auto untrainable = generateUntrainableOperation(conv2d);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto fp16_to_fp32 = generateConvertFp16ToFp32();
    auto untrainable = generateUntrainableOperation(fp16_to_fp32);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto fp32_to_fp16 = generateConvertFp32ToFp16();
    auto untrainable = generateUntrainableOperation(fp32_to_fp16);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto custom = generateCustom();
    auto untrainable = generateUntrainableOperation(custom);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto depth_to_space = generateDepthToSpace();
    auto untrainable = generateUntrainableOperation(depth_to_space);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto depthwise_conv2d = generateDepthwiseConv2D();
    auto untrainable = generateUntrainableOperation(depthwise_conv2d);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto detection = generateDetectionPostProcess();
    auto untrainable = generateUntrainableOperation(detection);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto einsum = generateEinsum();
    auto untrainable = generateUntrainableOperation(einsum);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto activation = generateElementwiseActivation();
    auto untrainable = generateUntrainableOperation(activation);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto binary = generateElementwiseBinary();
    auto untrainable = generateUntrainableOperation(binary);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto unary = generateElementwiseUnary();
    auto untrainable = generateUntrainableOperation(unary);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto embed = generateEmbeddingLookup();
    auto untrainable = generateUntrainableOperation(embed);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto expand_dims = generateExpandDims();
    auto untrainable = generateUntrainableOperation(expand_dims);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto fill = generateFill();
    auto untrainable = generateUntrainableOperation(fill);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto fc = generateFullyConnected();
    auto untrainable = generateUntrainableOperation(fc);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto fused_batch_norm = generateFusedBatchNorm();
    auto untrainable = generateUntrainableOperation(fused_batch_norm);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto gather = generateGather();
    auto untrainable = generateUntrainableOperation(gather);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto hashtable = generateHashtableLookup();
    auto untrainable = generateUntrainableOperation(hashtable);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto if_op = generateIf();
    auto untrainable = generateUntrainableOperation(if_op);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto in_norm = generateInstanceNorm();
    auto untrainable = generateUntrainableOperation(in_norm);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto l2_norm = generateL2Normalization();
    auto untrainable = generateUntrainableOperation(l2_norm);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto local_norm = generateLocalResponseNormalization();
    auto untrainable = generateUntrainableOperation(local_norm);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto log_softmax = generateLogSoftmax();
    auto untrainable = generateUntrainableOperation(log_softmax);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto lstm = generateLSTM();
    auto untrainable = generateUntrainableOperation(lstm);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto matrix_band_part = generateMatrixBandPart();
    auto untrainable = generateUntrainableOperation(matrix_band_part);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto one_hot = generateOneHot();
    auto untrainable = generateUntrainableOperation(one_hot);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto pack = generatePack();
    auto untrainable = generateUntrainableOperation(pack);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto pad = generatePad();
    auto untrainable = generateUntrainableOperation(pad);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto permute = generatePermute();
    auto untrainable = generateUntrainableOperation(permute);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto pool2d = generatePool2D();
    auto untrainable = generateUntrainableOperation(pool2d);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto pow = generatePow();
    auto untrainable = generateUntrainableOperation(pow);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto prelu = generatePReLU();
    auto untrainable = generateUntrainableOperation(prelu);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto range = generateRange();
    auto untrainable = generateUntrainableOperation(range);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto rank = generateRank();
    auto untrainable = generateUntrainableOperation(rank);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto reduce = generateReduce();
    auto untrainable = generateUntrainableOperation(reduce);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto reshape = generateReshape();
    auto untrainable = generateUntrainableOperation(reshape);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto resize_bilinear = generateResizeBilinear();
    auto untrainable = generateUntrainableOperation(resize_bilinear);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto resize_nearest_neighbor = generateResizeNearestNeighbor();
    auto untrainable = generateUntrainableOperation(resize_nearest_neighbor);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto reverse = generateReverse();
    auto untrainable = generateUntrainableOperation(reverse);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto rnn = generateRNN();
    auto untrainable = generateUntrainableOperation(rnn);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto select = generateSelect();
    auto untrainable = generateUntrainableOperation(select);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto shape = generateShape();
    auto untrainable = generateUntrainableOperation(shape);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto slice = generateSlice();
    auto untrainable = generateUntrainableOperation(slice);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto softmax = generateSoftmax();
    auto untrainable = generateUntrainableOperation(softmax);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto space_to_batchnd = generateSpaceToBatchND();
    auto untrainable = generateUntrainableOperation(space_to_batchnd);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto space_to_depth = generateSpaceToDepth();
    auto untrainable = generateUntrainableOperation(space_to_depth);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto split = generateSplit();
    auto untrainable = generateUntrainableOperation(split);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto splitv = generateSplitV();
    auto untrainable = generateUntrainableOperation(splitv);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto squared_diff = generateSquaredDifference();
    auto untrainable = generateUntrainableOperation(squared_diff);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto squeeze = generateSqueeze();
    auto untrainable = generateUntrainableOperation(squeeze);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto stateless_random_uniform = generateStatelessRandomUniform();
    auto untrainable = generateUntrainableOperation(stateless_random_uniform);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto strided_slice = generateStridedSlice();
    auto untrainable = generateUntrainableOperation(strided_slice);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto tile = generateTile();
    auto untrainable = generateUntrainableOperation(tile);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto topkv2 = generateTopKV2();
    auto untrainable = generateUntrainableOperation(topkv2);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto transpose = generateTranspose();
    auto untrainable = generateUntrainableOperation(transpose);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto transpose_conv = generateTransposeConv();
    auto untrainable = generateUntrainableOperation(transpose_conv);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto unpack = generateUnpack();
    auto untrainable = generateUntrainableOperation(unpack);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }

  {
    const auto while_op = generateWhile();
    auto untrainable = generateUntrainableOperation(while_op);
    EXPECT_ANY_THROW(visitor.invoke(*untrainable));
  }
}
