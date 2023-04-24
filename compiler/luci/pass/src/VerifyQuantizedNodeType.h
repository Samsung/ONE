/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_VERIFY_QUANTIZED_NODE_TYPE_H__
#define __LUCI_VERIFY_QUANTIZED_NODE_TYPE_H__

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>

namespace luci
{

/**
 * @brief Verify the data type of quantized node
 * @details
 *
 * Targets to verify
 * - node's output (i.e., node itself)
 * - node's inputs
 */
class VerifyQuantizedNodeType
{
public:
  static std::shared_ptr<VerifyQuantizedNodeType> create(loco::DataType dtype);

public:
  virtual bool verify(luci::CircleNode *node) = 0;
};

/**
 * @brief Verify using quantization type of a node and bias
 *
 * @tparam Qtype Quantization type for a node (e.g. Q8, Q16, ...)
 * @tparam Btype Bias quantization type (e.g. For Q8, S32 is used)
 */
template <loco::DataType Qtype, loco::DataType Btype>
class VerifyQuantizedNodeTypeBase : public luci::CircleNodeVisitor<bool>,
                                    public VerifyQuantizedNodeType
{
public:
  bool verify(luci::CircleNode *node) { return node->accept(this); }

protected:
  bool has_type(const loco::Node *node, loco::DataType dtype)
  {
    auto circle_node = loco::must_cast<const luci::CircleNode *>(node);
    return circle_node->dtype() == dtype;
  }

  // Check whether a node and all of its inputs have dtype or not
  bool group_has_type(const loco::Node *node, loco::DataType dtype)
  {
    if (!has_type(node, dtype))
      return false;

    for (uint32_t i = 0; i < node->arity(); ++i)
      if (!has_type(node->arg(i), dtype))
        return false;

    return true;
  }

private:
  bool visit(const luci::CircleAdd *node);
  bool visit(const luci::CircleArgMax *node);
  bool visit(const luci::CircleAveragePool2D *node);
  bool visit(const luci::CircleBatchToSpaceND *node);
  bool visit(const luci::CircleCast *node);
  bool visit(const luci::CircleConv2D *node);
  bool visit(const luci::CircleConcatenation *node);
  bool visit(const luci::CircleDepthToSpace *node);
  bool visit(const luci::CircleDepthwiseConv2D *node);
  bool visit(const luci::CircleDiv *node);
  bool visit(const luci::CircleElu *node);
  bool visit(const luci::CircleFloor *node);
  bool visit(const luci::CircleFloorDiv *node);
  bool visit(const luci::CircleFullyConnected *node);
  bool visit(const luci::CircleGelu *node);
  bool visit(const luci::CircleGreater *node);
  bool visit(const luci::CircleGreaterEqual *node);
  bool visit(const luci::CircleInstanceNorm *node);
  bool visit(const luci::CircleLocalResponseNormalization *node);
  bool visit(const luci::CircleLogicalOr *node);
  bool visit(const luci::CircleMaxPool2D *node);
  bool visit(const luci::CircleMean *node);
  bool visit(const luci::CircleMirrorPad *node);
  bool visit(const luci::CircleMul *node);
  bool visit(const luci::CircleNotEqual *node);
  bool visit(const luci::CircleOneHot *node);
  bool visit(const luci::CirclePack *node);
  bool visit(const luci::CirclePad *node);
  bool visit(const luci::CirclePadV2 *node);
  bool visit(const luci::CirclePRelu *node);
  bool visit(const luci::CirclePow *node);
  bool visit(const luci::CircleReduceMax *node);
  bool visit(const luci::CircleRelu *node);
  bool visit(const luci::CircleReshape *node);
  bool visit(const luci::CircleResizeBilinear *node);
  bool visit(const luci::CircleResizeNearestNeighbor *node);
  bool visit(const luci::CircleRsqrt *node);
  bool visit(const luci::CircleSlice *node);
  bool visit(const luci::CircleSpaceToBatchND *node);
  bool visit(const luci::CircleSpaceToDepth *node);
  bool visit(const luci::CircleSplit *node);
  bool visit(const luci::CircleSplitOut *node);
  bool visit(const luci::CircleSplitV *node);
  bool visit(const luci::CircleSplitVOut *node);
  bool visit(const luci::CircleSqrt *node);
  bool visit(const luci::CircleStridedSlice *node);
  bool visit(const luci::CircleSum *node);
  bool visit(const luci::CircleTranspose *node);
  bool visit(const luci::CircleTransposeConv *node);
  bool visit(const luci::CircleUnpack *node);
  bool visit(const luci::CircleUnpackOut *node);

  // NOTE below nodes has differnent implementation for Qtype/Btype and
  //      implementations exist in VerifyQuantizedNodeU8Type, VerifyQuantizedNodeS16Type
  // bool visit(const luci::CircleLogistic *node);
  // bool visit(const luci::CircleSoftmax *node);
  // bool visit(const luci::CircleTanh *node);

  // TODO: Implement more Ops

  bool visit(const luci::CircleNode *) { return true; }
};

class VerifyQuantizedNodeU8Type
  : public VerifyQuantizedNodeTypeBase<loco::DataType::U8, loco::DataType::S32>
{
private:
  bool visit(const luci::CircleLogistic *node);
  bool visit(const luci::CircleSoftmax *node);
  bool visit(const luci::CircleTanh *node);
};

class VerifyQuantizedNodeS16Type
  : public VerifyQuantizedNodeTypeBase<loco::DataType::S16, loco::DataType::S64>
{
private:
  bool visit(const luci::CircleLogistic *node);
  bool visit(const luci::CircleSoftmax *node);
  bool visit(const luci::CircleTanh *node);
};

} // namespace luci

#endif // __LUCI_VERIFY_QUANTIZED_NODE_TYPE_H__
