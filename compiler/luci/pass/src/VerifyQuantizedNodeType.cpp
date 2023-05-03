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

#include "VerifyQuantizedNodeType.h"

#include <cmath>
#include <memory>

// This macro is undef at the end of the file
#define RETURN_FALSE_UNLESS(ARG) \
  if (not(ARG))                  \
  {                              \
    return false;                \
  }

namespace luci
{

std::shared_ptr<VerifyQuantizedNodeType> VerifyQuantizedNodeType::create(loco::DataType dtype)
{
  if (dtype == loco::DataType::U8)
    return std::make_shared<VerifyQuantizedNodeU8Type>();
  else if (dtype == loco::DataType::S16)
    return std::make_shared<VerifyQuantizedNodeS16Type>();
  else
    throw std::domain_error("Not supported Quantized type");
}

} // namespace luci

namespace luci
{

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleAdd *node)
{
  // Allow add of indices
  if (group_has_type(node, loco::DataType::S32) or group_has_type(node, loco::DataType::S64))
    return true;

  return group_has_type(node, Qtype);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleArgMax *node)
{
  RETURN_FALSE_UNLESS(has_type(node, node->output_type()))
  RETURN_FALSE_UNLESS(has_type(node->input(), Qtype))
  RETURN_FALSE_UNLESS(has_type(node->dimension(), loco::DataType::S32) ||
                      has_type(node->dimension(), loco::DataType::S64))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleAveragePool2D *node)
{
  return group_has_type(node, Qtype);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleBatchToSpaceND *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))
  RETURN_FALSE_UNLESS(has_type(node->input(), Qtype))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleCast *node)
{
  auto *input = loco::must_cast<luci::CircleNode *>(node->x());
  bool input_quantized = input->quantparam() != nullptr;
  if (input_quantized)
  {
    RETURN_FALSE_UNLESS(has_type(input, node->in_data_type()))
    RETURN_FALSE_UNLESS(has_type(input, Qtype))
  }

  bool node_quantized = node->quantparam() != nullptr;
  if (node_quantized)
  {
    RETURN_FALSE_UNLESS(has_type(node, node->out_data_type()))
    RETURN_FALSE_UNLESS(has_type(node, Qtype))
  }
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleConv2D *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))
  RETURN_FALSE_UNLESS(has_type(node->input(), Qtype))
  RETURN_FALSE_UNLESS(has_type(node->filter(), Qtype))
  RETURN_FALSE_UNLESS(has_type(node->bias(), Btype))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleConcatenation *node)
{
  // Allow concatenation of indices
  if (group_has_type(node, loco::DataType::S32) or group_has_type(node, loco::DataType::S64))
    return true;

  return group_has_type(node, Qtype);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleDepthToSpace *node)
{
  return group_has_type(node, Qtype);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleDepthwiseConv2D *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))
  RETURN_FALSE_UNLESS(has_type(node->input(), Qtype))
  RETURN_FALSE_UNLESS(has_type(node->filter(), Qtype))
  RETURN_FALSE_UNLESS(has_type(node->bias(), Btype))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleDiv *node)
{
  return group_has_type(node, Qtype);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleElu *node)
{
  return group_has_type(node, Qtype);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleFloor *node)
{
  RETURN_FALSE_UNLESS(group_has_type(node, Qtype));

  // This checks the value of scale is an integer
  RETURN_FALSE_UNLESS(node->quantparam());
  RETURN_FALSE_UNLESS(std::roundf(node->quantparam()->scale[0]) == node->quantparam()->scale[0]);
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleFloorDiv *node)
{
  RETURN_FALSE_UNLESS(group_has_type(node, Qtype));

  // This checks the value of scale is an integer
  RETURN_FALSE_UNLESS(node->quantparam());
  RETURN_FALSE_UNLESS(std::roundf(node->quantparam()->scale[0]) == node->quantparam()->scale[0]);
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleFullyConnected *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))
  RETURN_FALSE_UNLESS(has_type(node->input(), Qtype))
  RETURN_FALSE_UNLESS(has_type(node->weights(), Qtype))
  luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(node->bias());
  if (bias != nullptr)
    RETURN_FALSE_UNLESS(has_type(bias, Btype))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleGelu *node)
{
  return group_has_type(node, Qtype);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleGreater *node)
{
  RETURN_FALSE_UNLESS(has_type(node, loco::DataType::BOOL))
  RETURN_FALSE_UNLESS(has_type(node->x(), Qtype))
  RETURN_FALSE_UNLESS(has_type(node->y(), Qtype))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleGreaterEqual *node)
{
  RETURN_FALSE_UNLESS(has_type(node, loco::DataType::BOOL))
  RETURN_FALSE_UNLESS(has_type(node->x(), Qtype))
  RETURN_FALSE_UNLESS(has_type(node->y(), Qtype))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleInstanceNorm *node)
{
  return group_has_type(node, Qtype);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(
  const luci::CircleLocalResponseNormalization *node)
{
  return group_has_type(node, Qtype);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleLogicalOr *node)
{
  return group_has_type(node, loco::DataType::BOOL);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleMaxPool2D *node)
{
  return group_has_type(node, Qtype);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleMean *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))
  RETURN_FALSE_UNLESS(has_type(node->input(), Qtype))
  RETURN_FALSE_UNLESS(has_type(node->reduction_indices(), loco::DataType::S32))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleMirrorPad *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))
  RETURN_FALSE_UNLESS(has_type(node->input(), Qtype))
  RETURN_FALSE_UNLESS(has_type(node->paddings(), loco::DataType::S32))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleMul *node)
{
  // Allow mul of indices
  if (group_has_type(node, loco::DataType::S32) or group_has_type(node, loco::DataType::S64))
    return true;

  return group_has_type(node, Qtype);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleNotEqual *node)
{
  RETURN_FALSE_UNLESS(has_type(node, loco::DataType::BOOL))
  RETURN_FALSE_UNLESS(has_type(node->x(), Qtype))
  RETURN_FALSE_UNLESS(has_type(node->y(), Qtype))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleOneHot *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype));
  RETURN_FALSE_UNLESS(has_type(node->indices(), loco::DataType::S32) ||
                      has_type(node->indices(), loco::DataType::S64));
  RETURN_FALSE_UNLESS(has_type(node->depth(), loco::DataType::S32));
  RETURN_FALSE_UNLESS(has_type(node->on_value(), Qtype));
  RETURN_FALSE_UNLESS(has_type(node->off_value(), Qtype));
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CirclePack *node)
{
  return group_has_type(node, Qtype);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CirclePad *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))
  RETURN_FALSE_UNLESS(has_type(node->input(), Qtype))
  RETURN_FALSE_UNLESS(has_type(node->paddings(), loco::DataType::S32))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CirclePadV2 *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))
  RETURN_FALSE_UNLESS(has_type(node->input(), Qtype))
  RETURN_FALSE_UNLESS(has_type(node->paddings(), loco::DataType::S32))
  RETURN_FALSE_UNLESS(has_type(node->constant_values(), Qtype))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CirclePRelu *node)
{
  return group_has_type(node, Qtype);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CirclePow *node)
{
  return group_has_type(node, Qtype);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleReduceMax *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))
  RETURN_FALSE_UNLESS(has_type(node->input(), Qtype))
  RETURN_FALSE_UNLESS(has_type(node->reduction_indices(), loco::DataType::S32))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleRelu *node)
{
  return group_has_type(node, Qtype);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleReshape *node)
{
  if (node->quantparam())
  {
    RETURN_FALSE_UNLESS(has_type(node, Qtype))
    RETURN_FALSE_UNLESS(has_type(node->tensor(), Qtype))
  }
  else
  {
    RETURN_FALSE_UNLESS(has_type(node->tensor(), node->dtype()))
  }
  luci::CircleConst *shape = dynamic_cast<luci::CircleConst *>(node->shape());
  if (shape != nullptr)
    RETURN_FALSE_UNLESS(has_type(shape, loco::DataType::S32))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleResizeBilinear *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))
  RETURN_FALSE_UNLESS(has_type(node->input(), Qtype))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleResizeNearestNeighbor *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))
  RETURN_FALSE_UNLESS(has_type(node->input(), Qtype))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleRsqrt *node)
{
  return group_has_type(node, Qtype);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleSlice *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))
  RETURN_FALSE_UNLESS(has_type(node->input(), Qtype))
  RETURN_FALSE_UNLESS(has_type(node->begin(), loco::DataType::S32) ||
                      has_type(node->begin(), loco::DataType::S64))
  RETURN_FALSE_UNLESS(has_type(node->size(), loco::DataType::S32) ||
                      has_type(node->size(), loco::DataType::S64))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleSpaceToBatchND *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))
  RETURN_FALSE_UNLESS(has_type(node->input(), Qtype))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleSpaceToDepth *node)
{
  return group_has_type(node, Qtype);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleSplit *node)
{
  // node's output is the input of CircleSplitOut, thus not quantized
  RETURN_FALSE_UNLESS(has_type(node->input(), Qtype))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleSplitOut *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))

  // SplitOut has the same qparam with the input of Split
  auto split = loco::must_cast<luci::CircleSplit *>(node->input());
  auto input = loco::must_cast<luci::CircleNode *>(split->input());
  RETURN_FALSE_UNLESS(node->quantparam());
  RETURN_FALSE_UNLESS(node->quantparam()->scale[0] == input->quantparam()->scale[0]);
  RETURN_FALSE_UNLESS(node->quantparam()->zerop[0] == input->quantparam()->zerop[0]);
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleSplitV *node)
{
  // node's output is the input of CircleSplitVOut, thus not quantized
  RETURN_FALSE_UNLESS(has_type(node->input(), Qtype))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleSplitVOut *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))

  // SplitVOut has the same qparam with the input of SplitV
  auto splitv = loco::must_cast<luci::CircleSplitV *>(node->input());
  auto input = loco::must_cast<luci::CircleNode *>(splitv->input());
  RETURN_FALSE_UNLESS(node->quantparam());
  RETURN_FALSE_UNLESS(node->quantparam()->scale[0] == input->quantparam()->scale[0]);
  RETURN_FALSE_UNLESS(node->quantparam()->zerop[0] == input->quantparam()->zerop[0]);
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleSqrt *node)
{
  return group_has_type(node, Qtype);
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleStridedSlice *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))
  RETURN_FALSE_UNLESS(has_type(node->input(), Qtype))

  auto input = loco::must_cast<luci::CircleNode *>(node->input());
  RETURN_FALSE_UNLESS(node->quantparam());
  RETURN_FALSE_UNLESS(node->quantparam()->scale[0] == input->quantparam()->scale[0]);
  RETURN_FALSE_UNLESS(node->quantparam()->zerop[0] == input->quantparam()->zerop[0]);
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleSum *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))
  RETURN_FALSE_UNLESS(has_type(node->input(), Qtype))
  RETURN_FALSE_UNLESS(has_type(node->reduction_indices(), loco::DataType::S32))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleTranspose *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))
  RETURN_FALSE_UNLESS(has_type(node->a(), Qtype))
  RETURN_FALSE_UNLESS(has_type(node->perm(), loco::DataType::S32))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleTransposeConv *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))
  RETURN_FALSE_UNLESS(has_type(node->outBackprop(), Qtype))
  RETURN_FALSE_UNLESS(has_type(node->filter(), Qtype))
  luci::CircleConst *bias = dynamic_cast<luci::CircleConst *>(node->bias());
  if (bias != nullptr)
    RETURN_FALSE_UNLESS(has_type(bias, Btype))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleUnpack *node)
{
  // node's output is the input of CircleUnpackOut, thus not quantized
  RETURN_FALSE_UNLESS(has_type(node->value(), Qtype))
  return true;
}

template <loco::DataType Qtype, loco::DataType Btype>
bool VerifyQuantizedNodeTypeBase<Qtype, Btype>::visit(const luci::CircleUnpackOut *node)
{
  RETURN_FALSE_UNLESS(has_type(node, Qtype))

  // UnpackOut has the same qparam with the input of Unpack
  auto Unpack = loco::must_cast<luci::CircleUnpack *>(node->input());
  auto input = loco::must_cast<luci::CircleNode *>(Unpack->value());
  RETURN_FALSE_UNLESS(node->quantparam() && input->quantparam());
  RETURN_FALSE_UNLESS(node->quantparam()->scale[0] == input->quantparam()->scale[0]);
  RETURN_FALSE_UNLESS(node->quantparam()->zerop[0] == input->quantparam()->zerop[0]);
  return true;
}

} // namespace luci

namespace luci
{

bool VerifyQuantizedNodeU8Type::visit(const luci::CircleTanh *node)
{
  RETURN_FALSE_UNLESS(group_has_type(node, loco::DataType::U8));

  RETURN_FALSE_UNLESS(node->quantparam());
  RETURN_FALSE_UNLESS(node->quantparam()->scale[0] == 2.0f / 256.0f);
  RETURN_FALSE_UNLESS(node->quantparam()->zerop[0] == 128);
  return true;
}

bool VerifyQuantizedNodeU8Type::visit(const luci::CircleLogistic *node)
{
  RETURN_FALSE_UNLESS(group_has_type(node, loco::DataType::U8));

  RETURN_FALSE_UNLESS(node->quantparam());
  RETURN_FALSE_UNLESS(node->quantparam()->scale[0] == 1.0f / 256.0f);
  RETURN_FALSE_UNLESS(node->quantparam()->zerop[0] == 0);
  return true;
}

bool VerifyQuantizedNodeU8Type::visit(const luci::CircleSoftmax *node)
{
  RETURN_FALSE_UNLESS(group_has_type(node, loco::DataType::U8));

  RETURN_FALSE_UNLESS(node->quantparam());
  RETURN_FALSE_UNLESS(node->quantparam()->scale[0] == 1.0f / 255.0f);
  RETURN_FALSE_UNLESS(node->quantparam()->zerop[0] == 0);
  return true;
}

} // namespace luci

namespace luci
{

bool VerifyQuantizedNodeS16Type::visit(const luci::CircleTanh *node)
{
  RETURN_FALSE_UNLESS(group_has_type(node, loco::DataType::S16));

  RETURN_FALSE_UNLESS(node->quantparam());
  RETURN_FALSE_UNLESS(node->quantparam()->scale[0] == 1.0f / 32768.0f);
  RETURN_FALSE_UNLESS(node->quantparam()->zerop[0] == 0);
  return true;
}

bool VerifyQuantizedNodeS16Type::visit(const luci::CircleLogistic *node)
{
  RETURN_FALSE_UNLESS(group_has_type(node, loco::DataType::S16));

  RETURN_FALSE_UNLESS(node->quantparam());
  RETURN_FALSE_UNLESS(node->quantparam()->scale[0] == 1.0f / 32768.0f);
  RETURN_FALSE_UNLESS(node->quantparam()->zerop[0] == 0);
  return true;
}

bool VerifyQuantizedNodeS16Type::visit(const luci::CircleSoftmax *node)
{
  RETURN_FALSE_UNLESS(group_has_type(node, loco::DataType::S16));

  RETURN_FALSE_UNLESS(node->quantparam());
  RETURN_FALSE_UNLESS(node->quantparam()->scale[0] == 1.0f / 32767.0f);
  RETURN_FALSE_UNLESS(node->quantparam()->zerop[0] == 0);
  return true;
}

} // namespace luci

#undef RETURN_FALSE_UNLESS
