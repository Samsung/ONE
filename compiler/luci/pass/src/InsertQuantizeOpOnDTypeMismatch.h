/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_INSERT_QUANTIZE_OP_ON_DTYPE_MISMATCH_H__
#define __LUCI_INSERT_QUANTIZE_OP_ON_DTYPE_MISMATCH_H__

#include <luci/IR/CircleNodeVisitor.h>

namespace luci
{

struct InsertQuantizeOpOnDTypeMismatch final : public luci::CircleNodeMutableVisitor<void>
{
  InsertQuantizeOpOnDTypeMismatch() = default;

private:
  void visit(luci::CircleNode *) {}

  void visit(luci::CircleFullyConnected *node);
  void visit(luci::CircleMul *node);
  void visit(luci::CircleBatchMatMul *node);

  // TODO Support more operators
};

} // namespace luci

#endif // __LUCI_INSERT_QUANTIZE_OP_ON_DTYPE_MISMATCH_H__
