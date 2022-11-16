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

#include "luci/Pass/UnrollUnidirectionalSequenceLSTMPass.h"

#include "helpers/NodeFiller.h"
#include "helpers/TypeMapper.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <string>
#include <vector>

/**
 *  BEFORE
 *        [CircleNode]
 *              |
 *   [UnidirectionalSequenceLSTM]
 *              |
 *        [CircleNode]
 *
 *  AFTER
 *
 *        [CircleNode]
 *              |
 *      [CircleTranspose]
 *              |
 *        [CircleUnpack]
 *              |
 *       [CircleUnpackOut]
 *              |
 *      (Unrolled sub network)
 *              |
 *        [CirclePack]
 *              |                        |
 *      [CircleTranspose]     [UnidirectionalSequenceLSTM]
 *              |                        |
 *        [CircleNode]
 *
 *  NOTE for timesteps = 1,
 *       first [CircleTranspose] is not added and
 *       last [CirclePack] + [CircleTranspose] is replaced with [CircleReshape]
 *
 *  First unrolled sub network is as follows
 *    - [] and 'Circle' are omitted
 *    - all FC has one or two Const for Weight/Bias
 *
 *            (input)
 *              |
 *              FC
 *              |
 *            Split
 *    +---------+----------+----------+
 *    |         |          |          |
 *    |      Logistic   Logistic     Tanh
 *    |  Const  |          |          |
 *    |    |    |          |          |
 *    |    +-- Mul         +-- Mul ---+
 *    |         |               |
 *    |         +---- Add ------+
 *    |                |
 *    |           +----+----+
 *    |           |         |
 *  Logistic     Tanh       |
 *    |           |         |
 *    +-- Mul ----+         |
 *         |                |
 *       (output)          (A)
 *
 *  and following unrolled sub networks are;
 *
 *   (prev-output) (input)
 *        |          |
 *        FC         FC
 *        |          |
 *        +--- Add --+
 *   Const      |
 *     |        |
 *     +------ Add
 *              |
 *            Split
 *              |
 *    +---------+----------+----------+
 * SplitOut SplitOut   SplitOut   SplitOut
 *    |         |          |          |
 *    |      Logistic   Logistic     Tanh
 *    |  (A')   |          |          |
 *    |   |     |          |          |
 *    |   +--- Mul         +-- Mul ---+
 *    |         |               |
 *    |         +---- Add ------+
 *    |                |
 *    |           +----+----+
 *    |           |         |
 *  Logistic     Tanh       |
 *    |           |         |
 *    +-- Mul ----+         |
 *         |                |
 *      (output)          (next)
 *
 * where (A) and (A') are connected
 *
 */

namespace
{

bool unroll_lstm(luci::CircleUnidirectionalSequenceLSTM *lstm)
{
  // TODO implement
  (void)lstm;
  return false;
}

} // namespace

namespace luci
{

bool UnrollUnidirectionalSequenceLSTMPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto lstm = dynamic_cast<luci::CircleUnidirectionalSequenceLSTM *>(node))
    {
      if (unroll_lstm(lstm))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
