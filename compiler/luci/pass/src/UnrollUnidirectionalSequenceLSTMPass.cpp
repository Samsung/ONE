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

struct UnrollLSTM
{
  luci::CircleUnidirectionalSequenceLSTM *_lstm{nullptr};
  loco::Graph::NodeContext *_nctx{nullptr};
  std::string _name;
  uint32_t _batch{0};
  uint32_t _timesteps{0};
  uint32_t _units{0}; // output space dim
};

bool unroll_lstm(luci::CircleUnidirectionalSequenceLSTM *lstm)
{
  // NOTE shape of input of lstm is interpreted as [batch, timesteps, feature]
  //      shape of output of lstm is interpreted as [batch, timesteps, units]
  // TODO add more conditions to check LSTM
  assert(lstm->rank() == 3); // use assert to findout when this happens
  if (lstm->rank() != 3)
    return false;
  if (!(lstm->dim(0).known() and lstm->dim(1).known() and lstm->dim(2).known()))
    return false;

  UnrollLSTM ulstm;
  ulstm._lstm = lstm;
  ulstm._nctx = lstm->graph()->nodes();
  ulstm._name = lstm->name();
  ulstm._batch = lstm->dim(0).value();
  ulstm._timesteps = lstm->dim(1).value();
  ulstm._units = lstm->dim(2).value(); // output space dim

  luci::CircleNode *input = loco::must_cast<luci::CircleNode *>(lstm->input());
  assert(input->rank() == 3); // use assert to findout when this happens
  if (input->rank() != 3)
    return false;
  assert(input->dim(0).value() == ulstm._batch);
  assert(input->dim(1).value() == ulstm._timesteps);

  // TODO implement
  (void)ulstm;

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
