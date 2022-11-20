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
  luci::CircleConst *transpose_perm(void);
  luci::CircleTranspose *first_transpose(luci::CircleNode *input);
  std::vector<luci::CircleUnpackOut *> input_unpacks(luci::CircleNode *input);
  luci::CircleConst *merged_weights(luci::CircleConst *iw, luci::CircleConst *fw,
                                    luci::CircleConst *cw, luci::CircleConst *ow);
  luci::CircleFullyConnected *create_input_matmul(luci::CircleNode *input);
  std::vector<luci::CircleSplitOut *> matmul_splits(luci::CircleNode *input, uint32_t step);

  luci::CircleUnidirectionalSequenceLSTM *_lstm{nullptr};
  loco::Graph::NodeContext *_nctx{nullptr};
  std::string _name;
  uint32_t _batch{0};
  uint32_t _timesteps{0};
  uint32_t _units{0}; // output space dim
};

luci::CircleConst *UnrollLSTM::transpose_perm(void)
{
  auto perm = _nctx->create<luci::CircleConst>();
  perm->dtype(loco::DataType::S32);
  perm->rank(1);
  perm->dim(0) = 3;
  perm->size<loco::DataType::S32>(3);
  perm->at<loco::DataType::S32>(0) = 1;
  perm->at<loco::DataType::S32>(1) = 0;
  perm->at<loco::DataType::S32>(2) = 2;
  perm->shape_status(luci::ShapeStatus::VALID);

  return perm;
}

luci::CircleTranspose *UnrollLSTM::first_transpose(luci::CircleNode *input)
{
  auto perm = transpose_perm();
  perm->name(_name + "_perm1");
  luci::add_origin(perm, luci::get_origin(_lstm));

  auto transpose = _nctx->create<luci::CircleTranspose>();
  transpose->a(input);
  transpose->perm(perm);
  transpose->name(_name + "_trans1");
  luci::add_origin(transpose, luci::get_origin(_lstm));

  return transpose;
}

std::vector<luci::CircleUnpackOut *> UnrollLSTM::input_unpacks(luci::CircleNode *input)
{
  assert(input != nullptr);

  // NOTE unpack input can be LSTM or Transpose
  auto unpack = _nctx->create<luci::CircleUnpack>();
  unpack->num(_timesteps);
  unpack->axis(0);
  unpack->value(input);
  unpack->name(_name + "_unpack");
  luci::add_origin(unpack, luci::get_origin(_lstm));

  std::vector<luci::CircleUnpackOut *> outs;
  for (uint32_t idx = 0; idx < _timesteps; ++idx)
  {
    auto unpackout = _nctx->create<luci::CircleUnpackOut>();
    unpackout->input(unpack);
    unpackout->index(idx);
    unpackout->name(_name + "_unpackout_" + std::to_string(idx));
    luci::add_origin(unpackout, luci::get_origin(_lstm));
    outs.push_back(unpackout);
  }

  return outs;
}

luci::CircleConst *UnrollLSTM::merged_weights(luci::CircleConst *iw, luci::CircleConst *fw,
                                              luci::CircleConst *cw, luci::CircleConst *ow)
{
  assert(iw != nullptr);
  assert(fw != nullptr);
  assert(cw != nullptr);
  assert(ow != nullptr);

  auto iw_rank = iw->rank();
  assert(iw_rank == fw->rank());
  assert(iw_rank == cw->rank());
  assert(iw_rank == ow->rank());

  uint32_t ne_w = 1;
  for (uint32_t i = 0; i < iw_rank; i++)
    ne_w *= iw->dim(i).value();

  assert(iw->dtype() == loco::DataType::FLOAT32);
  assert(fw->dtype() == loco::DataType::FLOAT32);
  assert(cw->dtype() == loco::DataType::FLOAT32);
  assert(ow->dtype() == loco::DataType::FLOAT32);

  // merged weights
  auto mw = _nctx->create<luci::CircleConst>();
  mw->dtype(iw->dtype());
  mw->rank(iw_rank);
  mw->dim(0) = 4u * iw->dim(0).value();
  for (uint32_t i = 1; i < iw_rank; i++)
    mw->dim(i) = iw->dim(i);
  mw->size<loco::DataType::FLOAT32>(4 * ne_w);
  mw->shape_status(luci::ShapeStatus::VALID);
  for (uint32_t i = 0; i < ne_w; ++i)
  {
    mw->at<loco::DataType::FLOAT32>(i + ne_w * 0) = iw->at<loco::DataType::FLOAT32>(i);
    mw->at<loco::DataType::FLOAT32>(i + ne_w * 1) = fw->at<loco::DataType::FLOAT32>(i);
    mw->at<loco::DataType::FLOAT32>(i + ne_w * 2) = cw->at<loco::DataType::FLOAT32>(i);
    mw->at<loco::DataType::FLOAT32>(i + ne_w * 3) = ow->at<loco::DataType::FLOAT32>(i);
  }
  return mw;
}

luci::CircleFullyConnected *UnrollLSTM::create_input_matmul(luci::CircleNode *input)
{
  assert(input != nullptr);

  // weights
  auto iw = loco::must_cast<luci::CircleConst *>(_lstm->input_to_input_weights());
  auto fw = loco::must_cast<luci::CircleConst *>(_lstm->input_to_forget_weights());
  auto cw = loco::must_cast<luci::CircleConst *>(_lstm->input_to_cell_weights());
  auto ow = loco::must_cast<luci::CircleConst *>(_lstm->input_to_output_weights());

  auto fcw = merged_weights(iw, fw, cw, ow);
  fcw->name(_name + "_fc_w");
  luci::add_origin(fcw, luci::get_origin(_lstm));

  // bias
  auto ib = loco::must_cast<luci::CircleConst *>(_lstm->input_gate_bias());
  auto fb = loco::must_cast<luci::CircleConst *>(_lstm->forget_gate_bias());
  auto cb = loco::must_cast<luci::CircleConst *>(_lstm->cell_gate_bias());
  auto ob = loco::must_cast<luci::CircleConst *>(_lstm->output_gate_bias());

  auto fcb = merged_weights(ib, fb, cb, ob);
  fcb->name(_name + "_fc_b");
  luci::add_origin(fcb, luci::get_origin(_lstm));

  auto fc = _nctx->create<luci::CircleFullyConnected>();
  fc->input(input);
  fc->weights(fcw);
  fc->bias(fcb);
  fc->fusedActivationFunction(luci::FusedActFunc::NONE);
  fc->name(_name + "_fc");
  luci::add_origin(fc, luci::get_origin(_lstm));

  return fc;
}

std::vector<luci::CircleSplitOut *> UnrollLSTM::matmul_splits(luci::CircleNode *input,
                                                              uint32_t step)
{
  assert(input != nullptr);
  assert(step < _timesteps);

  std::string split_name = _name + "_sp" + std::to_string(step);

  auto split_dim = _nctx->create<luci::CircleConst>();
  split_dim->dtype(loco::DataType::S32);
  split_dim->rank(1);
  split_dim->dim(0) = 1;
  split_dim->size<loco::DataType::S32>(1);
  split_dim->at<loco::DataType::S32>(0) = 1;
  split_dim->shape_status(luci::ShapeStatus::VALID);
  split_dim->name(split_name + "_dim");
  luci::add_origin(split_dim, luci::get_origin(_lstm));

  auto split = _nctx->create<luci::CircleSplit>();
  split->num_split(4);
  split->split_dim(split_dim);
  split->input(input);
  split->name(split_name);
  luci::add_origin(split, luci::get_origin(_lstm));

  auto split_o0 = _nctx->create<luci::CircleSplitOut>();
  split_o0->input(split);
  split_o0->index(0);
  split_o0->name(split_name + "_spo0");
  luci::add_origin(split_o0, luci::get_origin(_lstm));

  auto split_o1 = _nctx->create<luci::CircleSplitOut>();
  split_o1->input(split);
  split_o1->index(1);
  split_o1->name(split_name + "_spo1");
  luci::add_origin(split_o1, luci::get_origin(_lstm));

  auto split_o2 = _nctx->create<luci::CircleSplitOut>();
  split_o2->input(split);
  split_o2->index(2);
  split_o2->name(split_name + "_spo2");
  luci::add_origin(split_o2, luci::get_origin(_lstm));

  auto split_o3 = _nctx->create<luci::CircleSplitOut>();
  split_o3->input(split);
  split_o3->index(3);
  split_o3->name(split_name + "_spo3");
  luci::add_origin(split_o3, luci::get_origin(_lstm));

  std::vector<luci::CircleSplitOut *> outs;
  outs.push_back(split_o0);
  outs.push_back(split_o1);
  outs.push_back(split_o2);
  outs.push_back(split_o3);
  return outs;
}

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

  if (ulstm._timesteps > 1)
  {
    // Transpose to switch batch <-> timesteps
    // NOTE TF uses Reshape when batch is 1 but as there is Transpose->Reshape
    //      Pass, we can just use Transpose for both cases
    auto transpose = ulstm.first_transpose(input);
    input = transpose;
  }

  auto unpacks = ulstm.input_unpacks(input);
  assert(unpacks.size() == ulstm._timesteps);
  uint32_t step = 0;
  auto unpackout = unpacks[step];

  // First FC
  auto fc_1 = ulstm.create_input_matmul(unpackout);
  assert(fc_1 != nullptr);
  auto splits = ulstm.matmul_splits(fc_1, step);
  assert(splits.size() == 4);

  // TODO implement
  (void)splits;

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
