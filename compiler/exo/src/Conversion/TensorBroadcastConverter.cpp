/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TensorBroadcastConverter.h"

#include "Dialect/IR/TFLDialect.h"
#include "Dialect/IR/TFLNodeVisitor.h"

#include <loco.h>
#include <loco/IR/CanonicalDialect.h>
#include <loco/IR/CanonicalNode.h>

#include <set>

namespace
{

template <class T> loco::TensorBroadcast *input_as_tbc(T *node)
{
  loco::TensorBroadcast *tbc = dynamic_cast<loco::TensorBroadcast *>(node->x());
  if (tbc == nullptr)
    tbc = dynamic_cast<loco::TensorBroadcast *>(node->y());

  return tbc;
}

struct Collector final : public locoex::TFLNodeMutableVisitor<void>
{
  using NodePair = std::pair<loco::TensorBroadcast *, loco::Node *>;

  void visit(locoex::TFLAdd *node) final
  {
    if (auto tbc = input_as_tbc<locoex::TFLAdd>(node))
    {
      NodePair pair(tbc, node);
      candidates.insert(pair);
    }
  }

  void visit(locoex::TFLDiv *node) final
  {
    if (auto tbc = input_as_tbc<locoex::TFLDiv>(node))
    {
      NodePair pair(tbc, node);
      candidates.insert(pair);
    }
  }

  void visit(locoex::TFLMul *node) final
  {
    if (auto tbc = input_as_tbc<locoex::TFLMul>(node))
    {
      NodePair pair(tbc, node);
      candidates.insert(pair);
    }
  }

  void visit(locoex::TFLSub *node) final
  {
    if (auto tbc = input_as_tbc<locoex::TFLSub>(node))
    {
      NodePair pair(tbc, node);
      candidates.insert(pair);
    }
  }

  void visit(locoex::TFLMaximum *node) final
  {
    if (auto tbc = input_as_tbc<locoex::TFLMaximum>(node))
    {
      NodePair pair(tbc, node);
      candidates.insert(pair);
    }
  }

  void visit(locoex::TFLNode *) final { return; }

  std::set<NodePair> candidates;
};

bool mapping_condition(Collector::NodePair &)
{
  // TODO fill condition

  return true;
}

template <class T> void jump_connection(loco::TensorBroadcast *tbc, T *tflnode)
{
  if (tflnode->x() == tbc)
    tflnode->x(tbc->input());
  else if (tflnode->y() == tbc)
    tflnode->y(tbc->input());
  else
    assert(false);

  tbc->input(nullptr);
}

} // namespace

namespace exo
{

/**
 * @brief  Disconnects loco::TensorBroadcast from the graph if following node
 *         is one of binary node: TFLAdd, TFLSub, TFLMul, TFLDiv, TFLMaximum
 *         and meets condition (TBA)
 * @note
 *         Before:
 *            x --- TensorBroadcast --- TFLXXX --- output
 *            y ----------------------/
 *
 *         After:
 *              --- TensorBroadcast ---
 *            x --- TFLXXX --- output
 *            y --/
 */
bool TensorBroadcastConverter::run(loco::Graph *graph)
{
  Collector collector;

  auto active_nodes = loco::active_nodes(loco::output_nodes(graph));

  for (auto node : active_nodes)
  {
    if (node->dialect() == locoex::TFLDialect::get())
    {
      auto tfl_node = dynamic_cast<locoex::TFLNode *>(node);
      tfl_node->accept(&collector);
    }
  }

  bool changed = false;

  for (auto pair : collector.candidates)
  {
    if (mapping_condition(pair))
    {
      loco::TensorBroadcast *tensorbroadcast = pair.first;
      if (auto tfladd = dynamic_cast<locoex::TFLAdd *>(pair.second))
      {
        jump_connection<locoex::TFLAdd>(tensorbroadcast, tfladd);
        changed = true;
      }
      else if (auto tfldiv = dynamic_cast<locoex::TFLDiv *>(pair.second))
      {
        jump_connection<locoex::TFLDiv>(tensorbroadcast, tfldiv);
        changed = true;
      }
      else if (auto tflmul = dynamic_cast<locoex::TFLMul *>(pair.second))
      {
        jump_connection<locoex::TFLMul>(tensorbroadcast, tflmul);
        changed = true;
      }
      else if (auto tflsub = dynamic_cast<locoex::TFLSub *>(pair.second))
      {
        jump_connection<locoex::TFLSub>(tensorbroadcast, tflsub);
        changed = true;
      }
      else if (auto tflmaximum = dynamic_cast<locoex::TFLMaximum *>(pair.second))
      {
        jump_connection<locoex::TFLMaximum>(tensorbroadcast, tflmaximum);
        changed = true;
      }
      else
      {
        assert(false);
      }
    }
  }

  return changed;
}

} // namespace exo
