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

#include <logo/ReorderDecodePass.h>

#include <loco/IR/CanonicalDialect.h>
#include <loco/IR/CanonicalNode.h>

#include <loco/IR/Nodes.h>

#include <cassert>
#include <queue>

namespace
{

bool isTensorBiasAdd(const loco::Node *node)
{
  return node->opnum() == static_cast<uint32_t>(loco::CanonicalOpcode::TensorBiasAdd);
}

bool isReLU(const loco::Node *node)
{
  return node->opnum() == static_cast<uint32_t>(loco::CanonicalOpcode::ReLU);
}

// Update queue
class Collector final : public loco::CanonicalNodeMutableVisitor<void>
{
public:
  Collector(std::queue<loco::FeatureDecode *> *out) : _out{out}
  {
    // DO NOTHING
  }

  void visit(loco::FeatureDecode *node) final
  {
    if (node->input() != nullptr)
    {
      _out->push(node);
    }
  }

  void visit(loco::Node *) final { return; }

private:
  // TODO This definition should be revised to support other decode operations
  std::queue<loco::FeatureDecode *> *_out;
};

void gather_candidates(loco::Graph *g, std::queue<loco::FeatureDecode *> &q)
{
  Collector collector{&q};

  for (auto node : loco::all_nodes(g))
  {
    if (node->dialect() == loco::CanonicalDialect::get())
    {
      auto canonical_node = loco::must_cast<loco::CanonicalNode *>(node);
      canonical_node->accept(&collector);
    }
  }
}

} // namespace

namespace logo
{

bool ReorderDecodePass<loco::TensorBiasAdd>::run(loco::Graph *g)
{
  std::queue<loco::FeatureDecode *> q;

  // Update queue
  class Collector final : public loco::CanonicalNodeMutableVisitor<void>
  {
  public:
    Collector(std::queue<loco::FeatureDecode *> *out) : _out{out}
    {
      // DO NOTHING
    }

    void visit(loco::FeatureDecode *node) final
    {
      if (node->input() != nullptr)
      {
        _out->push(node);
      }
    }

    void visit(loco::Node *) final { return; }

  private:
    // TODO This definition should be revised to support other decode operations
    std::queue<loco::FeatureDecode *> *_out;
  };

  Collector collector{&q};

  for (auto node : loco::all_nodes(g))
  {
    if (node->dialect() == loco::CanonicalDialect::get())
    {
      auto canonical_node = loco::must_cast<loco::CanonicalNode *>(node);
      canonical_node->accept(&collector);
    }
  }

  bool changed = false;

  while (!q.empty())
  {
    auto cur_decode = q.front();
    q.pop();

    // Collector IS EXPECTED TO guarantee this property
    assert(cur_decode->input() != nullptr);

    for (auto u : loco::succs(cur_decode))
    {
      /**
       * Let us consider the following graph:
       *
       *   A ---> FeatureDecode(1) ---> ReLU(2)
       *
       * ReorderDecodeTransform rewrites this graph as follows:
       *
       *   A -+-> FeatureDecode(1) ---> ReLU(2)
       *      |
       *      +-> ReLU(2') ---> FeatureDecode(1')
       *
       * Let us feed this updates graph to ReorderDecodeTransform.
       *
       * The naive implementation will create a new ReLU->FeatureDecode
       * chain again, and results in unbounded graph blow-up.
       *
       *   A -+-> FeatureDeocde(1) ---> ReLU(2)
       *      |
       *      +-> ReLU(2') ---> FeatureDecode(1')
       *      |
       *      +-> ReLU(2'') ---> FeatureDecode(1'')
       *
       * This check prevents such unbounded graph blow-up.
       */
      if (loco::succs(u).empty())
      {
        continue;
      }

      // Q. Is it better to create an independent transform for this rewriting rule?
      if (isTensorBiasAdd(u))
      {
        auto old_badd = loco::must_cast<loco::TensorBiasAdd *>(u);

        /**
         * Let us consider the following example:
         *
         * A -=-> FeatureDecode(1) -+-> TensorBiasAdd(2) -+-> B1
         *                          |                     |
         *                          |                     +-> B2
         *                          |                     |
         *                          |                     +-> ...
         *                          |
         *                          +-> ...
         *
         * At this point, "cur_decode" points to (1) and "u" points to (2).
         *
         * First rewrite the graph as follows:
         *
         *    A -+-> FeatureBiasAdd(2') ---> FeatureDecode(1') -+-> B1
         *       |                                              |
         *       |                                              +-> B2
         *       |                                              |
         *       |                                              +-> ...
         *       |
         *       +-> FeatureDecode(1) -+-> TensorBiasAdd(2) ; NO USE
         *                             |
         *                             +-> ...
         *
         * Q. Is it safe to apply this transform without "decoder" check?
         */
        auto new_badd = g->nodes()->create<loco::FeatureBiasAdd>();
        auto new_decode = g->nodes()->create<loco::FeatureDecode>();

        new_badd->value(cur_decode->input());
        new_badd->bias(old_badd->bias());

        new_decode->input(new_badd);
        new_decode->decoder(cur_decode->decoder()->clone());

        loco::replace(u).with(new_decode);

        // Enque FeatureDeocde(1') for the further optimization.
        q.push(new_decode);

        changed = true;
      }
    }
  }

  return changed;
}

bool ReorderDecodePass<loco::ReLU>::run(loco::Graph *g)
{
  std::queue<loco::FeatureDecode *> q;

  // Update queue
  class Collector final : public loco::CanonicalNodeMutableVisitor<void>
  {
  public:
    Collector(std::queue<loco::FeatureDecode *> *out) : _out{out}
    {
      // DO NOTHING
    }

    void visit(loco::FeatureDecode *node) final
    {
      if (node->input() != nullptr)
      {
        _out->push(node);
      }
    }

    void visit(loco::Node *) final { return; }

  private:
    // TODO This definition should be revised to support other decode operations
    std::queue<loco::FeatureDecode *> *_out;
  };

  Collector collector{&q};

  for (auto node : loco::all_nodes(g))
  {
    if (node->dialect() == loco::CanonicalDialect::get())
    {
      auto canonical_node = loco::must_cast<loco::CanonicalNode *>(node);
      canonical_node->accept(&collector);
    }
  }

  bool changed = false;

  while (!q.empty())
  {
    auto cur_decode = q.front();
    q.pop();

    // Collector IS EXPECTED TO guarantee this property
    assert(cur_decode->input() != nullptr);

    for (auto u : loco::succs(cur_decode))
    {
      /**
       * Let us consider the following graph:
       *
       *   A ---> FeatureDecode(1) ---> ReLU(2)
       *
       * ReorderDecodeTransform rewrites this graph as follows:
       *
       *   A -+-> FeatureDecode(1) ---> ReLU(2)
       *      |
       *      +-> ReLU(2') ---> FeatureDecode(1')
       *
       * Let us feed this updates graph to ReorderDecodeTransform.
       *
       * The naive implementation will create a new ReLU->FeatureDecode
       * chain again, and results in unbounded graph blow-up.
       *
       *   A -+-> FeatureDeocde(1) ---> ReLU(2)
       *      |
       *      +-> ReLU(2') ---> FeatureDecode(1')
       *      |
       *      +-> ReLU(2'') ---> FeatureDecode(1'')
       *
       * This check prevents such unbounded graph blow-up.
       */
      if (loco::succs(u).empty())
      {
        continue;
      }

      if (isReLU(u))
      {
        /**
         * Let us consider the following example:
         *
         * A -=-> FeatureDecode(1) -+-> ReLU(2) -+-> B1
         *                          |            |
         *                          |            +-> B2
         *                          |            |
         *                          |            +-> ...
         *                          |
         *                          +-> ...
         *
         * At this point, "cur_decode" points to FeatureDecode(1) and "u" points to ReLU(2).
         *
         * First rewrite the graph as follows:
         *
         *    A -+-> ReLU(2') ---> FeatureDecode(1') -+-> B1
         *       |                                    |
         *       |                                    +-> B2
         *       |                                    |
         *       |                                    +-> ...
         *       |
         *       +-> FeatureDecode -+-> ReLU(2) ; NO USE
         *                          |
         *                          +-> ...
         */
        auto new_relu = g->nodes()->create<loco::ReLU>();
        auto new_decode = g->nodes()->create<loco::FeatureDecode>();

        new_relu->input(cur_decode->input());

        new_decode->input(new_relu);
        new_decode->decoder(cur_decode->decoder()->clone());

        loco::replace(u).with(new_decode);

        /**
         * Enque FeatureDeocde(1') for the further optimization.
         */
        q.push(new_decode);

        changed = true;
      }
    }
  }

  return changed;
}

} // namespace logo
