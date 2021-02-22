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

#ifndef __TEST_GRAPH_H__
#define __TEST_GRAPH_H__

#include "Dialect/IR/TFLNodes.h"
#include "GraphBlock.h"
#include "TestHelper.h"

#include <loco.h>

#include <cassert>

namespace exo
{
namespace test
{

class TestGraph
{
public:
  std::unique_ptr<loco::Graph> g;
  loco::Pull *pull;
  loco::Push *push;

  TestGraph() // creates Pull and Push
  {
    g = loco::make_graph();

    pull = g->nodes()->create<loco::Pull>();

    push = g->nodes()->create<loco::Push>();

    auto input = g->inputs()->create();
    {
      input->name("input");
      loco::link(input, pull);
    }
    auto output = g->outputs()->create();
    {
      output->name("output");
      loco::link(output, push);
    }

    _next_input = pull;
  }

  loco::Graph *graph() { return g.get(); }

  /// @brief Creates node with NO arg and appends it to graph
  template <class T> T *append()
  {
    auto node = g->nodes()->create<T>();
    _next_input = node;

    return node;
  }

  /// @brief Creates op T (arity=1) with arg1 as an input and appends it to graph
  template <class T> T *append(loco::Node *arg1)
  {
    auto node = g->nodes()->create<T>();
    setInput(node, arg1);
    _next_input = node;

    return node;
  }

  /// @brief Creates op T (arity=2) with arg1, arg2 as inputs and appends it to graph
  template <class T> T *append(loco::Node *arg1, loco::Node *arg2)
  {
    auto node = g->nodes()->create<T>();
    setInput(node, arg1, arg2);
    _next_input = node;

    return node;
  }

  /// @brief Creates op T (arity=3) with arg1, arg2, arg3 as inputs and appends it to graph
  template <class T> T *append(loco::Node *arg1, loco::Node *arg2, loco::Node *arg3)
  {
    auto node = g->nodes()->create<T>();
    setInput(node, arg1, arg2, arg3);
    _next_input = node;

    return node;
  }

  // push will get the last appended node
  void complete() { push->from(_next_input); }

  void complete(loco::Node *last_node) { push->from(last_node); }

private:
  // arity 1
  void setInput(loco::Node *node, loco::Node *) { assert(false && "NYI"); };

  void setInput(loco::AvgPool2D *node, loco::Node *input) { node->ifm(input); }
  void setInput(loco::BiasDecode *node, loco::Node *input) { node->input(input); };
  void setInput(loco::BiasEncode *node, loco::Node *input) { node->input(input); };
  void setInput(loco::FeatureDecode *node, loco::Node *input) { node->input(input); };
  void setInput(loco::FeatureEncode *node, loco::Node *input) { node->input(input); };
  void setInput(loco::MaxPool2D *node, loco::Node *input) { node->ifm(input); }
  void setInput(loco::Push *node, loco::Node *input) { node->from(input); };
  void setInput(loco::ReLU *node, loco::Node *input) { node->input(input); };
  void setInput(loco::ReLU6 *node, loco::Node *input) { node->input(input); };
  void setInput(loco::Tanh *node, loco::Node *input) { node->input(input); };
  void setInput(loco::TensorTranspose *node, loco::Node *input) { node->input(input); };

  void setInput(locoex::TFLAveragePool2D *node, loco::Node *input) { node->value(input); };
  void setInput(locoex::TFLMaxPool2D *node, loco::Node *input) { node->value(input); };
  void setInput(locoex::TFLRelu *node, loco::Node *input) { node->features(input); };
  void setInput(locoex::TFLRelu6 *node, loco::Node *input) { node->features(input); };

  // arity 2
  void setInput(loco::Node *node, loco::Node *, loco::Node *) { assert(false && "NYI"); };

  void setInput(loco::Conv2D *node, loco::Node *input, loco::Node *filter)
  {
    node->ifm(input);
    node->ker(filter);
  }

  void setInput(loco::EltwiseAdd *node, loco::Node *arg1, loco::Node *arg2)
  {
    node->lhs(arg1);
    node->rhs(arg2);
  };

  void setInput(loco::FeatureBiasAdd *node, loco::Node *arg1, loco::Node *arg2)
  {
    node->value(arg1);
    node->bias(arg2);
  };

  void setInput(locoex::TFLAdd *node, loco::Node *arg1, loco::Node *arg2)
  {
    node->x(arg1);
    node->y(arg2);
  };

  void setInput(locoex::TFLMul *node, loco::Node *arg1, loco::Node *arg2)
  {
    node->x(arg1);
    node->y(arg2);
  };

  void setInput(locoex::TFLSub *node, loco::Node *arg1, loco::Node *arg2)
  {
    node->x(arg1);
    node->y(arg2);
  };

  void setInput(locoex::TFLTranspose *node, loco::Node *arg1, loco::Node *arg2)
  {
    node->a(arg1);
    node->perm(arg2);
  };

  // arity 3
  void setInput(loco::Node *node, loco::Node *, loco::Node *, loco::Node *)
  {
    assert(false && "NYI");
  };

  void setInput(locoex::TFLConv2D *node, loco::Node *input, loco::Node *filter, loco::Node *bias)
  {
    node->input(input);
    node->filter(filter);
    node->bias(bias);
  }

private:
  loco::Node *_next_input;
};

enum class ExampleGraphType
{
  FeatureBiasAdd,
  ConstGen_ReLU,
  FilterEncode_FilterDecode,
  Transpose,

  TFLTranspose,
};

template <ExampleGraphType T> class ExampleGraph;

/**
 * @brief Class to create the following:
 *
 *   Pull - FeatureEncoder - FeatureBiasAdd - FeatureDecode - Push
 *                             |
 *     ConstGen - BiasEncode --+
 */
template <> class ExampleGraph<ExampleGraphType::FeatureBiasAdd> : public TestGraph
{
public:
  loco::FeatureEncode *fea_enc = nullptr;
  loco::ConstGen *constgen = nullptr;
  loco::BiasEncode *bias_enc = nullptr;
  loco::FeatureBiasAdd *fea_bias_add = nullptr;
  loco::FeatureDecode *fea_dec = nullptr;

public:
  ExampleGraph()
  {
    fea_enc = exo::make_feature_encode<exo::FeatureLayout::NHWC>(pull);
    constgen = append<loco::ConstGen>();
    bias_enc = append<loco::BiasEncode>(constgen);
    fea_bias_add = append<loco::FeatureBiasAdd>(fea_enc, bias_enc);
    fea_dec = exo::make_feature_decode<exo::FeatureLayout::NHWC>(fea_bias_add);
    complete(fea_dec);
  }
};

/**
 * @brief Class to creates the following:
 *
 *     ConstGen -- ReLU -- Push
 */
template <> class ExampleGraph<ExampleGraphType::ConstGen_ReLU> : public TestGraph
{
public:
  loco::ConstGen *constgen = nullptr;
  loco::ReLU *relu = nullptr;

public:
  ExampleGraph()
  {
    constgen = append<loco::ConstGen>();
    relu = append<loco::ReLU>(constgen);
    complete(relu);
  }
};

/**
 * @brief Class to creates the following:
 *
 *     Pull -- Transpose -- Push
 */
template <> class ExampleGraph<ExampleGraphType::Transpose> : public TestGraph
{
public:
  loco::TensorTranspose *transpose = nullptr;

public:
  ExampleGraph()
  {
    transpose = append<loco::TensorTranspose>(pull);
    complete(transpose);
  }
};

/**
 * @brief Class to creates the following:
 *
 *     Pull -- FilterEncode -- FilterDecode -- Push
 */
template <> class ExampleGraph<ExampleGraphType::FilterEncode_FilterDecode> : public TestGraph
{
public:
  loco::FilterEncode *filterEncode = nullptr;
  loco::FilterDecode *filterDecode = nullptr;

public:
  ExampleGraph()
  {
    filterEncode = exo::make_filter_encode<exo::FilterLayout::HWIO>(pull); // from Tensorflow
    filterDecode =
      exo::make_filter_decode<exo::FilterLayout::OHWI>(filterEncode); // to Tensorflow Lite
    complete(filterDecode);
  }
};

/**
 * @brief Class to create the following:
 *
 *     Pull -- TFLTranspose -- Push
 */
template <> class ExampleGraph<ExampleGraphType::TFLTranspose> : public TestGraph
{
public:
  loco::ConstGen *const_perm = nullptr;
  locoex::TFLTranspose *tfl_transpose = nullptr;

public:
  ExampleGraph()
  {
    const_perm = append<loco::ConstGen>();
    tfl_transpose = append<locoex::TFLTranspose>(pull, const_perm);
    complete(tfl_transpose);
  }
};

} // namespace test
} // namespace exo

#endif // __TEST_GRAPH_H__
