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

#ifndef __GRAPH_TESTCASE_H__
#define __GRAPH_TESTCASE_H__

#include "loco/IR/Graph.h"
#include "loco/IR/PermutingCodec.h"

#include "GraphBuilder.h"

enum class GraphCode
{
  Identity,
  ConstGen,
  Relu,
  FeatureCodec,
  AvgPool2D,
  DepthwiseConv2D,
  TransposedConv2D,
  MaxPool2D,
  TensorBroadcast,
  TensorConcat,
  TensorTranspose,
  FixedReshape,
};

namespace
{

template <loco::Domain D> loco::Permutation<D> make_NHWC_perm(void);

template <> loco::Permutation<loco::Domain::Feature> make_NHWC_perm(void)
{
  loco::Permutation<loco::Domain::Feature> perm;

  perm[loco::FeatureAxis::Count] = 0;
  perm[loco::FeatureAxis::Height] = 1;
  perm[loco::FeatureAxis::Width] = 2;
  perm[loco::FeatureAxis::Depth] = 3;

  return perm;
}

template <loco::Domain D> loco::Permutation<D> make_HWCN_perm(void);

/// @note  Also known as HWIO permutation
template <> loco::Permutation<loco::Domain::Filter> make_HWCN_perm(void)
{
  loco::Permutation<loco::Domain::Filter> perm;

  perm[loco::FilterAxis::Height] = 0;
  perm[loco::FilterAxis::Width] = 1;
  perm[loco::FilterAxis::Depth] = 2;
  perm[loco::FilterAxis::Count] = 3;

  return perm;
}

template <loco::Domain D> loco::Permutation<D> make_HWCM_perm(void);

template <> loco::Permutation<loco::Domain::DepthwiseFilter> make_HWCM_perm(void)
{
  loco::Permutation<loco::Domain::DepthwiseFilter> perm;

  perm[loco::DepthwiseFilterAxis::Height] = 0;
  perm[loco::DepthwiseFilterAxis::Width] = 1;
  perm[loco::DepthwiseFilterAxis::Depth] = 2;
  perm[loco::DepthwiseFilterAxis::Multiplier] = 3;

  return perm;
}

} // namespace

template <GraphCode Code> class GraphTestcase;

template <> class GraphTestcase<GraphCode::Identity> final
{
private:
  void init(std::initializer_list<uint32_t> dims)
  {
    // Create a sample network
    _graph = loco::make_graph();

    auto graph_builder = make_graph_builder(_graph.get());

    pull_node = graph_builder->push<InputLayer>()->name("input")->shape(dims)->node();
    push_node = graph_builder->push<OutputLayer>()->name("output")->node();
  }

public:
  // NOTE This default constructor guarantees backward compatbility.
  GraphTestcase() { init({1, 4, 8, 3}); }
  GraphTestcase(std::initializer_list<uint32_t> dims) { init(dims); }

public:
  loco::Graph *graph() { return _graph.get(); }

  loco::Pull *pull_node = nullptr;
  loco::Push *push_node = nullptr;

private:
  std::unique_ptr<loco::Graph> _graph;
};

template <> class GraphTestcase<GraphCode::ConstGen> final
{
public:
  GraphTestcase()
  {
    _graph = loco::make_graph();

    auto graph_builder = make_graph_builder(_graph.get());

    const_node = graph_builder->push<ConstGenLayer>()->node();

    push_node = graph_builder->push<OutputLayer>()->name("output")->node();
  }

public:
  loco::Graph *graph() { return _graph.get(); }

  loco::ConstGen *const_node = nullptr;
  loco::Push *push_node = nullptr;

private:
  std::unique_ptr<loco::Graph> _graph;
};

template <> class GraphTestcase<GraphCode::Relu> final
{
public:
  GraphTestcase()
  {
    // Create a sample network
    _graph = loco::make_graph();

    auto graph_builder = make_graph_builder(_graph.get());

    pull_node = graph_builder->push<InputLayer>()->name("input")->node();
    relu_node = graph_builder->push<ReLULayer>()->node();
    push_node = graph_builder->push<OutputLayer>()->name("output")->node();
  }

public:
  loco::Graph *graph() { return _graph.get(); }

  loco::Pull *pull_node = nullptr;
  loco::ReLU *relu_node = nullptr;
  loco::Push *push_node = nullptr;

private:
  std::unique_ptr<loco::Graph> _graph;
};

template <> class GraphTestcase<GraphCode::FeatureCodec> final
{
public:
  GraphTestcase()
  {
    using namespace loco;

    Permutation<Domain::Feature> perm;

    perm[FeatureAxis::Count] = 0;
    perm[FeatureAxis::Height] = 1;
    perm[FeatureAxis::Width] = 2;
    perm[FeatureAxis::Depth] = 3;

    // Create a sample network
    _graph = make_graph();

    auto graph_builder = make_graph_builder(_graph.get());

    pull_node = graph_builder->push<InputLayer>()->name("input")->node();
    encode_node = graph_builder->push<FeatureEncodeLayer>()->perm(perm)->node();
    decode_node = graph_builder->push<FeatureDecodeLayer>()->perm(perm)->node();
    push_node = graph_builder->push<OutputLayer>()->name("output")->node();
  }

public:
  loco::Graph *graph() { return _graph.get(); }

  loco::Pull *pull_node = nullptr;
  loco::FeatureEncode *encode_node = nullptr;
  loco::FeatureDecode *decode_node = nullptr;
  loco::Push *push_node = nullptr;

private:
  std::unique_ptr<loco::Graph> _graph;
};

template <> class GraphTestcase<GraphCode::AvgPool2D> final
{
public:
  GraphTestcase()
  {
    using namespace loco;

    // Create a sample network
    _graph = make_graph();

    // Create Graph Input/Output
    auto graph_input = _graph->inputs()->create();
    auto graph_output = _graph->outputs()->create();

    graph_input->name("input");
    graph_output->name("output");

    // Create and connect nodes
    pull_node = _graph->nodes()->create<Pull>();
    pull_node->index(0);

    encode_node = _graph->nodes()->create<FeatureEncode>();
    encode_node->input(pull_node);

    avgpool2d_node = _graph->nodes()->create<AvgPool2D>();
    avgpool2d_node->ifm(encode_node);

    decode_node = _graph->nodes()->create<FeatureDecode>();
    decode_node->input(avgpool2d_node);

    push_node = _graph->nodes()->create<loco::Push>();
    push_node->index(0);
    push_node->from(decode_node);

    // Create a link between input/output and corresponding nodes
    loco::link(graph_input, pull_node);
    loco::link(graph_output, push_node);
  }

public:
  loco::Graph *graph() { return _graph.get(); }

  loco::Pull *pull_node = nullptr;
  loco::FeatureEncode *encode_node = nullptr;
  loco::AvgPool2D *avgpool2d_node = nullptr;
  loco::FeatureDecode *decode_node = nullptr;
  loco::Push *push_node = nullptr;

private:
  std::unique_ptr<loco::Graph> _graph;
};

template <> class GraphTestcase<GraphCode::DepthwiseConv2D> final
{
public:
  GraphTestcase()
  {
    using namespace loco;

    _graph = make_graph();

    auto graph_builder = make_graph_builder(_graph.get());

    Permutation<Domain::Feature> perm = make_NHWC_perm<Domain::Feature>();
    Permutation<Domain::DepthwiseFilter> filter_perm = make_HWCM_perm<Domain::DepthwiseFilter>();

    pull_node = graph_builder->push<InputLayer>()->name("input")->node();
    encode_node = graph_builder->push<FeatureEncodeLayer>()->perm(perm)->node();

    const_node = graph_builder->push<ConstGenLayer>()->node();

    filter_encode_node =
      graph_builder->push<DepthwiseFilterEncodeLayer>()->perm(filter_perm)->node();

    depthwiseconv2d_node = graph_builder->push<DepthwiseConv2DLayer>()->node();

    decode_node = graph_builder->push<FeatureDecodeLayer>()->perm(perm)->node();
    push_node = graph_builder->push<OutputLayer>()->name("output")->node();
  }

public:
  loco::Graph *graph() { return _graph.get(); }

  loco::Pull *pull_node = nullptr;
  loco::FeatureEncode *encode_node = nullptr;
  loco::ConstGen *const_node = nullptr;
  loco::DepthwiseFilterEncode *filter_encode_node = nullptr;
  loco::DepthwiseConv2D *depthwiseconv2d_node = nullptr;
  loco::FeatureDecode *decode_node = nullptr;
  loco::Push *push_node = nullptr;

private:
  std::unique_ptr<loco::Graph> _graph;
};

template <> class GraphTestcase<GraphCode::TransposedConv2D> final
{
public:
  GraphTestcase()
  {
    using namespace loco;

    // Prepare permutations
    Permutation<Domain::Feature> feature_perm = make_NHWC_perm<Domain::Feature>();
    Permutation<Domain::Filter> filter_perm = make_HWCN_perm<Domain::Filter>();

    // Build graph
    _graph = make_graph();
    auto graph_builder = make_graph_builder(_graph.get());

    pull_node = graph_builder->push<InputLayer>()->name("input")->node();
    encode_node = graph_builder->push<FeatureEncodeLayer>()->perm(feature_perm)->node();
    const_node = graph_builder->push<ConstGenLayer>()->node();
    filter_encode_node = graph_builder->push<FilterEncodeLayer>()->perm(filter_perm)->node();
    tr_conv2d_node = graph_builder->push<TransposedConv2DLayer>()->node();
    decode_node = graph_builder->push<FeatureDecodeLayer>()->perm(feature_perm)->node();
    push_node = graph_builder->push<OutputLayer>()->name("output")->node();
  }

public:
  loco::Graph *graph() { return _graph.get(); }

  loco::Pull *pull_node = nullptr;
  loco::FeatureEncode *encode_node = nullptr;
  loco::ConstGen *const_node = nullptr;
  loco::FilterEncode *filter_encode_node = nullptr;
  loco::TransposedConv2D *tr_conv2d_node = nullptr;
  loco::FeatureDecode *decode_node = nullptr;
  loco::Push *push_node = nullptr;

private:
  std::unique_ptr<loco::Graph> _graph;
};

template <> class GraphTestcase<GraphCode::MaxPool2D> final
{
public:
  GraphTestcase()
  {
    using namespace loco;

    // Create a sample network
    _graph = make_graph();

    // Create Graph Input/Output
    auto graph_input = _graph->inputs()->create();
    auto graph_output = _graph->outputs()->create();

    graph_input->name("input");
    graph_output->name("output");

    // Create and connect nodes
    pull_node = _graph->nodes()->create<Pull>();
    pull_node->index(0);

    encode_node = _graph->nodes()->create<FeatureEncode>();
    encode_node->input(pull_node);

    maxpool2d_node = _graph->nodes()->create<MaxPool2D>();
    maxpool2d_node->ifm(encode_node);

    decode_node = _graph->nodes()->create<FeatureDecode>();
    decode_node->input(maxpool2d_node);

    push_node = _graph->nodes()->create<loco::Push>();
    push_node->index(0);
    push_node->from(decode_node);

    // Create a link between input/output and corresponding nodes
    loco::link(graph_input, pull_node);
    loco::link(graph_output, push_node);
  }

public:
  loco::Graph *graph() { return _graph.get(); }

  loco::Pull *pull_node = nullptr;
  loco::FeatureEncode *encode_node = nullptr;
  loco::MaxPool2D *maxpool2d_node = nullptr;
  loco::FeatureDecode *decode_node = nullptr;
  loco::Push *push_node = nullptr;

private:
  std::unique_ptr<loco::Graph> _graph;
};

template <> class GraphTestcase<GraphCode::TensorConcat> final
{
public:
  GraphTestcase()
  {
    using namespace loco;

    // Create a sample network
    _graph = make_graph();

    // Create Graph Input/Output
    auto graph_lhs = _graph->inputs()->create();
    auto graph_rhs = _graph->inputs()->create();
    auto graph_out = _graph->outputs()->create();

    graph_lhs->name("lhs");
    graph_rhs->name("rhs");
    graph_out->name("output");

    // Create and connect nodes
    lhs_node = _graph->nodes()->create<Pull>();
    lhs_node->index(0);

    rhs_node = _graph->nodes()->create<Pull>();
    rhs_node->index(1);

    concat_node = _graph->nodes()->create<TensorConcat>();
    concat_node->lhs(lhs_node);
    concat_node->rhs(rhs_node);

    push_node = _graph->nodes()->create<loco::Push>();
    push_node->index(0);
    push_node->from(concat_node);

    // Create a link between input/output and corresponding nodes
    loco::link(graph_lhs, lhs_node);
    loco::link(graph_rhs, rhs_node);
    loco::link(graph_out, push_node);
  }

public:
  loco::Graph *graph() { return _graph.get(); }

  loco::Pull *lhs_node = nullptr;
  loco::Pull *rhs_node = nullptr;
  loco::TensorConcat *concat_node = nullptr;
  loco::Push *push_node = nullptr;

private:
  std::unique_ptr<loco::Graph> _graph;
};

template <> class GraphTestcase<GraphCode::FixedReshape> final
{
public:
  GraphTestcase()
  {
    _graph = loco::make_graph();

    auto graph_builder = make_graph_builder(_graph.get());

    pull_node = graph_builder->push<InputLayer>()->name("input")->node();
    reshape_node = graph_builder->push<FixedReshapeLayer>()->node();
    push_node = graph_builder->push<OutputLayer>()->name("output")->node();
  }

public:
  loco::Graph *graph() { return _graph.get(); }

  loco::Pull *pull_node = nullptr;
  loco::FixedReshape *reshape_node = nullptr;
  loco::Push *push_node = nullptr;

private:
  std::unique_ptr<loco::Graph> _graph;
};

template <> class GraphTestcase<GraphCode::TensorBroadcast> final
{
public:
  GraphTestcase(std::initializer_list<uint32_t> dims)
  {
    _graph = loco::make_graph();

    auto graph_builder = make_graph_builder(_graph.get());

    pull_node = graph_builder->push<InputLayer>()->name("input")->shape(dims)->node();
    broadcast_node = graph_builder->push<TensorBroadcastLayer>()->node();
    push_node = graph_builder->push<OutputLayer>()->name("output")->node();
  }

public:
  loco::Graph *graph(void) { return _graph.get(); }

  loco::Pull *pull_node = nullptr;
  loco::TensorBroadcast *broadcast_node = nullptr;
  loco::Push *push_node = nullptr;

private:
  std::unique_ptr<loco::Graph> _graph;
};

template <> class GraphTestcase<GraphCode::TensorTranspose> final
{
public:
  GraphTestcase()
  {
    using namespace loco;

    // Create a sample network
    _graph = make_graph();

    // Create Graph Input/Output
    auto graph_input = _graph->inputs()->create();
    auto graph_output = _graph->outputs()->create();

    graph_input->name("input");
    graph_output->name("output");

    // Create and connect nodes
    pull_node = _graph->nodes()->create<Pull>();
    pull_node->index(0);

    transpose_node = _graph->nodes()->create<TensorTranspose>();
    transpose_node->input(pull_node);

    push_node = _graph->nodes()->create<loco::Push>();
    push_node->index(0);
    push_node->from(transpose_node);

    // Create a link between input/output and corresponding nodes
    loco::link(graph_input, pull_node);
    loco::link(graph_output, push_node);
  }

public:
  loco::Graph *graph() { return _graph.get(); }

  loco::Pull *pull_node = nullptr;
  loco::TensorTranspose *transpose_node = nullptr;
  loco::Push *push_node = nullptr;

private:
  std::unique_ptr<loco::Graph> _graph;
};

#endif // __GRAPH_TESTCASE_H__
