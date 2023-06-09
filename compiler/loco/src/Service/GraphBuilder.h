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

#ifndef __GRAPH_BUILDER_H__
#define __GRAPH_BUILDER_H__

// loco-internal headers
#include "loco/IR/Graph.h"

// C++ standard headers
#include <memory>
#include <stack>

//
// This file includes a stack-based loco graph builder
//
// HOW TO USE
//
//   loco::Graph *g = ...
//   auto builder = make_graph_builder(g);
//
//   builder->push<YourAwesomeLayer>(...);
//

class GraphBuilder final
{
public:
  class Stack final
  {
  public:
    Stack() = default;

  public:
    loco::Node *top(void) const { return _content.top(); }

  public:
    loco::Node *pop(void)
    {
      auto ret = top();
      _content.pop();
      return ret;
    }

  public:
    void push(loco::Node *node) { _content.push(node); }

  private:
    std::stack<loco::Node *> _content;
  };

  class Context final
  {
  public:
    Context(loco::Graph *graph) : _graph{graph}
    {
      // DO NOTHING
    }

  public:
    loco::Graph *graph(void) { return _graph; }
    Stack *stack(void) { return &_stack; }

  private:
    loco::Graph *_graph = nullptr;
    Stack _stack;
  };

public:
  GraphBuilder(loco::Graph *graph) : _context{graph}
  {
    // DO NOTHING
  }

public:
  // "Layer" is in theory a subgraph builder.
  template <typename Layer, typename... Args>
  auto push(Args &&...args)
    -> decltype(static_cast<Layer *>(nullptr)->operator()(static_cast<Context *>(nullptr)))
  {
    Layer layer{std::forward<Args>(args)...};
    return layer(ctx());
  }

public:
  loco::Node *pop(void) { return ctx()->stack()->pop(); }

private:
  Context *ctx(void) { return &_context; }

private:
  Context _context;
};

static inline std::unique_ptr<GraphBuilder> make_graph_builder(loco::Graph *g)
{
  return std::make_unique<GraphBuilder>(g);
}

// "InputLayer" creates both GraphInput and Pull node at once
struct InputLayer final
{
  class Return
  {
  public:
    Return(loco::GraphInput *input, loco::Pull *node) : _input{input}, _node{node}
    {
      // DO NOTHING
    }

  public:
    loco::Pull *node(void) { return _node; }

  public:
    Return *name(const std::string &value)
    {
      _input->name(value);
      return this;
    }

  public:
    Return *shape(std::initializer_list<uint32_t> dims)
    {
      // TODO Uncomment this line when GraphInput is ready
      // _graph_input->shape(dims)
      _node->shape(dims);
      return this;
    }

  private:
    loco::GraphInput *_input = nullptr;
    loco::Pull *_node = nullptr;
  };

  std::unique_ptr<Return> operator()(GraphBuilder::Context *ctx)
  {
    auto input_index = ctx->graph()->inputs()->size();
    auto graph_input = ctx->graph()->inputs()->create();

    auto pull_node = ctx->graph()->nodes()->create<loco::Pull>();

    pull_node->index(input_index);

    loco::link(graph_input, pull_node);

    ctx->stack()->push(pull_node);

    return std::make_unique<Return>(graph_input, pull_node);
  }
};

// "OutputLayer" creates both GraphOutput and Push node at once.
struct OutputLayer final
{
  class Return
  {
  public:
    Return(loco::GraphOutput *output, loco::Push *node) : _output{output}, _node{node}
    {
      // DO NOTHING
    }

  public:
    loco::Push *node(void) { return _node; }

  public:
    Return *name(const std::string &value)
    {
      // TODO Uncomment this line when GraphOutput is ready
      // _graph_output->shape(dims)
      _output->name(value);
      return this;
    }

  private:
    loco::GraphOutput *_output = nullptr;
    loco::Push *_node = nullptr;
  };

  std::unique_ptr<Return> operator()(GraphBuilder::Context *ctx)
  {
    auto output_index = ctx->graph()->outputs()->size();
    auto graph_output = ctx->graph()->outputs()->create();

    auto push_node = ctx->graph()->nodes()->create<loco::Push>();

    push_node->from(ctx->stack()->pop());
    push_node->index(output_index);

    loco::link(graph_output, push_node);

    ctx->stack()->push(push_node);

    return std::make_unique<Return>(graph_output, push_node);
  }
};

struct ReLULayer final
{
  // This "Return" is unnecessary for ReLU as ReLU has no attributes), but
  // introduced for consistency.
  class Return
  {
  public:
    Return(loco::ReLU *node) : _node{node}
    {
      // DO NOTHING
    }

  public:
    loco::ReLU *node(void) { return _node; }

  private:
    loco::ReLU *_node = nullptr;
  };

  std::unique_ptr<Return> operator()(GraphBuilder::Context *ctx)
  {
    auto relu_node = ctx->graph()->nodes()->create<loco::ReLU>();

    relu_node->input(ctx->stack()->pop());

    ctx->stack()->push(relu_node);

    return std::make_unique<Return>(relu_node);
  }
};

struct ConstGenLayer final
{
  class Return
  {
  public:
    Return(loco::ConstGen *node) : _node{node}
    {
      // DO NOTHING
    }

  public:
    loco::ConstGen *node(void) { return _node; }

  private:
    loco::ConstGen *_node = nullptr;
  };

  std::unique_ptr<Return> operator()(GraphBuilder::Context *ctx)
  {
    auto const_node = ctx->graph()->nodes()->create<loco::ConstGen>();

    ctx->stack()->push(const_node);

    return std::make_unique<Return>(const_node);
  }
};

#include "loco/IR/PermutingCodec.h"

struct FeatureEncodeLayer final
{
  class Return
  {
  public:
    Return(loco::FeatureEncode *node) : _node{node}
    {
      // DO NOTHING
    }

  public:
    Return *perm(const loco::Permutation<loco::Domain::Feature> &perm)
    {
      using namespace loco;
      _node->encoder(std::make_unique<PermutingEncoder<Domain::Feature>>(perm));
      return this;
    }

  public:
    loco::FeatureEncode *node(void) { return _node; }

  private:
    loco::FeatureEncode *_node;
  };

  std::unique_ptr<Return> operator()(GraphBuilder::Context *ctx)
  {
    auto encode_node = ctx->graph()->nodes()->create<loco::FeatureEncode>();

    encode_node->input(ctx->stack()->pop());

    ctx->stack()->push(encode_node);

    return std::make_unique<Return>(encode_node);
  }
};

struct FeatureDecodeLayer final
{
  class Return
  {
  public:
    Return(loco::FeatureDecode *node) : _node{node}
    {
      // DO NOTHING
    }

  public:
    Return *perm(const loco::Permutation<loco::Domain::Feature> &perm)
    {
      using namespace loco;
      _node->decoder(std::make_unique<PermutingDecoder<Domain::Feature>>(perm));
      return this;
    }

  public:
    loco::FeatureDecode *node(void) { return _node; }

  private:
    loco::FeatureDecode *_node;
  };

  std::unique_ptr<Return> operator()(GraphBuilder::Context *ctx)
  {
    using namespace loco;

    auto decode_node = ctx->graph()->nodes()->create<FeatureDecode>();

    decode_node->input(ctx->stack()->pop());

    ctx->stack()->push(decode_node);

    return std::make_unique<Return>(decode_node);
  }
};

struct FilterEncodeLayer final
{
  class Return
  {
  public:
    Return(loco::FilterEncode *node) : _node{node}
    {
      // DO NOTHING
    }

  public:
    Return *perm(const loco::Permutation<loco::Domain::Filter> &perm)
    {
      auto encoder = std::make_unique<loco::PermutingEncoder<loco::Domain::Filter>>();
      encoder->perm(perm);
      _node->encoder(std::move(encoder));
      return this;
    }

  public:
    loco::FilterEncode *node(void) { return _node; }

  private:
    loco::FilterEncode *_node;
  };

  std::unique_ptr<Return> operator()(GraphBuilder::Context *ctx)
  {
    auto encode_node = ctx->graph()->nodes()->create<loco::FilterEncode>();

    encode_node->input(ctx->stack()->pop());

    ctx->stack()->push(encode_node);

    return std::make_unique<Return>(encode_node);
  }
};

struct DepthwiseFilterEncodeLayer final
{
  class Return
  {
  public:
    Return(loco::DepthwiseFilterEncode *node) : _node{node}
    {
      // DO NOTHING
    }

  public:
    Return *perm(const loco::Permutation<loco::Domain::DepthwiseFilter> &perm)
    {
      using namespace loco;
      _node->encoder(std::make_unique<PermutingEncoder<Domain::DepthwiseFilter>>(perm));
      return this;
    }

  public:
    loco::DepthwiseFilterEncode *node(void) { return _node; }

  private:
    loco::DepthwiseFilterEncode *_node;
  };

  std::unique_ptr<Return> operator()(GraphBuilder::Context *ctx)
  {
    auto encode_node = ctx->graph()->nodes()->create<loco::DepthwiseFilterEncode>();

    encode_node->input(ctx->stack()->pop());

    ctx->stack()->push(encode_node);

    return std::make_unique<Return>(encode_node);
  }
};

struct DepthwiseConv2DLayer final
{
  class Return
  {
  public:
    Return(loco::DepthwiseConv2D *node) : _node{node}
    {
      // DO NOTHING
    }

  public:
    loco::DepthwiseConv2D *node(void) { return _node; }

  private:
    loco::DepthwiseConv2D *_node;
  };

  std::unique_ptr<Return> operator()(GraphBuilder::Context *ctx)
  {
    auto depthwiseconv2d_node = ctx->graph()->nodes()->create<loco::DepthwiseConv2D>();

    depthwiseconv2d_node->ker(ctx->stack()->pop());
    depthwiseconv2d_node->ifm(ctx->stack()->pop());

    ctx->stack()->push(depthwiseconv2d_node);

    return std::make_unique<Return>(depthwiseconv2d_node);
  }
};

struct TransposedConv2DLayer final
{
  class Return
  {
  public:
    Return(loco::TransposedConv2D *node) : _node{node}
    {
      // DO NOTHING
    }

  public:
    loco::TransposedConv2D *node(void) { return _node; }

  private:
    loco::TransposedConv2D *_node;
  };

  std::unique_ptr<Return> operator()(GraphBuilder::Context *ctx)
  {
    auto tr_conv2d_node = ctx->graph()->nodes()->create<loco::TransposedConv2D>();

    tr_conv2d_node->ker(ctx->stack()->pop());
    tr_conv2d_node->ifm(ctx->stack()->pop());

    ctx->stack()->push(tr_conv2d_node);

    return std::make_unique<Return>(tr_conv2d_node);
  }
};

struct FixedReshapeLayer final
{
  class Return
  {
  public:
    Return(loco::FixedReshape *node) : _node{node}
    {
      // DO NOTHING
    }

  public:
    Return *shape(std::initializer_list<uint32_t> dims)
    {
      _node->shape(dims);
      return this;
    }

  public:
    loco::FixedReshape *node(void) { return _node; }

  private:
    loco::FixedReshape *_node = nullptr;
  };

  std::unique_ptr<Return> operator()(GraphBuilder::Context *ctx)
  {
    auto reshape_node = ctx->graph()->nodes()->create<loco::FixedReshape>();

    reshape_node->input(ctx->stack()->pop());

    ctx->stack()->push(reshape_node);

    return std::make_unique<Return>(reshape_node);
  }
};

struct TensorBroadcastLayer final
{
  class Return
  {
  public:
    Return(loco::TensorBroadcast *node) : _node{node}
    {
      // DO NOTHING
    }

  public:
    loco::TensorBroadcast *node(void) { return _node; }

  private:
    loco::TensorBroadcast *_node = nullptr;
  };

  std::unique_ptr<Return> operator()(GraphBuilder::Context *ctx)
  {
    auto broadcast_node = ctx->graph()->nodes()->create<loco::TensorBroadcast>();

    broadcast_node->input(ctx->stack()->pop());
    ctx->stack()->push(broadcast_node);

    return std::make_unique<Return>(broadcast_node);
  }
};

#endif // __GRAPH_BUILDER_H__
