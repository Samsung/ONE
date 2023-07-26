/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <logo/Phase.h>

#include <luci/test/TestIOGraph.h>

#include "luci/Pass/ConvertNCHWToNHWCPass.h"
#include "luci/Pass/CircleShapeInferencePass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

using namespace luci::test;

namespace
{

/**
 *  Graph with a single Op (example: Add).
 *
 *  BEFORE
 *  - All Ops including Input/Output are NCHW.
 *
 *             [Input] [beta]
 *                |  /
 *              [Add]
 *                |
 *             [Output]
 *
 *  AFTER
 *  - All Ops including Input/Output are NHWC.
 *
 *             [Input]
 *                |
 *         [Transpose]
 *                |
 *        [Transpose] [beta]
 *                |  /
 *              [Add]
 *                |
 *         [Transpose]
 *                |
 *         [Transpose]
 *                |
 *             [Output]
 */
class SimpleGraph
{
public:
  SimpleGraph() = default;

public:
  void init()
  {
    input = g.nodes()->create<luci::CircleInput>();
    output = g.nodes()->create<luci::CircleOutput>();
    input->name("input");
    output->name("output");

    auto graph_input = g.inputs()->create();
    input->index(graph_input->index());
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    graph_input->dtype(loco::DataType::FLOAT32);
    input->dtype(loco::DataType::FLOAT32);
    output->dtype(loco::DataType::FLOAT32);
    graph_output->dtype(loco::DataType::FLOAT32);

    uint32_t channel_size = 16;
    graph_input->shape({1, channel_size, 4, 4});
    input->shape({1, channel_size, 4, 4});
    output->shape({1, channel_size, 4, 4});
    graph_output->shape({1, channel_size, 4, 4});

    auto graph_body = insertGraphBody(input);
    output->from(graph_body);
  }

  virtual ~SimpleGraph() = default;

protected:
  virtual loco::Node *insertGraphBody(loco::Node *input) = 0;

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleOutput *output = nullptr;
};

class AddGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    add = g.nodes()->create<luci::CircleAdd>();
    beta = g.nodes()->create<luci::CircleConst>();

    add->dtype(loco::DataType::FLOAT32);
    beta->dtype(loco::DataType::FLOAT32);

    uint32_t channel_size = 16;
    add->shape({1, channel_size, 4, 4});
    beta->shape({1, channel_size, 1, 1});

    beta->size<loco::DataType::FLOAT32>(channel_size);
    for (uint32_t i = 0; i < channel_size; i++)
    {
      beta->at<loco::DataType::FLOAT32>(i) = i;
    }

    add->x(input);
    add->y(beta);

    add->name("add");
    beta->name("beta");

    return add;
  }

public:
  void update_const_shape_to_nchw(void)
  {
    uint32_t channel_size = 16;
    beta->shape({1, channel_size, 4, 4});

    beta->size<loco::DataType::FLOAT32>(channel_size * 4 * 4);
    for (uint32_t i = 0; i < channel_size; i++)
    {
      beta->at<loco::DataType::FLOAT32>(i) = i;
    }
  }

public:
  luci::CircleAdd *add = nullptr;
  luci::CircleConst *beta = nullptr;
};

class NHWCReluGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    relu = g.nodes()->create<luci::CircleRelu>();
    pre_reshape = g.nodes()->create<luci::CircleReshape>();
    post_reshape = g.nodes()->create<luci::CircleReshape>();
    pre_shape = g.nodes()->create<luci::CircleConst>();
    post_shape = g.nodes()->create<luci::CircleConst>();

    pre_shape->dtype(loco::DataType::S32);
    post_shape->dtype(loco::DataType::S32);

    uint32_t channel_size = 16;
    auto in = loco::must_cast<luci::CircleNode *>(input);
    in->shape({1, channel_size, 4, 4});
    pre_shape->shape({4});
    post_shape->shape({4});

    pre_shape->size<loco::DataType::S32>(4);
    pre_shape->at<loco::DataType::S32>(0) = 1;
    pre_shape->at<loco::DataType::S32>(1) = 4;
    pre_shape->at<loco::DataType::S32>(2) = 4;
    pre_shape->at<loco::DataType::S32>(3) = channel_size;

    post_shape->size<loco::DataType::S32>(4);
    post_shape->at<loco::DataType::S32>(0) = 1;
    post_shape->at<loco::DataType::S32>(1) = channel_size;
    post_shape->at<loco::DataType::S32>(2) = 4;
    post_shape->at<loco::DataType::S32>(3) = 4;

    pre_reshape->tensor(input);
    pre_reshape->shape(pre_shape);

    relu->features(pre_reshape);

    post_reshape->tensor(relu);
    post_reshape->shape(post_shape);

    relu->name("Relu");
    pre_reshape->name("pre-reshape");
    post_reshape->name("post-reshape");

    return post_reshape;
  }

public:
  luci::CircleRelu *relu = nullptr;
  luci::CircleReshape *pre_reshape = nullptr;
  luci::CircleReshape *post_reshape = nullptr;
  luci::CircleConst *pre_shape = nullptr;
  luci::CircleConst *post_shape = nullptr;
};

/**
 *  Graph with pre-Reshape but no post-Transpose/Reshape.
 *
 *  BEFORE
 *             [Input]
 *                |
 *          [Pre-Reshape]
 *                |
 *              [Relu]
 *                |
 *             [Output]
 *
 *  AFTER
 *             [Input]
 *                |
 *          [Pre-Reshape]
 *                |
 *          [Pre-Transpose]
 *                |
 *              [Relu]
 *                |
 *          [Post-Transpose]
 *                |
 *             [Output]
 */
class NoPostReshapeGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    relu = g.nodes()->create<luci::CircleRelu>();
    pre_reshape = g.nodes()->create<luci::CircleReshape>();
    pre_shape = g.nodes()->create<luci::CircleConst>();

    pre_shape->dtype(loco::DataType::S32);

    uint32_t channel_size = 16;
    auto in = loco::must_cast<luci::CircleNode *>(input);
    in->shape({1, channel_size, 4, 4});
    pre_shape->shape({4});

    pre_shape->size<loco::DataType::S32>(4);
    pre_shape->at<loco::DataType::S32>(0) = 1;
    pre_shape->at<loco::DataType::S32>(1) = 4;
    pre_shape->at<loco::DataType::S32>(2) = 4;
    pre_shape->at<loco::DataType::S32>(3) = channel_size;

    pre_reshape->tensor(input);
    pre_reshape->shape(pre_shape);
    relu->features(pre_reshape);

    relu->name("Relu");
    pre_reshape->name("pre-reshape");

    return relu;
  }

public:
  luci::CircleRelu *relu = nullptr;
  luci::CircleReshape *pre_reshape = nullptr;
  luci::CircleConst *pre_shape = nullptr;
};

/**
 *  Graph with two pre-Reshapes
 *
 *  BEFORE
 *             [Input]
 *                |
 *          [Pre-Reshape]
 *                |
 *              [Relu]
 *                |
 *          [Pre-Reshape]
 *                |
 *          [Post-Reshape]
 *                |
 *             [Output]
 *
 *  AFTER
 *             [Input]
 *                |
 *          [Pre-Reshape]
 *                |
 *          [Pre-Transpose]
 *                |
 *              [Relu]
 *                |
 *          [Post-Transpose]
 *                |
 *          [Pre-Reshape]
 *                |
 *          [Post-Reshape]
 *                |
 *             [Output]
 */
class ReluNotClosedGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    relu = g.nodes()->create<luci::CircleRelu>();
    pre_reshape = g.nodes()->create<luci::CircleReshape>();
    pre_reshape_2 = g.nodes()->create<luci::CircleReshape>();
    post_reshape = g.nodes()->create<luci::CircleReshape>();
    pre_shape = g.nodes()->create<luci::CircleConst>();
    pre_shape_2 = g.nodes()->create<luci::CircleConst>();
    post_shape = g.nodes()->create<luci::CircleConst>();

    pre_shape->dtype(loco::DataType::S32);
    pre_shape_2->dtype(loco::DataType::S32);
    post_shape->dtype(loco::DataType::S32);

    uint32_t channel_size = 16;
    auto in = loco::must_cast<luci::CircleNode *>(input);
    in->shape({1, channel_size, 4, 4});
    pre_shape->shape({4});
    pre_shape_2->shape({4});
    post_shape->shape({4});

    pre_shape->size<loco::DataType::S32>(4);
    pre_shape->at<loco::DataType::S32>(0) = 1;
    pre_shape->at<loco::DataType::S32>(1) = 4;
    pre_shape->at<loco::DataType::S32>(2) = 4;
    pre_shape->at<loco::DataType::S32>(3) = channel_size;

    pre_shape_2->size<loco::DataType::S32>(4);
    pre_shape_2->at<loco::DataType::S32>(0) = 1;
    pre_shape_2->at<loco::DataType::S32>(1) = 4;
    pre_shape_2->at<loco::DataType::S32>(2) = channel_size;
    pre_shape_2->at<loco::DataType::S32>(3) = 4;

    post_shape->size<loco::DataType::S32>(4);
    post_shape->at<loco::DataType::S32>(0) = 1;
    post_shape->at<loco::DataType::S32>(1) = 4;
    post_shape->at<loco::DataType::S32>(2) = 4;
    post_shape->at<loco::DataType::S32>(3) = channel_size;

    pre_reshape->tensor(input);
    pre_reshape->shape(pre_shape);

    relu->features(pre_reshape);

    pre_reshape_2->tensor(relu);
    pre_reshape_2->shape(pre_shape_2);

    post_reshape->tensor(pre_reshape_2);
    post_reshape->shape(post_shape);

    relu->name("Relu");
    pre_reshape->name("pre-reshape");
    pre_reshape->name("pre-reshape-2");
    post_reshape->name("post-reshape");

    return post_reshape;
  }

public:
  luci::CircleRelu *relu = nullptr;
  luci::CircleReshape *pre_reshape = nullptr;
  luci::CircleReshape *pre_reshape_2 = nullptr;
  luci::CircleReshape *post_reshape = nullptr;
  luci::CircleConst *pre_shape = nullptr;
  luci::CircleConst *pre_shape_2 = nullptr;
  luci::CircleConst *post_shape = nullptr;
};

class AddScalarGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    add = g.nodes()->create<luci::CircleAdd>();
    beta = g.nodes()->create<luci::CircleConst>();

    add->dtype(loco::DataType::FLOAT32);
    beta->dtype(loco::DataType::FLOAT32);

    uint32_t channel_size = 16;
    add->shape({1, channel_size, 4, 4});
    beta->shape({1});

    beta->size<loco::DataType::FLOAT32>(1);
    beta->at<loco::DataType::FLOAT32>(0) = 3.14;

    add->x(input);
    add->y(beta);

    add->name("add");
    beta->name("beta");

    return add;
  }

public:
  luci::CircleAdd *add = nullptr;
  luci::CircleConst *beta = nullptr;
};

class ConcatenationGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    concat = g.nodes()->create<luci::CircleConcatenation>(2);
    concat->values(0, input);
    concat->axis(1);

    input2 = g.nodes()->create<luci::CircleConst>();
    input2->dtype(loco::DataType::FLOAT32);
    input2->shape({1, 16, 4, 4});
    input2->size<loco::DataType::FLOAT32>(16 * 4 * 4);
    for (uint32_t i = 0; i < 16 * 4 * 4; i++)
    {
      input2->at<loco::DataType::FLOAT32>(i) = i;
    }
    concat->values(1, input2);

    concat->name("concat");
    input2->name("input2");

    return concat;
  }

public:
  luci::CircleConcatenation *concat = nullptr;
  luci::CircleConst *input2 = nullptr;
};

class EluGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    elu = g.nodes()->create<luci::CircleElu>();
    elu->features(input);
    elu->name("elu");

    return elu;
  }

public:
  luci::CircleElu *elu = nullptr;
};

class LeakyReluGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    leakyrelu = g.nodes()->create<luci::CircleLeakyRelu>();
    leakyrelu->features(input);
    leakyrelu->name("leakyrelu");

    return leakyrelu;
  }

public:
  luci::CircleLeakyRelu *leakyrelu = nullptr;
};

class LogisticGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    logistic = g.nodes()->create<luci::CircleLogistic>();
    logistic->x(input);
    logistic->name("logistic");

    return logistic;
  }

public:
  luci::CircleLogistic *logistic = nullptr;
};

class MaximumGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    max = g.nodes()->create<luci::CircleMaximum>();
    limit = g.nodes()->create<luci::CircleConst>();

    max->dtype(loco::DataType::FLOAT32);
    limit->dtype(loco::DataType::FLOAT32);

    max->shape({1, 16, 4, 4});
    limit->shape({});

    limit->size<loco::DataType::FLOAT32>(1);
    limit->at<loco::DataType::FLOAT32>(0) = 100;

    max->x(input);
    max->y(limit);

    max->name("max");
    limit->name("limit");

    return max;
  }

public:
  luci::CircleMaximum *max = nullptr;
  luci::CircleConst *limit = nullptr;
};

class MaximumNonConstGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    max = g.nodes()->create<luci::CircleMaximum>();
    max->dtype(loco::DataType::FLOAT32);
    max->shape({1, 16, 4, 4});

    max->x(input);
    max->y(input);

    max->name("max");

    return max;
  }

public:
  luci::CircleMaximum *max = nullptr;
};

static constexpr std::initializer_list<uint32_t> kDefaultShape = {1, 16, 1, 1};

class MeanGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    mean = g.nodes()->create<luci::CircleMean>();
    rindices = g.nodes()->create<luci::CircleConst>();

    mean->dtype(loco::DataType::FLOAT32);
    rindices->dtype(loco::DataType::S32);

    mean->shape(_shape);
    rindices->shape({static_cast<uint32_t>(_axes.size())});

    rindices->size<loco::DataType::S32>(_axes.size());
    for (uint32_t i = 0; i < _axes.size(); ++i)
    {
      rindices->at<loco::DataType::S32>(i) = _axes[i];
    }

    mean->input(input);
    mean->reduction_indices(rindices);
    mean->keep_dims(_keep_dims);

    mean->name("mean");
    rindices->name("rindices");

    return mean;
  }

public:
  void keep_dims(bool val) { _keep_dims = val; }
  void axes(std::vector<int32_t> val) { _axes = val; }
  void shape(std::initializer_list<uint32_t> val) { _shape = val; }

public:
  luci::CircleMean *mean = nullptr;
  luci::CircleConst *rindices = nullptr;

private:
  bool _keep_dims = true;
  std::vector<int32_t> _axes = {2, 3};
  std::initializer_list<uint32_t> _shape = kDefaultShape;
};

class MinimumGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    min = g.nodes()->create<luci::CircleMinimum>();
    limit = g.nodes()->create<luci::CircleConst>();

    min->dtype(loco::DataType::FLOAT32);
    limit->dtype(loco::DataType::FLOAT32);

    min->shape({1, 16, 4, 4});
    limit->shape({});

    limit->size<loco::DataType::FLOAT32>(1);
    limit->at<loco::DataType::FLOAT32>(0) = 100;

    min->x(input);
    min->y(limit);

    min->name("min");
    limit->name("limit");

    return min;
  }

public:
  luci::CircleMinimum *min = nullptr;
  luci::CircleConst *limit = nullptr;
};

class MulGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    mul = g.nodes()->create<luci::CircleMul>();
    multiplier = g.nodes()->create<luci::CircleConst>();

    mul->dtype(loco::DataType::FLOAT32);
    multiplier->dtype(loco::DataType::FLOAT32);

    uint32_t channel_size = 16;
    mul->shape({1, channel_size, 4, 4});
    multiplier->shape({1, channel_size, 1, 1});

    multiplier->size<loco::DataType::FLOAT32>(channel_size);
    for (uint32_t i = 0; i < channel_size; i++)
    {
      multiplier->at<loco::DataType::FLOAT32>(i) = i;
    }

    mul->x(input);
    mul->y(multiplier);

    mul->name("mul");
    multiplier->name("multiplier");

    return mul;
  }

public:
  void update_const_shape_to_nchw(void)
  {
    uint32_t channel_size = 16;
    multiplier->shape({1, channel_size, 4, 4});

    multiplier->size<loco::DataType::FLOAT32>(channel_size * 4 * 4);
    for (uint32_t i = 0; i < channel_size; i++)
    {
      multiplier->at<loco::DataType::FLOAT32>(i) = i;
    }
  }

public:
  luci::CircleMul *mul = nullptr;
  luci::CircleConst *multiplier = nullptr;
};

class MulScalarGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    mul = g.nodes()->create<luci::CircleMul>();
    multiplier = g.nodes()->create<luci::CircleConst>();

    mul->dtype(loco::DataType::FLOAT32);
    multiplier->dtype(loco::DataType::FLOAT32);

    uint32_t channel_size = 16;
    mul->shape({1, channel_size, 4, 4});
    multiplier->shape({1});

    multiplier->size<loco::DataType::FLOAT32>(1);
    multiplier->at<loco::DataType::FLOAT32>(0) = 2;

    mul->x(input);
    mul->y(multiplier);

    mul->name("mul");
    multiplier->name("multiplier");

    return mul;
  }

public:
  luci::CircleMul *mul = nullptr;
  luci::CircleConst *multiplier = nullptr;
};

class MulBothNormGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    mul = g.nodes()->create<luci::CircleMul>();

    mul->dtype(loco::DataType::FLOAT32);

    uint32_t channel_size = 16;
    mul->shape({1, channel_size, 4, 4});

    mul->x(input);
    mul->y(input);

    mul->name("mul");

    return mul;
  }

public:
  luci::CircleMul *mul = nullptr;
};

class NegGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    neg = g.nodes()->create<luci::CircleNeg>();
    neg->x(input);
    neg->name("neg");

    return neg;
  }

public:
  luci::CircleNeg *neg = nullptr;
};

class PadGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    pad = g.nodes()->create<luci::CirclePad>();
    paddings = g.nodes()->create<luci::CircleConst>();

    pad->dtype(loco::DataType::FLOAT32);
    paddings->dtype(loco::DataType::S32);

    uint32_t channel_size = 16;
    pad->shape({1, channel_size, 4, 4});
    paddings->shape({4, 2});

    // paddings data (NCHW)
    // [[0,0], [0,0], [1,1], [2,2]]
    paddings->size<loco::DataType::S32>(8);
    for (uint32_t dim = 0; dim < 4; dim++)
    {
      for (uint32_t i = 0; i < 2; i++)
      {
        int32_t data = 0;

        if (dim == 2)
          data = 1;
        else if (dim == 3)
          data = 2;

        paddings->at<loco::DataType::S32>(dim * 2 + i) = data;
      }
    }

    pad->input(input);
    pad->paddings(paddings);

    pad->name("pad");
    paddings->name("paddings");

    return pad;
  }

public:
  luci::CirclePad *pad = nullptr;
  luci::CircleConst *paddings = nullptr;
};

class PadV2Graph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    pad = g.nodes()->create<luci::CirclePadV2>();
    paddings = g.nodes()->create<luci::CircleConst>();
    const_value = g.nodes()->create<luci::CircleConst>();

    pad->dtype(loco::DataType::FLOAT32);
    paddings->dtype(loco::DataType::S32);
    const_value->dtype(loco::DataType::FLOAT32);

    uint32_t channel_size = 16;
    pad->shape({1, channel_size, 4, 4});
    paddings->shape({4, 2});
    const_value->shape({1});

    // paddings data (NCHW)
    // [[0,0], [0,0], [1,1], [2,2]]
    paddings->size<loco::DataType::S32>(8);
    for (uint32_t dim = 0; dim < 4; dim++)
    {
      for (uint32_t i = 0; i < 2; i++)
      {
        int32_t data = 0;

        if (dim == 2)
          data = 1;
        else if (dim == 3)
          data = 2;

        paddings->at<loco::DataType::S32>(dim * 2 + i) = data;
      }
    }

    const_value->size<loco::DataType::FLOAT32>(1);
    const_value->at<loco::DataType::FLOAT32>(0) = -3.4;

    pad->input(input);
    pad->paddings(paddings);
    pad->constant_values(paddings);

    pad->name("padV2");
    paddings->name("paddings");
    const_value->name("constant_values");

    return pad;
  }

public:
  luci::CirclePadV2 *pad = nullptr;
  luci::CircleConst *paddings = nullptr;
  luci::CircleConst *const_value = nullptr;
};

class ReduceMaxGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    rm = g.nodes()->create<luci::CircleReduceMax>();
    rindices = g.nodes()->create<luci::CircleConst>();

    rm->dtype(loco::DataType::FLOAT32);
    rindices->dtype(loco::DataType::S32);

    rm->shape(_shape);
    rindices->shape({static_cast<uint32_t>(_axes.size())});

    rindices->size<loco::DataType::S32>(_axes.size());
    for (uint32_t i = 0; i < _axes.size(); ++i)
    {
      rindices->at<loco::DataType::S32>(i) = _axes[i];
    }

    rm->input(input);
    rm->reduction_indices(rindices);
    rm->keep_dims(_keep_dims);

    rm->name("reduce_max");
    rindices->name("rindices");

    return rm;
  }

public:
  void keep_dims(bool val) { _keep_dims = val; }
  void axes(std::vector<int32_t> val) { _axes = val; }
  void shape(std::initializer_list<uint32_t> val) { _shape = val; }

public:
  luci::CircleReduceMax *rm = nullptr;
  luci::CircleConst *rindices = nullptr;

private:
  bool _keep_dims = true;
  std::vector<int32_t> _axes = {2, 3};
  std::initializer_list<uint32_t> _shape = kDefaultShape;
};

class ReduceMinGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    rm = g.nodes()->create<luci::CircleReduceMin>();
    rindices = g.nodes()->create<luci::CircleConst>();

    rm->dtype(loco::DataType::FLOAT32);
    rindices->dtype(loco::DataType::S32);

    rm->shape(_shape);
    rindices->shape({static_cast<uint32_t>(_axes.size())});

    rindices->size<loco::DataType::S32>(_axes.size());
    for (uint32_t i = 0; i < _axes.size(); ++i)
    {
      rindices->at<loco::DataType::S32>(i) = _axes[i];
    }

    rm->input(input);
    rm->reduction_indices(rindices);
    rm->keep_dims(_keep_dims);

    rm->name("reduce_max");
    rindices->name("rindices");

    return rm;
  }

public:
  void keep_dims(bool val) { _keep_dims = val; }
  void axes(std::vector<int32_t> val) { _axes = val; }
  void shape(std::initializer_list<uint32_t> val) { _shape = val; }

public:
  luci::CircleReduceMin *rm = nullptr;
  luci::CircleConst *rindices = nullptr;

private:
  bool _keep_dims = true;
  std::vector<int32_t> _axes = {2, 3};
  std::initializer_list<uint32_t> _shape = kDefaultShape;
};

class ReluGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    relu = g.nodes()->create<luci::CircleRelu>();
    relu->features(input);
    relu->name("Relu");

    return relu;
  }

public:
  luci::CircleRelu *relu = nullptr;
};

class Relu6Graph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    relu6 = g.nodes()->create<luci::CircleRelu6>();
    relu6->features(input);
    relu6->name("relu6");

    return relu6;
  }

public:
  luci::CircleRelu6 *relu6 = nullptr;
};

class RsqrtGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    rsqrt = g.nodes()->create<luci::CircleRsqrt>();
    rsqrt->x(input);
    rsqrt->name("rsqrt");

    return rsqrt;
  }

public:
  luci::CircleRsqrt *rsqrt = nullptr;
};

class SplitVGraphlet
{
public:
  SplitVGraphlet() = default;

public:
  void init(loco::Graph *g)
  {
    // CircleCustom(SplitV)
    _splitv = g->nodes()->create<luci::CircleSplitV>();
    _splitv->shape({1, 2, 2, 192});
    _splitv->dtype(loco::DataType::FLOAT32);
    _splitv->name("splitv");

    // CircleConst
    auto size_splits = g->nodes()->create<luci::CircleConst>();
    size_splits->dtype(loco::DataType::S32);
    size_splits->shape({3});
    size_splits->size<loco::DataType::S32>(3);
    size_splits->at<loco::DataType::S32>(0) = 32;
    size_splits->at<loco::DataType::S32>(1) = 32;
    size_splits->at<loco::DataType::S32>(2) = 128;

    // CircleConst
    auto split_dim = g->nodes()->create<luci::CircleConst>();
    split_dim->dtype(loco::DataType::S32);
    split_dim->rank(0);
    split_dim->size<loco::DataType::S32>(1);
    split_dim->scalar<loco::DataType::S32>() = 3;

    _splitv->size_splits(size_splits);
    _splitv->split_dim(split_dim);
    _splitv->num_split(3);

    // CircleSplitVOut
    _splitv_out1 = g->nodes()->create<luci::CircleSplitVOut>();
    _splitv_out1->shape({1, 2, 2, 32});
    _splitv_out1->dtype(loco::DataType::FLOAT32);
    _splitv_out1->index(0);
    _splitv_out1->input(_splitv);
    _splitv_out1->name("splitv_out1");

    // CircleSplitVOut
    _splitv_out2 = g->nodes()->create<luci::CircleSplitVOut>();
    _splitv_out2->shape({1, 2, 2, 32});
    _splitv_out2->dtype(loco::DataType::FLOAT32);
    _splitv_out2->index(1);
    _splitv_out2->input(_splitv);
    _splitv_out2->name("splitv_out2");

    // CircleSplitVOut
    _splitv_out3 = g->nodes()->create<luci::CircleSplitVOut>();
    _splitv_out3->shape({1, 2, 2, 128});
    _splitv_out3->dtype(loco::DataType::FLOAT32);
    _splitv_out3->index(2);
    _splitv_out3->input(_splitv);
    _splitv_out3->name("splitv_out3");
  }

public:
  luci::CircleSplitV *splitv() { return _splitv; }

protected:
  luci::CircleSplitV *_splitv = nullptr;
  luci::CircleSplitVOut *_splitv_out1 = nullptr;
  luci::CircleSplitVOut *_splitv_out2 = nullptr;
  luci::CircleSplitVOut *_splitv_out3 = nullptr;
};

class SplitVGraph : public TestIGraphlet, public TestOsGraphlet<3>, public SplitVGraphlet
{
public:
  SplitVGraph() = default;

  void init(void)
  {
    TestIGraphlet::init(g(), {1, 2, 2, 192});
    TestOsGraphlet<3>::init(g(), {{1, 2, 2, 32}, {1, 2, 2, 32}, {1, 2, 2, 128}});
    SplitVGraphlet::init(g());

    // connect graph
    _splitv->input(input());

    output(0)->from(_splitv_out1);
    output(1)->from(_splitv_out2);
    output(2)->from(_splitv_out3);
  }
};

class SquaredDifferenceGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    sqdiff = g.nodes()->create<luci::CircleSquaredDifference>();
    sqdiff->x(input);
    sqdiff->y(input);
    sqdiff->name("sqdiff");

    return sqdiff;
  }

public:
  luci::CircleSquaredDifference *sqdiff = nullptr;
};

class SubGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    sub = g.nodes()->create<luci::CircleSub>();
    beta = g.nodes()->create<luci::CircleConst>();

    sub->dtype(loco::DataType::FLOAT32);
    beta->dtype(loco::DataType::FLOAT32);

    uint32_t channel_size = 16;
    sub->shape({1, channel_size, 4, 4});
    beta->shape({1, channel_size, 1, 1});

    beta->size<loco::DataType::FLOAT32>(channel_size);
    for (uint32_t i = 0; i < channel_size; i++)
    {
      beta->at<loco::DataType::FLOAT32>(i) = i;
    }

    sub->x(input);
    sub->y(beta);

    sub->name("sub");
    beta->name("beta");

    return sub;
  }

public:
  void update_const_shape_to_nchw(void)
  {
    uint32_t channel_size = 16;
    beta->shape({1, channel_size, 4, 4});

    beta->size<loco::DataType::FLOAT32>(channel_size * 4 * 4);
    for (uint32_t i = 0; i < channel_size; i++)
    {
      beta->at<loco::DataType::FLOAT32>(i) = i;
    }
  }

public:
  luci::CircleSub *sub = nullptr;
  luci::CircleConst *beta = nullptr;
};

class SubScalarGraph final : public SimpleGraph
{
protected:
  loco::Node *insertGraphBody(loco::Node *input) override
  {
    sub = g.nodes()->create<luci::CircleSub>();
    beta = g.nodes()->create<luci::CircleConst>();

    sub->dtype(loco::DataType::FLOAT32);
    beta->dtype(loco::DataType::FLOAT32);

    uint32_t channel_size = 16;
    sub->shape({1, channel_size, 4, 4});
    beta->shape({1});

    beta->size<loco::DataType::FLOAT32>(1);
    beta->at<loco::DataType::FLOAT32>(0) = 5;

    sub->x(beta);
    sub->y(input);

    sub->name("sub");
    beta->name("beta");

    return sub;
  }

public:
  luci::CircleSub *sub = nullptr;
  luci::CircleConst *beta = nullptr;
};

void check_pre_trans(loco::Node *node)
{
  auto pre_trans = dynamic_cast<luci::CircleTranspose *>(node);
  EXPECT_NE(nullptr, pre_trans);
  auto pre_trans_perm = dynamic_cast<luci::CircleConst *>(pre_trans->perm());
  EXPECT_NE(nullptr, pre_trans_perm);
  EXPECT_EQ(1, pre_trans_perm->rank());
  EXPECT_EQ(4, pre_trans_perm->dim(0).value());
  EXPECT_EQ(loco::DataType::S32, pre_trans_perm->dtype());
  EXPECT_EQ(0, pre_trans_perm->at<loco::DataType::S32>(0));
  EXPECT_EQ(2, pre_trans_perm->at<loco::DataType::S32>(1));
  EXPECT_EQ(3, pre_trans_perm->at<loco::DataType::S32>(2));
  EXPECT_EQ(1, pre_trans_perm->at<loco::DataType::S32>(3));
}

void check_post_trans(loco::Node *node)
{
  auto post_trans = dynamic_cast<luci::CircleTranspose *>(node);
  EXPECT_NE(nullptr, post_trans);
  auto post_trans_perm = dynamic_cast<luci::CircleConst *>(post_trans->perm());
  EXPECT_NE(nullptr, post_trans_perm);
  EXPECT_EQ(1, post_trans_perm->rank());
  EXPECT_EQ(4, post_trans_perm->dim(0).value());
  EXPECT_EQ(loco::DataType::S32, post_trans_perm->dtype());
  EXPECT_EQ(0, post_trans_perm->at<loco::DataType::S32>(0));
  EXPECT_EQ(3, post_trans_perm->at<loco::DataType::S32>(1));
  EXPECT_EQ(1, post_trans_perm->at<loco::DataType::S32>(2));
  EXPECT_EQ(2, post_trans_perm->at<loco::DataType::S32>(3));
}

void run_phase(loco::Graph *g, bool preserve_input, bool preserve_output)
{
  logo::Phase phase;

  // Default passes.
  phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());

  // Pass to test
  phase.emplace_back(
    std::make_unique<luci::ConvertNCHWToNHWCPass>(preserve_input, preserve_output));

  logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{g};
  phase_runner.run(phase);
}

} // namespace

TEST(ConvertNCHWToNHWCPassTest, name)
{
  luci::ConvertNCHWToNHWCPass pass(false, false);
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(ConvertNCHWToNHWC, Add)
{
  AddGraph g;
  g.init();

  run_phase(&g.g, false, false);

  auto input_succs = loco::succs(g.input);
  EXPECT_EQ(1, input_succs.size());
  check_post_trans(*input_succs.begin());

  check_pre_trans(g.add->x());

  auto add_succs = loco::succs(g.add);
  EXPECT_EQ(1, add_succs.size());
  check_post_trans(*add_succs.begin());

  uint32_t channel_size = 16;
  auto new_beta = dynamic_cast<luci::CircleConst *>(g.add->y());
  EXPECT_NE(nullptr, new_beta);
  EXPECT_EQ(4, new_beta->rank());
  EXPECT_EQ(1, new_beta->dim(0).value());
  EXPECT_EQ(1, new_beta->dim(1).value());
  EXPECT_EQ(1, new_beta->dim(2).value());
  EXPECT_EQ(channel_size, new_beta->dim(3).value());

  check_pre_trans(g.output->from());
}

TEST(ConvertNCHWToNHWC, Add_NCHW_const)
{
  AddGraph g;
  g.init();
  g.update_const_shape_to_nchw();

  run_phase(&g.g, false, false);

  check_pre_trans(g.add->x());

  auto add_succs = loco::succs(g.add);
  EXPECT_EQ(1, add_succs.size());
  check_post_trans(*add_succs.begin());

  uint32_t channel_size = 16;
  auto new_beta = dynamic_cast<luci::CircleConst *>(g.add->y());
  EXPECT_NE(nullptr, new_beta);
  EXPECT_EQ(4, new_beta->rank());
  EXPECT_EQ(1, new_beta->dim(0).value());
  EXPECT_EQ(4, new_beta->dim(1).value());
  EXPECT_EQ(4, new_beta->dim(2).value());
  EXPECT_EQ(channel_size, new_beta->dim(3).value());
}

TEST(ConvertNCHWToNHWC, NHWC_Relu)
{
  // Relu is already NHWC, so it should not be converted
  // i.e., the graph is not changed
  NHWCReluGraph g;
  g.init();

  run_phase(&g.g, false, false);

  EXPECT_EQ(g.pre_reshape, g.relu->features());

  auto relu_succs = loco::succs(g.relu);
  EXPECT_EQ(1, relu_succs.size());
  EXPECT_EQ(g.post_reshape, *relu_succs.begin());
}

TEST(ConvertNCHWToNHWC, AddScalar)
{
  AddScalarGraph g;
  g.init();

  run_phase(&g.g, false, false);

  auto input_succs = loco::succs(g.input);
  EXPECT_EQ(1, input_succs.size());
  check_post_trans(*input_succs.begin());

  check_pre_trans(g.add->x());

  auto add_succs = loco::succs(g.add);
  EXPECT_EQ(1, add_succs.size());
  check_post_trans(*add_succs.begin());

  auto new_beta = dynamic_cast<luci::CircleConst *>(g.add->y());
  EXPECT_NE(nullptr, new_beta);
  EXPECT_EQ(4, new_beta->rank());
  EXPECT_EQ(1, new_beta->dim(0).value());
  EXPECT_EQ(1, new_beta->dim(1).value());
  EXPECT_EQ(1, new_beta->dim(2).value());
  EXPECT_EQ(1, new_beta->dim(3).value());

  check_pre_trans(g.output->from());
}

TEST(ConvertNCHWToNHWC, Concatenation)
{
  ConcatenationGraph g;
  g.init();

  run_phase(&g.g, true, true);

  check_pre_trans(g.concat->values(0));
  check_pre_trans(g.concat->values(1));

  auto concat_succs = loco::succs(g.concat);
  EXPECT_EQ(1, concat_succs.size());
  check_post_trans(*concat_succs.begin());

  // Check concat shape, axis
  EXPECT_EQ(1, g.concat->dim(0).value());
  EXPECT_EQ(4, g.concat->dim(1).value());
  EXPECT_EQ(4, g.concat->dim(2).value());
  EXPECT_EQ(32, g.concat->dim(3).value());
  EXPECT_EQ(3, g.concat->axis());
}

TEST(ConvertNCHWToNHWC, Elu)
{
  EluGraph g;
  g.init();

  run_phase(&g.g, true, true);

  check_pre_trans(g.elu->features());

  auto elu_succs = loco::succs(g.elu);
  EXPECT_EQ(1, elu_succs.size());
  check_post_trans(*elu_succs.begin());

  // Check elu shape
  EXPECT_EQ(1, g.elu->dim(0).value());
  EXPECT_EQ(4, g.elu->dim(1).value());
  EXPECT_EQ(4, g.elu->dim(2).value());
  EXPECT_EQ(16, g.elu->dim(3).value());
}

TEST(ConvertNCHWToNHWC, LeakyRelu)
{
  LeakyReluGraph g;
  g.init();

  run_phase(&g.g, true, true);

  check_pre_trans(g.leakyrelu->features());

  auto leakyrelu_succs = loco::succs(g.leakyrelu);
  EXPECT_EQ(1, leakyrelu_succs.size());
  check_post_trans(*leakyrelu_succs.begin());

  // Check leakyrelu shape
  EXPECT_EQ(1, g.leakyrelu->dim(0).value());
  EXPECT_EQ(4, g.leakyrelu->dim(1).value());
  EXPECT_EQ(4, g.leakyrelu->dim(2).value());
  EXPECT_EQ(16, g.leakyrelu->dim(3).value());
}

TEST(ConvertNCHWToNHWC, Logistic)
{
  LogisticGraph g;
  g.init();

  run_phase(&g.g, true, true);

  check_pre_trans(g.logistic->x());

  auto logistic_succs = loco::succs(g.logistic);
  EXPECT_EQ(1, logistic_succs.size());
  check_post_trans(*logistic_succs.begin());

  // Check logistic shape
  EXPECT_EQ(1, g.logistic->dim(0).value());
  EXPECT_EQ(4, g.logistic->dim(1).value());
  EXPECT_EQ(4, g.logistic->dim(2).value());
  EXPECT_EQ(16, g.logistic->dim(3).value());
}

TEST(ConvertNCHWToNHWC, Maximum)
{
  MaximumGraph g;
  g.init();

  run_phase(&g.g, false, false);

  auto input_succs = loco::succs(g.input);
  EXPECT_EQ(1, input_succs.size());
  check_post_trans(*input_succs.begin());

  check_pre_trans(g.max->x());

  auto max_succs = loco::succs(g.max);
  EXPECT_EQ(1, max_succs.size());
  check_post_trans(*max_succs.begin());

  check_pre_trans(g.output->from());
}

TEST(ConvertNCHWToNHWC, Maximum_non_scalar_NEG)
{
  MaximumGraph g;
  g.init();

  g.limit->shape({3});

  luci::ConvertNCHWToNHWCPass pass(true, true);
  EXPECT_FALSE(pass.run(&g.g));
}

TEST(ConvertNCHWToNHWC, MaximumNonConst)
{
  MaximumNonConstGraph g;
  g.init();

  run_phase(&g.g, true, true);

  check_pre_trans(g.max->x());
  check_pre_trans(g.max->y());

  auto max_succs = loco::succs(g.max);
  EXPECT_EQ(1, max_succs.size());
  check_post_trans(*max_succs.begin());
}

TEST(ConvertNCHWToNHWC, Mean)
{
  MeanGraph g;
  g.init();

  run_phase(&g.g, false, false);

  check_pre_trans(g.mean->input());

  auto mean_succs = loco::succs(g.mean);
  EXPECT_EQ(1, mean_succs.size());
  check_post_trans(*mean_succs.begin());

  auto new_rindices = dynamic_cast<luci::CircleConst *>(g.mean->reduction_indices());
  EXPECT_NE(nullptr, new_rindices);
  EXPECT_EQ(1, new_rindices->rank());
  EXPECT_EQ(2, new_rindices->dim(0).value());
  EXPECT_EQ(2, new_rindices->size<loco::DataType::S32>());
  EXPECT_EQ(1, new_rindices->at<loco::DataType::S32>(0));
  EXPECT_EQ(2, new_rindices->at<loco::DataType::S32>(1));
}

TEST(ConvertNCHWToNHWC, Mean_keep_dims_false)
{
  struct TC
  {
    std::vector<int32_t> nchw_ind;
    std::vector<int32_t> nhwc_ind;
    std::initializer_list<uint32_t> shape;
    bool needs_transpose = false;
  };

  uint32_t n = 1;
  uint32_t c = 16;
  uint32_t h = 4;
  uint32_t w = 4;

  std::vector<TC> test_cases{{{0}, {0}, {c, h, w}, true},       {{1}, {3}, {n, h, w}, false},
                             {{2}, {1}, {n, c, w}, true},       {{3}, {2}, {n, c, h}, true},
                             {{0, 1}, {0, 3}, {h, w}, false},   {{0, 2}, {0, 1}, {c, w}, true},
                             {{0, 3}, {0, 2}, {c, h}, true},    {{1, 2}, {3, 1}, {n, w}, false},
                             {{1, 3}, {3, 2}, {n, h}, false},   {{2, 3}, {1, 2}, {n, c}, false},
                             {{0, 1, 2}, {0, 3, 1}, {w}, false}};

  for (auto &tc : test_cases)
  {
    MeanGraph g;
    g.keep_dims(false);
    g.axes(tc.nchw_ind);
    g.shape(tc.shape);
    g.init();

    run_phase(&g.g, false, true);

    check_pre_trans(g.mean->input());

    auto mean_succs = loco::succs(g.mean);
    EXPECT_EQ(1, mean_succs.size());
    if (tc.needs_transpose)
    {
      EXPECT_NE(nullptr, dynamic_cast<luci::CircleTranspose *>(*mean_succs.begin()));
    }
    else
    {
      EXPECT_NE(nullptr, dynamic_cast<luci::CircleOutput *>(*mean_succs.begin()));
    }

    auto new_rindices = dynamic_cast<luci::CircleConst *>(g.mean->reduction_indices());
    EXPECT_NE(nullptr, new_rindices);
    EXPECT_EQ(1, new_rindices->rank());
    EXPECT_EQ(tc.nhwc_ind.size(), new_rindices->dim(0).value());
    EXPECT_EQ(tc.nhwc_ind.size(), new_rindices->size<loco::DataType::S32>());
    for (uint32_t i = 0; i < tc.nhwc_ind.size(); ++i)
    {
      EXPECT_EQ(tc.nhwc_ind[i], new_rindices->at<loco::DataType::S32>(i));
    }
  }
}

TEST(ConvertNCHWToNHWC, ConvertNCHWToNHWC_Mean_keep_dims_false_NEG)
{
  loco::Graph g;
  auto input = g.nodes()->create<luci::CircleInput>();
  auto output = g.nodes()->create<luci::CircleOutput>();
  input->name("input");
  output->name("output");

  auto graph_input = g.inputs()->create();
  input->index(graph_input->index());
  auto graph_output = g.outputs()->create();
  output->index(graph_output->index());

  graph_input->dtype(loco::DataType::FLOAT32);
  input->dtype(loco::DataType::FLOAT32);
  output->dtype(loco::DataType::FLOAT32);
  graph_output->dtype(loco::DataType::FLOAT32);

  uint32_t channel_size = 16;
  graph_input->shape({channel_size, 4, 4});
  input->shape({channel_size, 4, 4});
  output->shape({channel_size});
  graph_output->shape({channel_size});

  auto mean = g.nodes()->create<luci::CircleMean>();
  auto rindices = g.nodes()->create<luci::CircleConst>();

  mean->dtype(loco::DataType::FLOAT32);
  rindices->dtype(loco::DataType::S32);

  mean->shape({channel_size});
  rindices->shape({2});

  rindices->size<loco::DataType::S32>(2);
  rindices->at<loco::DataType::S32>(0) = 1;
  rindices->at<loco::DataType::S32>(1) = 2;

  mean->input(input);
  mean->reduction_indices(rindices);
  mean->keep_dims(false);

  mean->name("mean");
  rindices->name("rindices");

  output->from(mean);

  run_phase(&g, true, true);

  auto new_rindices = dynamic_cast<luci::CircleConst *>(mean->reduction_indices());
  EXPECT_NE(nullptr, new_rindices);
  EXPECT_EQ(1, new_rindices->rank());
  EXPECT_EQ(2, new_rindices->dim(0).value());
  EXPECT_EQ(2, new_rindices->size<loco::DataType::S32>());
  EXPECT_EQ(1, new_rindices->at<loco::DataType::S32>(0));
  EXPECT_EQ(2, new_rindices->at<loco::DataType::S32>(1));
}

TEST(ConvertNCHWToNHWC, Minimum)
{
  MinimumGraph g;
  g.init();

  run_phase(&g.g, false, false);

  auto input_succs = loco::succs(g.input);
  EXPECT_EQ(1, input_succs.size());
  check_post_trans(*input_succs.begin());

  check_pre_trans(g.min->x());

  auto min_succs = loco::succs(g.min);
  EXPECT_EQ(1, min_succs.size());
  check_post_trans(*min_succs.begin());

  check_pre_trans(g.output->from());
}

TEST(ConvertNCHWToNHWC, Minimum_non_scalar_NEG)
{
  MinimumGraph g;
  g.init();

  g.limit->shape({3});

  luci::ConvertNCHWToNHWCPass pass(true, true);
  EXPECT_FALSE(pass.run(&g.g));
}

TEST(ConvertNCHWToNHWC, Mul)
{
  MulGraph g;
  g.init();

  run_phase(&g.g, false, false);

  auto input_succs = loco::succs(g.input);
  EXPECT_EQ(1, input_succs.size());
  check_post_trans(*input_succs.begin());

  check_pre_trans(g.mul->x());

  auto mul_succs = loco::succs(g.mul);
  EXPECT_EQ(1, mul_succs.size());
  check_post_trans(*mul_succs.begin());

  uint32_t channel_size = 16;
  auto new_multiplier = dynamic_cast<luci::CircleConst *>(g.mul->y());
  EXPECT_NE(nullptr, new_multiplier);
  EXPECT_EQ(4, new_multiplier->rank());
  EXPECT_EQ(1, new_multiplier->dim(0).value());
  EXPECT_EQ(1, new_multiplier->dim(1).value());
  EXPECT_EQ(1, new_multiplier->dim(2).value());
  EXPECT_EQ(channel_size, new_multiplier->dim(3).value());

  check_pre_trans(g.output->from());
}

TEST(ConvertNCHWToNHWC, Mul_NCHW_const)
{
  MulGraph g;
  g.init();
  g.update_const_shape_to_nchw();

  run_phase(&g.g, false, false);

  check_pre_trans(g.mul->x());

  auto mul_succs = loco::succs(g.mul);
  EXPECT_EQ(1, mul_succs.size());
  check_post_trans(*mul_succs.begin());

  uint32_t channel_size = 16;
  auto new_multiplier = dynamic_cast<luci::CircleConst *>(g.mul->y());
  EXPECT_NE(nullptr, new_multiplier);
  EXPECT_EQ(4, new_multiplier->rank());
  EXPECT_EQ(1, new_multiplier->dim(0).value());
  EXPECT_EQ(4, new_multiplier->dim(1).value());
  EXPECT_EQ(4, new_multiplier->dim(2).value());
  EXPECT_EQ(channel_size, new_multiplier->dim(3).value());
}

TEST(ConvertNCHWToNHWC, MulScalar)
{
  MulScalarGraph g;
  g.init();

  run_phase(&g.g, false, false);

  auto input_succs = loco::succs(g.input);
  EXPECT_EQ(1, input_succs.size());
  check_post_trans(*input_succs.begin());

  check_pre_trans(g.mul->x());

  auto mul_succs = loco::succs(g.mul);
  EXPECT_EQ(1, mul_succs.size());
  check_post_trans(*mul_succs.begin());

  auto new_multiplier = dynamic_cast<luci::CircleConst *>(g.mul->y());
  EXPECT_NE(nullptr, new_multiplier);
  EXPECT_EQ(4, new_multiplier->rank());
  EXPECT_EQ(1, new_multiplier->dim(0).value());
  EXPECT_EQ(1, new_multiplier->dim(1).value());
  EXPECT_EQ(1, new_multiplier->dim(2).value());
  EXPECT_EQ(1, new_multiplier->dim(3).value());

  check_pre_trans(g.output->from());
}

TEST(ConvertNCHWToNHWC, MulBothNorm)
{
  MulBothNormGraph g;
  g.init();

  run_phase(&g.g, false, false);

  auto input_succs = loco::succs(g.input);
  EXPECT_EQ(1, input_succs.size());
  check_post_trans(*input_succs.begin());

  check_pre_trans(g.mul->x());
  check_pre_trans(g.mul->y());

  auto mul_succs = loco::succs(g.mul);
  EXPECT_EQ(1, mul_succs.size());
  check_post_trans(*mul_succs.begin());

  check_pre_trans(g.output->from());
}

TEST(ConvertNCHWToNHWC, Neg)
{
  NegGraph g;
  g.init();

  run_phase(&g.g, true, true);

  check_pre_trans(g.neg->x());

  auto neg_succs = loco::succs(g.neg);
  EXPECT_EQ(1, neg_succs.size());
  check_post_trans(*neg_succs.begin());

  // Check leakyrelu shape
  EXPECT_EQ(1, g.neg->dim(0).value());
  EXPECT_EQ(4, g.neg->dim(1).value());
  EXPECT_EQ(4, g.neg->dim(2).value());
  EXPECT_EQ(16, g.neg->dim(3).value());
}

TEST(ConvertNCHWToNHWC, Pad)
{
  PadGraph g;
  g.init();

  run_phase(&g.g, false, false);

  auto input_succs = loco::succs(g.input);
  EXPECT_EQ(1, input_succs.size());
  check_post_trans(*input_succs.begin());

  check_pre_trans(g.pad->input());

  auto pad_succs = loco::succs(g.pad);
  EXPECT_EQ(1, pad_succs.size());
  check_post_trans(*pad_succs.begin());

  auto new_paddings = dynamic_cast<luci::CircleConst *>(g.pad->paddings());
  EXPECT_NE(nullptr, new_paddings);
  EXPECT_EQ(2, new_paddings->rank());
  EXPECT_EQ(4, new_paddings->dim(0).value());
  EXPECT_EQ(2, new_paddings->dim(1).value());
  EXPECT_EQ(0, new_paddings->at<loco::DataType::S32>(0));
  EXPECT_EQ(0, new_paddings->at<loco::DataType::S32>(1));
  EXPECT_EQ(1, new_paddings->at<loco::DataType::S32>(2));
  EXPECT_EQ(1, new_paddings->at<loco::DataType::S32>(3));
  EXPECT_EQ(2, new_paddings->at<loco::DataType::S32>(4));
  EXPECT_EQ(2, new_paddings->at<loco::DataType::S32>(5));
  EXPECT_EQ(0, new_paddings->at<loco::DataType::S32>(6));
  EXPECT_EQ(0, new_paddings->at<loco::DataType::S32>(7));

  check_pre_trans(g.output->from());
}

TEST(ConvertNCHWToNHWC, PadV2)
{
  PadV2Graph g;
  g.init();

  run_phase(&g.g, false, false);

  check_pre_trans(g.pad->input());

  auto pad_succs = loco::succs(g.pad);
  EXPECT_EQ(1, pad_succs.size());
  check_post_trans(*pad_succs.begin());

  auto new_paddings = dynamic_cast<luci::CircleConst *>(g.pad->paddings());
  EXPECT_NE(nullptr, new_paddings);
  EXPECT_EQ(2, new_paddings->rank());
  EXPECT_EQ(4, new_paddings->dim(0).value());
  EXPECT_EQ(2, new_paddings->dim(1).value());
  EXPECT_EQ(0, new_paddings->at<loco::DataType::S32>(0));
  EXPECT_EQ(0, new_paddings->at<loco::DataType::S32>(1));
  EXPECT_EQ(1, new_paddings->at<loco::DataType::S32>(2));
  EXPECT_EQ(1, new_paddings->at<loco::DataType::S32>(3));
  EXPECT_EQ(2, new_paddings->at<loco::DataType::S32>(4));
  EXPECT_EQ(2, new_paddings->at<loco::DataType::S32>(5));
  EXPECT_EQ(0, new_paddings->at<loco::DataType::S32>(6));
  EXPECT_EQ(0, new_paddings->at<loco::DataType::S32>(7));
}

TEST(ConvertNCHWToNHWC, Unknown_Shape_NEG)
{
  AddGraph g;
  g.init();

  // Unknown shape
  g.input->dim(0).unset();
  g.add->dim(0).unset();
  g.output->dim(0).unset();

  luci::ConvertNCHWToNHWCPass pass(false, false);
  EXPECT_EQ(false, pass.run(&g.g));
}

TEST(ConvertNCHWToNHWC, Preserve_Input_Output)
{
  // Preserve input
  {
    AddGraph g;
    g.init();

    run_phase(&g.g, true, false);

    // Check input shape
    EXPECT_EQ(1, g.input->dim(0).value());
    EXPECT_EQ(16, g.input->dim(1).value());
    EXPECT_EQ(4, g.input->dim(2).value());
    EXPECT_EQ(4, g.input->dim(3).value());

    // Check output shape
    EXPECT_EQ(1, g.output->dim(0).value());
    EXPECT_EQ(4, g.output->dim(1).value());
    EXPECT_EQ(4, g.output->dim(2).value());
    EXPECT_EQ(16, g.output->dim(3).value());
  }

  // Preserve output
  {
    AddGraph g;
    g.init();

    run_phase(&g.g, false, true);

    // Check input shape
    EXPECT_EQ(1, g.input->dim(0).value());
    EXPECT_EQ(4, g.input->dim(1).value());
    EXPECT_EQ(4, g.input->dim(2).value());
    EXPECT_EQ(16, g.input->dim(3).value());

    // Check output shape
    EXPECT_EQ(1, g.output->dim(0).value());
    EXPECT_EQ(16, g.output->dim(1).value());
    EXPECT_EQ(4, g.output->dim(2).value());
    EXPECT_EQ(4, g.output->dim(3).value());
  }

  // Preserve both input and output
  {
    AddGraph g;
    g.init();

    run_phase(&g.g, true, true);

    // Check input shape
    EXPECT_EQ(1, g.input->dim(0).value());
    EXPECT_EQ(16, g.input->dim(1).value());
    EXPECT_EQ(4, g.input->dim(2).value());
    EXPECT_EQ(4, g.input->dim(3).value());

    // Check output shape
    EXPECT_EQ(1, g.output->dim(0).value());
    EXPECT_EQ(16, g.output->dim(1).value());
    EXPECT_EQ(4, g.output->dim(2).value());
    EXPECT_EQ(4, g.output->dim(3).value());
  }
}

TEST(ConvertNCHWToNHWC, ReduceMax)
{
  ReduceMaxGraph g;
  g.init();

  run_phase(&g.g, false, false);

  check_pre_trans(g.rm->input());

  auto rm_succs = loco::succs(g.rm);
  EXPECT_EQ(1, rm_succs.size());
  check_post_trans(*rm_succs.begin());

  auto new_rindices = dynamic_cast<luci::CircleConst *>(g.rm->reduction_indices());
  EXPECT_NE(nullptr, new_rindices);
  EXPECT_EQ(1, new_rindices->rank());
  EXPECT_EQ(2, new_rindices->dim(0).value());
  EXPECT_EQ(2, new_rindices->size<loco::DataType::S32>());
  EXPECT_EQ(1, new_rindices->at<loco::DataType::S32>(0));
  EXPECT_EQ(2, new_rindices->at<loco::DataType::S32>(1));
}

TEST(ConvertNCHWToNHWC, ReduceMax_keep_dims_false)
{
  struct TC
  {
    std::vector<int32_t> nchw_ind;
    std::vector<int32_t> nhwc_ind;
    std::initializer_list<uint32_t> shape;
    bool needs_transpose = false;
  };

  uint32_t n = 1;
  uint32_t c = 16;
  uint32_t h = 4;
  uint32_t w = 4;

  std::vector<TC> test_cases{{{0}, {0}, {c, h, w}, true},       {{1}, {3}, {n, h, w}, false},
                             {{2}, {1}, {n, c, w}, true},       {{3}, {2}, {n, c, h}, true},
                             {{0, 1}, {0, 3}, {h, w}, false},   {{0, 2}, {0, 1}, {c, w}, true},
                             {{0, 3}, {0, 2}, {c, h}, true},    {{1, 2}, {3, 1}, {n, w}, false},
                             {{1, 3}, {3, 2}, {n, h}, false},   {{2, 3}, {1, 2}, {n, c}, false},
                             {{0, 1, 2}, {0, 3, 1}, {w}, false}};

  for (auto &tc : test_cases)
  {
    ReduceMaxGraph g;
    g.keep_dims(false);
    g.axes(tc.nchw_ind);
    g.shape(tc.shape);
    g.init();

    run_phase(&g.g, true, true);

    check_pre_trans(g.rm->input());

    auto rm_succs = loco::succs(g.rm);
    EXPECT_EQ(1, rm_succs.size());
    if (tc.needs_transpose)
    {
      EXPECT_NE(nullptr, dynamic_cast<luci::CircleTranspose *>(*rm_succs.begin()));
    }
    else
    {
      EXPECT_NE(nullptr, dynamic_cast<luci::CircleOutput *>(*rm_succs.begin()));
    }

    auto new_rindices = dynamic_cast<luci::CircleConst *>(g.rm->reduction_indices());
    EXPECT_NE(nullptr, new_rindices);
    EXPECT_EQ(1, new_rindices->rank());
    EXPECT_EQ(tc.nhwc_ind.size(), new_rindices->dim(0).value());
    EXPECT_EQ(tc.nhwc_ind.size(), new_rindices->size<loco::DataType::S32>());
    for (uint32_t i = 0; i < tc.nhwc_ind.size(); ++i)
    {
      EXPECT_EQ(tc.nhwc_ind[i], new_rindices->at<loco::DataType::S32>(i));
    }
  }
}

TEST(ConvertNCHWToNHWC, ReduceMin)
{
  ReduceMinGraph g;
  g.init();

  run_phase(&g.g, true, true);

  check_pre_trans(g.rm->input());

  auto rm_succs = loco::succs(g.rm);
  EXPECT_EQ(1, rm_succs.size());
  check_post_trans(*rm_succs.begin());

  auto new_rindices = dynamic_cast<luci::CircleConst *>(g.rm->reduction_indices());
  EXPECT_NE(nullptr, new_rindices);
  EXPECT_EQ(1, new_rindices->rank());
  EXPECT_EQ(2, new_rindices->dim(0).value());
  EXPECT_EQ(2, new_rindices->size<loco::DataType::S32>());
  EXPECT_EQ(1, new_rindices->at<loco::DataType::S32>(0));
  EXPECT_EQ(2, new_rindices->at<loco::DataType::S32>(1));
}

TEST(ConvertNCHWToNHWC, ReduceMin_keep_dims_false)
{
  struct TC
  {
    std::vector<int32_t> nchw_ind;
    std::vector<int32_t> nhwc_ind;
    std::initializer_list<uint32_t> shape;
    bool needs_transpose = false;
  };

  uint32_t n = 1;
  uint32_t c = 16;
  uint32_t h = 4;
  uint32_t w = 4;

  std::vector<TC> test_cases{{{0}, {0}, {c, h, w}, true},       {{1}, {3}, {n, h, w}, false},
                             {{2}, {1}, {n, c, w}, true},       {{3}, {2}, {n, c, h}, true},
                             {{0, 1}, {0, 3}, {h, w}, false},   {{0, 2}, {0, 1}, {c, w}, true},
                             {{0, 3}, {0, 2}, {c, h}, true},    {{1, 2}, {3, 1}, {n, w}, false},
                             {{1, 3}, {3, 2}, {n, h}, false},   {{2, 3}, {1, 2}, {n, c}, false},
                             {{0, 1, 2}, {0, 3, 1}, {w}, false}};

  for (auto &tc : test_cases)
  {
    ReduceMinGraph g;
    g.keep_dims(false);
    g.axes(tc.nchw_ind);
    g.shape(tc.shape);
    g.init();

    run_phase(&g.g, true, true);

    check_pre_trans(g.rm->input());

    auto rm_succs = loco::succs(g.rm);
    EXPECT_EQ(1, rm_succs.size());
    if (tc.needs_transpose)
    {
      EXPECT_NE(nullptr, dynamic_cast<luci::CircleTranspose *>(*rm_succs.begin()));
    }
    else
    {
      EXPECT_NE(nullptr, dynamic_cast<luci::CircleOutput *>(*rm_succs.begin()));
    }

    auto new_rindices = dynamic_cast<luci::CircleConst *>(g.rm->reduction_indices());
    EXPECT_NE(nullptr, new_rindices);
    EXPECT_EQ(1, new_rindices->rank());
    EXPECT_EQ(tc.nhwc_ind.size(), new_rindices->dim(0).value());
    EXPECT_EQ(tc.nhwc_ind.size(), new_rindices->size<loco::DataType::S32>());
    for (uint32_t i = 0; i < tc.nhwc_ind.size(); ++i)
    {
      EXPECT_EQ(tc.nhwc_ind[i], new_rindices->at<loco::DataType::S32>(i));
    }
  }
}

TEST(ConvertNCHWToNHWC, Relu)
{
  ReluGraph g;
  g.init();

  run_phase(&g.g, true, true);

  check_pre_trans(g.relu->features());

  auto relu_succs = loco::succs(g.relu);
  EXPECT_EQ(1, relu_succs.size());
  check_post_trans(*relu_succs.begin());

  // Check relu shape
  EXPECT_EQ(1, g.relu->dim(0).value());
  EXPECT_EQ(4, g.relu->dim(1).value());
  EXPECT_EQ(4, g.relu->dim(2).value());
  EXPECT_EQ(16, g.relu->dim(3).value());
}

TEST(ConvertNCHWToNHWC, Relu6)
{
  Relu6Graph g;
  g.init();

  run_phase(&g.g, true, true);

  check_pre_trans(g.relu6->features());

  auto relu6_succs = loco::succs(g.relu6);
  EXPECT_EQ(1, relu6_succs.size());
  check_post_trans(*relu6_succs.begin());

  // Check relu6 shape
  EXPECT_EQ(1, g.relu6->dim(0).value());
  EXPECT_EQ(4, g.relu6->dim(1).value());
  EXPECT_EQ(4, g.relu6->dim(2).value());
  EXPECT_EQ(16, g.relu6->dim(3).value());
}

TEST(ConvertNCHWToNHWC, Rsqrt)
{
  RsqrtGraph g;
  g.init();

  run_phase(&g.g, true, true);

  check_pre_trans(g.rsqrt->x());

  auto rsqrt_succs = loco::succs(g.rsqrt);
  EXPECT_EQ(1, rsqrt_succs.size());
  check_post_trans(*rsqrt_succs.begin());

  // Check rsqrt shape
  EXPECT_EQ(1, g.rsqrt->dim(0).value());
  EXPECT_EQ(4, g.rsqrt->dim(1).value());
  EXPECT_EQ(4, g.rsqrt->dim(2).value());
  EXPECT_EQ(16, g.rsqrt->dim(3).value());
}

TEST(ConvertNCHWToNHWC, SplitV)
{
  SplitVGraph g;
  g.init();

  run_phase(g.g(), true, true);

  check_pre_trans(g.splitv()->input());

  auto splitv_succs = loco::succs(g.splitv());
  for (auto svo : loco::succs(g.splitv()))
  {
    for (auto succ : loco::succs(svo))
    {
      check_post_trans(succ);
    }
  }

  // Check splitv() shape
  EXPECT_EQ(1, g.splitv()->dim(0).value());
  EXPECT_EQ(2, g.splitv()->dim(1).value());
  EXPECT_EQ(192, g.splitv()->dim(2).value());
  EXPECT_EQ(2, g.splitv()->dim(3).value());

  // Check axis
  auto axis = dynamic_cast<luci::CircleConst *>(g.splitv()->split_dim());
  EXPECT_NE(nullptr, axis);
  EXPECT_EQ(1, axis->size<loco::DataType::S32>());
  EXPECT_EQ(2, axis->at<loco::DataType::S32>(0));
}

TEST(ConvertNCHWToNHWC, SquaredDifference)
{
  SquaredDifferenceGraph g;
  g.init();

  run_phase(&g.g, true, true);

  check_pre_trans(g.sqdiff->x());
  check_pre_trans(g.sqdiff->y());

  auto sqdiff_succs = loco::succs(g.sqdiff);
  EXPECT_EQ(1, sqdiff_succs.size());
  check_post_trans(*sqdiff_succs.begin());
}

TEST(ConvertNCHWToNHWC, Sub)
{
  SubGraph g;
  g.init();

  run_phase(&g.g, false, false);

  auto input_succs = loco::succs(g.input);
  EXPECT_EQ(1, input_succs.size());
  check_post_trans(*input_succs.begin());

  check_pre_trans(g.sub->x());

  auto add_succs = loco::succs(g.sub);
  EXPECT_EQ(1, add_succs.size());
  check_post_trans(*add_succs.begin());

  uint32_t channel_size = 16;
  auto new_beta = dynamic_cast<luci::CircleConst *>(g.sub->y());
  EXPECT_NE(nullptr, new_beta);
  EXPECT_EQ(4, new_beta->rank());
  EXPECT_EQ(1, new_beta->dim(0).value());
  EXPECT_EQ(1, new_beta->dim(1).value());
  EXPECT_EQ(1, new_beta->dim(2).value());
  EXPECT_EQ(channel_size, new_beta->dim(3).value());

  check_pre_trans(g.output->from());
}

TEST(ConvertNCHWToNHWC, Sub_NCHW_const)
{
  SubGraph g;
  g.init();
  g.update_const_shape_to_nchw();

  run_phase(&g.g, false, false);

  check_pre_trans(g.sub->x());

  auto sub_succs = loco::succs(g.sub);
  EXPECT_EQ(1, sub_succs.size());
  check_post_trans(*sub_succs.begin());

  uint32_t channel_size = 16;
  auto new_beta = dynamic_cast<luci::CircleConst *>(g.sub->y());
  EXPECT_NE(nullptr, new_beta);
  EXPECT_EQ(4, new_beta->rank());
  EXPECT_EQ(1, new_beta->dim(0).value());
  EXPECT_EQ(4, new_beta->dim(1).value());
  EXPECT_EQ(4, new_beta->dim(2).value());
  EXPECT_EQ(channel_size, new_beta->dim(3).value());
}

TEST(ConvertNCHWToNHWC, SubScalar)
{
  SubScalarGraph g;
  g.init();

  run_phase(&g.g, false, false);

  auto input_succs = loco::succs(g.input);
  EXPECT_EQ(1, input_succs.size());
  check_post_trans(*input_succs.begin());

  check_pre_trans(g.sub->y());

  auto add_succs = loco::succs(g.sub);
  EXPECT_EQ(1, add_succs.size());
  check_post_trans(*add_succs.begin());

  auto new_beta = dynamic_cast<luci::CircleConst *>(g.sub->x());
  EXPECT_NE(nullptr, new_beta);
  EXPECT_EQ(1, new_beta->rank());

  check_pre_trans(g.output->from());
}

TEST(ConvertNCHWToNHWC, Not_Closed_Case1_NEG)
{
  NoPostReshapeGraph g;
  g.init();

  run_phase(&g.g, true, true);

  check_pre_trans(g.relu->features());

  auto relu_succs = loco::succs(g.relu);
  EXPECT_EQ(1, relu_succs.size());
  check_post_trans(*relu_succs.begin());
}

TEST(ConvertNCHWToNHWC, Not_Closed_Case2_NEG)
{
  ReluNotClosedGraph g;
  g.init();

  run_phase(&g.g, true, true);

  check_pre_trans(g.relu->features());

  auto relu_succs = loco::succs(g.relu);
  EXPECT_EQ(1, relu_succs.size());
  check_post_trans(*relu_succs.begin());
}
