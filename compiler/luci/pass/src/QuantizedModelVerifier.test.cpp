/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "QuantizedModelVerifier.h"

#include "luci/Pass/QuantizeWithMinMaxPass.h"
#include "luci/Pass/QuantizationParameters.h"
#include "luci/Pass/CircleTypeInferencePass.h"

#include <logo/Phase.h>
#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

using Type = loco::DataType;
using Granularity = luci::QuantizationGranularity;

namespace
{

/**
 * @brief A helper function to create dummy const node
 */
template <Type T> luci::CircleConst *create_dummy_const(loco::Graph *g, luci::test::ShapeU32 shape)
{
  auto node = g->nodes()->create<luci::CircleConst>();
  {
    node->dtype(T);
    node->shape(shape);
    node->size<T>(luci::test::num_elements(shape));

    for (int32_t i = 0; i < luci::test::num_elements(shape); i++)
    {
      // DESIGN NOTE
      //
      // Filling with any random numbers are fine
      // Q. Should it include minus numbers?
      switch (T)
      {
        case Type::FLOAT32:
          // Fill with index
          node->at<T>(i) = static_cast<float>(i);
          break;
        case Type::BOOL:
          // Fill by flip
          node->at<T>(i) = (i % 2) ? true : false;
          break;
        case Type::U8:
          // Fill with index
          node->at<T>(i) = static_cast<uint8_t>(i);
          break;
        case Type::S16:
          // Fill with index
          node->at<T>(i) = static_cast<int16_t>(i);
          break;
      }
    }
  }

  return node;
}

/**
 * @brief A helper function to create const node with value
 */
template <Type DT, typename T>
luci::CircleConst *create_const(loco::Graph *g, luci::test::ShapeU32 shape,
                                std::initializer_list<T> values)
{
  auto node = g->nodes()->create<luci::CircleConst>();
  {
    node->dtype(DT);
    node->shape(shape);
    node->size<DT>(luci::test::num_elements(shape));

    assert(values.size() == node->size<DT>());

    uint32_t index = 0;
    for (auto val : values)
    {
      node->at<DT>(index++) = static_cast<T>(val);
    }
  }

  return node;
}

void insert_scale_zp(luci::CircleNode *node, float scale, int64_t zp)
{
  auto qparam = node->quantparam();
  assert(qparam != nullptr); // FIX_CALLER_UNLESS
  qparam->scale.push_back(scale);
  qparam->zerop.push_back(zp);
}

void run_phase(loco::Graph *g, Type quantized_dtype, Granularity granularity)
{
  logo::Phase phase;

  // Default passes.
  phase.emplace_back(std::make_unique<luci::CircleTypeInferencePass>());

  auto ctx = std::make_unique<luci::QuantizeWithMinMaxPass::Context>();
  {
    ctx->input_model_dtype = loco::DataType::FLOAT32;
    ctx->output_model_dtype = quantized_dtype;
    ctx->granularity = granularity;
    // Test graph has only one input/output
    ctx->input_types = {quantized_dtype};
    ctx->output_types = {quantized_dtype};
  }

  phase.emplace_back(std::make_unique<luci::QuantizeWithMinMaxPass>(std::move(ctx)));

  logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{g};
  phase_runner.run(phase);
}

void run_phase(loco::Graph *g, std::unique_ptr<luci::QuantizeWithMinMaxPass::Context> &&ctx)
{
  logo::Phase phase;

  // Default passes.
  phase.emplace_back(std::make_unique<luci::CircleTypeInferencePass>());

  phase.emplace_back(std::make_unique<luci::QuantizeWithMinMaxPass>(std::move(ctx)));

  logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{g};
  phase_runner.run(phase);
}

void quantize_and_verify(loco::Graph *g, Type quantized_dtype, Granularity granularity)
{
  run_phase(g, quantized_dtype, granularity);

  auto ctx = std::make_unique<luci::QuantizedModelVerifier::Context>();
  {
    ctx->output_model_dtype = quantized_dtype;
    ctx->granularity = granularity;
    // Test graph has only one input/output
    ctx->input_types = {quantized_dtype};
    ctx->output_types = {quantized_dtype};
  }

  luci::QuantizedModelVerifier verifier(std::move(ctx));
  verifier.verify(g);
}

void quantize_and_verify_with_layer_info(loco::Graph *g, Type quantized_dtype,
                                         Granularity granularity)
{
  // A layer named "test" has dtype different from quantized_dtype
  luci::LayerInfo info;
  {
    info.name = "test";
    // dtype is different from quantized_dtype
    info.dtype = quantized_dtype == Type::U8 ? Type::S16 : Type::U8;
    info.granularity = Granularity::ChannelWise;
  }

  // Do quantization
  {
    auto ctx = std::make_unique<luci::QuantizeWithMinMaxPass::Context>();
    {
      ctx->input_model_dtype = Type::FLOAT32;
      ctx->output_model_dtype = quantized_dtype;
      ctx->granularity = granularity;
      // Test graph has only one input/output
      ctx->input_types = {quantized_dtype};
      ctx->output_types = {quantized_dtype};
      ctx->TF_style_maxpool = false;
      ctx->layers_info.push_back(info);
    }

    run_phase(g, std::move(ctx));
  }

  // Do verification
  {
    auto ctx = std::make_unique<luci::QuantizedModelVerifier::Context>();
    {
      ctx->output_model_dtype = quantized_dtype;
      ctx->granularity = granularity;
      ctx->input_types = {quantized_dtype};
      ctx->output_types = {quantized_dtype};
      ctx->TF_style_maxpool = false;
      ctx->layers_info.push_back(info);
    }

    luci::QuantizedModelVerifier verifier(std::move(ctx));
    verifier.verify(g);
  }
}

// Helper function to reduce duplicate test codes
// Assumption: g->output()->from() is the target node
void quantize_and_verify_with_wrong_type(luci::test::TestIOGraph *g, Type quantized_dtype,
                                         Granularity granularity, Type wrong_dtype)
{
  run_phase(g->g(), quantized_dtype, granularity);

  auto node = loco::must_cast<luci::CircleNode *>(g->output()->from());
  node->dtype(wrong_dtype);

  auto ctx = std::make_unique<luci::QuantizedModelVerifier::Context>();
  {
    ctx->output_model_dtype = quantized_dtype;
    ctx->granularity = granularity;
    // Test graph has only one input/output
    ctx->input_types = {quantized_dtype};
    ctx->output_types = {quantized_dtype};
  }

  luci::QuantizedModelVerifier verifier(std::move(ctx));
  verifier.verify(g->g());
}

// Helper function to reduce duplicate test codes
// Assumption: g->output()->from() is the target node
void quantize_and_verify_with_wrong_granularity(luci::test::TestIOGraph *g, Type quantized_dtype,
                                                Granularity granularity)
{
  run_phase(g->g(), quantized_dtype, granularity);

  auto node = loco::must_cast<luci::CircleNode *>(g->output()->from());
  insert_scale_zp(node, 1.0, 1);

  auto ctx = std::make_unique<luci::QuantizedModelVerifier::Context>();
  {
    ctx->output_model_dtype = quantized_dtype;
    ctx->granularity = granularity;
    // Test graph has only one input/output
    ctx->input_types = {quantized_dtype};
    ctx->output_types = {quantized_dtype};
  }

  luci::QuantizedModelVerifier verifier(std::move(ctx));
  verifier.verify(g->g());
}

// Set min/max for all non-const nodes in the graph
void set_minmax_to_non_const(loco::Graph *g, float min, float max)
{
  for (auto node : loco::all_nodes(g))
  {
    auto const_node = dynamic_cast<luci::CircleConst *>(node);
    if (const_node != nullptr)
      continue;

    // Min/Max is not recorded for ArgMax
    // See MinMaxObserver.cpp in record_minmax module
    auto argmax_node = dynamic_cast<luci::CircleArgMax *>(node);
    if (argmax_node != nullptr)
      continue;

    // Min/Max is not recorded for Split
    // See MinMaxObserver.cpp in record_minmax module
    auto split_node = dynamic_cast<luci::CircleSplit *>(node);
    if (split_node != nullptr)
      continue;

    // Min/Max is not recorded for SplitV
    // See MinMaxObserver.cpp in record_minmax module
    auto splitv_node = dynamic_cast<luci::CircleSplitV *>(node);
    if (splitv_node != nullptr)
      continue;

    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    auto qparam = std::make_unique<luci::CircleQuantParam>();
    {
      qparam->min.emplace_back(min);
      qparam->max.emplace_back(max);
    }
    circle_node->quantparam(std::move(qparam));
  }
}

/**
 * @brief Simple Test Graph
 * @note
 * The simple test graph's nodes are initialized with
 * simple shapes and values.
 */
class SimpleTestGraph : public luci::test::TestIOGraph
{
public:
  virtual void init(void) = 0;
};

class TypedTestGraph : public luci::test::TestIOGraph
{
protected:
  void init(Type T, const luci::test::ShapeU32 shape_in, const luci::test::ShapeU32 shape_out)
  {
    TestIOGraph::init(shape_in, shape_out);

    input()->dtype(T);
    output()->dtype(T);

    g()->inputs()->at(0)->dtype(T);
    g()->outputs()->at(0)->dtype(T);
  }

public:
  virtual void init(void) = 0;
};

class InstanceNormTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});
    _gamma = create_dummy_const<Type::FLOAT32>(g(), {32});
    _beta = create_dummy_const<Type::FLOAT32>(g(), {32});
    _instnorm = g()->nodes()->create<luci::CircleInstanceNorm>();
    {
      _instnorm->input(input());
      _instnorm->gamma(_gamma);
      _instnorm->beta(_beta);
      _instnorm->fusedActivationFunction(luci::FusedActFunc::NONE);
      _instnorm->name("test");
    }
    output()->from(_instnorm);

    set_minmax_to_non_const(g(), -1, 1);
  }

public:
  loco::Node *gamma(void) const { return _instnorm->gamma(); }
  loco::Node *beta(void) const { return _instnorm->beta(); }

private:
  luci::CircleInstanceNorm *_instnorm = nullptr;
  luci::CircleConst *_input = nullptr;
  luci::CircleConst *_gamma = nullptr;
  luci::CircleConst *_beta = nullptr;
};

class LogisticTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});
    _logistic = g()->nodes()->create<luci::CircleLogistic>();
    {
      _logistic->x(input());
      _logistic->name("test");
    }
    output()->from(_logistic);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleLogistic *_logistic = nullptr;
};

class LocalResponseNormalizationTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({1, 2, 2, 32}, {1, 2, 2, 32});
    _lrn = g()->nodes()->create<luci::CircleLocalResponseNormalization>();
    {
      _lrn->input(input());
      _lrn->name("test");
    }
    output()->from(_lrn);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleLocalResponseNormalization *_lrn = nullptr;
};

class SoftmaxTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});
    _softmax = g()->nodes()->create<luci::CircleSoftmax>();
    {
      _softmax->logits(input());
      _softmax->beta(0.1);
      _softmax->name("test");
    }
    output()->from(_softmax);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleSoftmax *_softmax = nullptr;
};

class SpaceToBatchNDTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({1, 2, 2, 1}, {4, 1, 1, 1});
    _block_shape = create_dummy_const<Type::S32>(g(), {2});
    for (uint32_t i = 0; i < 2; i++)
      _block_shape->at<Type::S32>(i) = 2;

    _paddings = create_dummy_const<Type::S32>(g(), {2, 2});
    for (uint32_t i = 0; i < 4; i++)
      _paddings->at<Type::S32>(i) = 0;

    _stob = g()->nodes()->create<luci::CircleSpaceToBatchND>();
    {
      _stob->input(input());
      _stob->block_shape(_block_shape);
      _stob->paddings(_paddings);
      _stob->name("test");
    }
    output()->from(_stob);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleSpaceToBatchND *_stob = nullptr;
  luci::CircleConst *_block_shape = nullptr;
  luci::CircleConst *_paddings = nullptr;
};

class SpaceToDepthTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({1, 2, 2, 1}, {1, 1, 1, 4});
    _stod = g()->nodes()->create<luci::CircleSpaceToDepth>();
    {
      _stod->input(input());
      _stod->block_size(2);
      _stod->name("test");
    }
    output()->from(_stod);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleSpaceToDepth *_stod = nullptr;
};

template <Type indexT> class SliceTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});
    _begin = g()->nodes()->template create<luci::CircleConst>();
    {
      _begin->dtype(indexT);
    }
    _size = g()->nodes()->template create<luci::CircleConst>();
    {
      _size->dtype(indexT);
    }
    _slice = g()->nodes()->template create<luci::CircleSlice>();
    {
      _slice->input(input());
      _slice->begin(_begin);
      _slice->size(_size);
      _slice->name("test");
    }
    output()->from(_slice);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleSlice *_slice = nullptr;
  luci::CircleConst *_begin = nullptr;
  luci::CircleConst *_size = nullptr;
};

class SplitTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({1, 32}, {32});
    _split_dim = create_dummy_const<Type::S32>(g(), {1});
    _split = g()->nodes()->create<luci::CircleSplit>();
    {
      _split->input(input());
      _split->split_dim(_split_dim);
    }
    _split_o1 = g()->nodes()->create<luci::CircleSplitOut>();
    {
      _split_o1->input(_split);
      _split_o1->index(0);
    }

    output()->from(_split_o1);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleSplit *_split = nullptr;
  luci::CircleSplitOut *_split_o1 = nullptr;
  luci::CircleConst *_split_dim = nullptr;
};

class SplitVTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({1, 32}, {32});
    _size_splits = create_dummy_const<Type::S32>(g(), {1});
    _split_dim = create_dummy_const<Type::S32>(g(), {1});
    _splitv = g()->nodes()->create<luci::CircleSplitV>();
    {
      _splitv->input(input());
      _splitv->size_splits(_size_splits);
      _splitv->split_dim(_split_dim);
    }
    _splitv_o1 = g()->nodes()->create<luci::CircleSplitVOut>();
    {
      _splitv_o1->input(_splitv);
      _splitv_o1->index(0);
    }

    output()->from(_splitv_o1);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleSplitV *_splitv = nullptr;
  luci::CircleSplitVOut *_splitv_o1 = nullptr;
  luci::CircleConst *_size_splits = nullptr;
  luci::CircleConst *_split_dim = nullptr;
};

class StridedSliceTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});
    _begin = g()->nodes()->create<luci::CircleConst>();
    {
      _begin->dtype(Type::S32);
    }
    _end = g()->nodes()->create<luci::CircleConst>();
    {
      _end->dtype(Type::S32);
    }
    _strides = g()->nodes()->create<luci::CircleConst>();
    {
      _strides->dtype(Type::S32);
    }
    _slice = g()->nodes()->create<luci::CircleStridedSlice>();
    {
      _slice->input(input());
      _slice->begin(_begin);
      _slice->end(_end);
      _slice->strides(_strides);
      _slice->name("test");
    }
    output()->from(_slice);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleStridedSlice *_slice = nullptr;
  luci::CircleConst *_begin = nullptr;
  luci::CircleConst *_end = nullptr;
  luci::CircleConst *_strides = nullptr;
};

class SumTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({4, 3, 2}, {2});

    _axis = create_const<Type::S32, int32_t>(g(), {2}, {1, 0});
    _sum = g()->nodes()->create<luci::CircleSum>();
    {
      _sum->input(input());
      _sum->reduction_indices(_axis);
      _sum->name("test");
      _sum->keep_dims(false);
    }
    output()->from(_sum);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleSum *_sum = nullptr;
  luci::CircleConst *_axis = nullptr;
};

class ReshapeTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});
    _shape = g()->nodes()->create<luci::CircleConst>();
    {
      _shape->dtype(Type::S32);
    }
    _reshape = g()->nodes()->create<luci::CircleReshape>();
    {
      _reshape->tensor(input());
      _reshape->shape(_shape);
      _reshape->name("test");
    }
    output()->from(_reshape);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleReshape *_reshape = nullptr;
  luci::CircleConst *_shape = nullptr;
};

class TanhTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});
    _tanh = g()->nodes()->create<luci::CircleTanh>();
    {
      _tanh->x(input());
      _tanh->name("test");
    }
    output()->from(_tanh);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleTanh *_tanh = nullptr;
};

class FloorTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});
    _floor = g()->nodes()->create<luci::CircleFloor>();
    {
      _floor->x(input());
      _floor->name("test");
    }
    output()->from(_floor);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleFloor *_floor = nullptr;
};

template <Type indexT> class ArgMaxTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {1});
    // output dtype is float by default, but ArgMax should have indexType (s32/s64)
    output()->dtype(indexT);
    _dimension = g()->nodes()->template create<luci::CircleConst>();
    {
      _dimension->dtype(indexT);
    }
    _argmax = g()->nodes()->template create<luci::CircleArgMax>();
    {
      _argmax->input(input());
      _argmax->dimension(_dimension);
      _argmax->output_type(indexT);
      _argmax->dtype(indexT);
    }
    output()->from(_argmax);

    set_minmax_to_non_const(g(), -1, 1);

    // Sync output dtype with graph's output dtype
    g()->outputs()->at(0)->dtype(output()->dtype());
  }

public:
  // NOTE: Do not override `luci::CircleNode* input(void)` incidentally
  loco::Node *input_argmax(void) { return _argmax->input(); }
  loco::Node *dimension(void) { return _argmax->dimension(); }

private:
  luci::CircleArgMax *_argmax = nullptr;
  luci::CircleConst *_dimension = nullptr;
};

class BatchToSpaceNDTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});
    _block_shape = g()->nodes()->create<luci::CircleConst>();
    {
      _block_shape->dtype(Type::S32);
    }
    _crops = g()->nodes()->create<luci::CircleConst>();
    {
      _crops->dtype(Type::S32);
    }
    _btos = g()->nodes()->create<luci::CircleBatchToSpaceND>();
    {
      _btos->input(input());
      _btos->block_shape(_block_shape);
      _btos->crops(_crops);
      _btos->name("test");
    }
    output()->from(_btos);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleBatchToSpaceND *_btos = nullptr;
  luci::CircleConst *_block_shape = nullptr;
  luci::CircleConst *_crops = nullptr;
};

class DepthToSpaceTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({1, 1, 1, 4}, {1, 2, 2, 1});
    _dtos = g()->nodes()->create<luci::CircleDepthToSpace>();
    {
      _dtos->input(input());
      _dtos->block_size(2);
      _dtos->name("test");
    }
    output()->from(_dtos);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleDepthToSpace *_dtos = nullptr;
};

class PackTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({16}, {32});
    _param = create_dummy_const<Type::FLOAT32>(g(), {16});
    _pack = g()->nodes()->create<luci::CirclePack>(2);
    {
      _pack->values(0, input());
      _pack->values(1, _param);
      _pack->axis(0);
      _pack->name("test");
    }
    output()->from(_pack);

    set_minmax_to_non_const(g(), -1, 1);

    // Set min/max of the input
    // pack's qparam will be propagted, overwritten to the input
    auto input = loco::must_cast<luci::CircleNode *>(pack()->values(0));
    auto qp = input->quantparam();
    qp->min[0] = -0.5;
    qp->max[0] = 0.5;
  }

public:
  luci::CirclePack *pack(void) { return _pack; }

private:
  luci::CirclePack *_pack = nullptr;
  luci::CircleConst *_param = nullptr;
};

class PadTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});
    _paddings = g()->nodes()->create<luci::CircleConst>();
    {
      _paddings->dtype(Type::S32);
    }
    _pad = g()->nodes()->create<luci::CirclePad>();
    {
      _pad->input(input());
      _pad->paddings(_paddings);
      _pad->name("test");
    }
    output()->from(_pad);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CirclePad *_pad = nullptr;
  luci::CircleConst *_paddings = nullptr;
};

class PadV2TestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});
    _paddings = g()->nodes()->create<luci::CircleConst>();
    {
      _paddings->dtype(Type::S32);
    }
    _constant_values = create_dummy_const<Type::FLOAT32>(g(), {1});
    _pad = g()->nodes()->create<luci::CirclePadV2>();
    {
      _pad->input(input());
      _pad->paddings(_paddings);
      _pad->constant_values(_constant_values);
      _pad->name("test");
    }
    output()->from(_pad);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CirclePadV2 *_pad = nullptr;
  luci::CircleConst *_paddings = nullptr;
  luci::CircleConst *_constant_values = nullptr;
};

class MirrorPadTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});
    _paddings = g()->nodes()->create<luci::CircleConst>();
    {
      _paddings->dtype(Type::S32);
    }
    _constant_values = create_dummy_const<Type::FLOAT32>(g(), {1});
    _mirror_pad = g()->nodes()->create<luci::CircleMirrorPad>();
    {
      _mirror_pad->input(input());
      _mirror_pad->paddings(_paddings);
      _mirror_pad->mode(luci::MirrorPadMode::REFLECT);
      _mirror_pad->name("test");
    }
    output()->from(_mirror_pad);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleMirrorPad *_mirror_pad = nullptr;
  luci::CircleConst *_paddings = nullptr;
  luci::CircleConst *_constant_values = nullptr;
};

class TransposeTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});
    _perm = g()->nodes()->create<luci::CircleConst>();
    {
      _perm->dtype(Type::S32);
    }
    _transpose = g()->nodes()->create<luci::CircleTranspose>();
    {
      _transpose->a(input());
      _transpose->perm(_perm);
      _transpose->name("test");
    }
    output()->from(_transpose);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleTranspose *_transpose = nullptr;
  luci::CircleConst *_perm = nullptr;
};

class ConcatenationTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({16}, {32});
    _param = create_dummy_const<Type::FLOAT32>(g(), {16});
    _concat = g()->nodes()->create<luci::CircleConcatenation>(2);
    {
      _concat->values(0, input());
      _concat->values(1, _param);
      _concat->axis(0);
      _concat->fusedActivationFunction(luci::FusedActFunc::NONE);
      _concat->name("test");
    }
    output()->from(_concat);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleConcatenation *_concat = nullptr;
  luci::CircleConst *_param = nullptr;
};

template <Type indexT> class OneHotTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32, 10});
    {
      // input dtype is float by default, but OneHot's input should have indexType (s32/s64)
      input()->dtype(indexT);
    }

    _depth = g()->nodes()->template create<luci::CircleConst>();
    {
      _depth->dtype(loco::DataType::S32);
    }

    _on_value = g()->nodes()->template create<luci::CircleConst>();
    {
      _on_value->dtype(loco::DataType::FLOAT32);
    }

    _off_value = g()->nodes()->template create<luci::CircleConst>();
    {
      _off_value->dtype(loco::DataType::FLOAT32);
    }

    _one_hot = g()->nodes()->template create<luci::CircleOneHot>();
    {
      _one_hot->indices(input());
      _one_hot->depth(_depth);
      _one_hot->on_value(_on_value);
      _one_hot->off_value(_off_value);
      _one_hot->axis(-1);
      _one_hot->dtype(loco::DataType::FLOAT32);
      _one_hot->name("test");
    }
    output()->from(_one_hot);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleOneHot *_one_hot = nullptr;
  luci::CircleConst *_depth = nullptr;
  luci::CircleConst *_on_value = nullptr;
  luci::CircleConst *_off_value = nullptr;
};

// Test graph for comparison Ops
// GREATER, GREATER_EQUAL, LESS, LESS_EQUAL, EQUAL, NOT_EQUAL
template <class Op> class ComparisonOpTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});
    output()->dtype(loco::DataType::BOOL);
    _y = create_dummy_const<Type::FLOAT32>(g(), {32});
    _op = g()->nodes()->template create<Op>();
    {
      _op->x(input());
      _op->y(_y);
      _op->dtype(loco::DataType::BOOL);
    }
    output()->from(_op);

    set_minmax_to_non_const(g(), -1, 1);

    // Sync output dtype with graph's output dtype
    g()->outputs()->at(0)->dtype(output()->dtype());
  }

  loco::Node *x(void) const { return _op->x(); }
  loco::Node *y(void) const { return _op->y(); }

private:
  Op *_op = nullptr;
  luci::CircleConst *_y = nullptr;
};

// Test graph for binary logical Ops
// LOGICAL_OR, LOGICAL_AND
template <class Op> class BinaryLogicalOpTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});
    input()->dtype(loco::DataType::BOOL);
    output()->dtype(loco::DataType::BOOL);
    _y = create_dummy_const<Type::BOOL>(g(), {32});
    _op = g()->nodes()->template create<Op>();
    {
      _op->x(input());
      _op->y(_y);
      _op->dtype(loco::DataType::BOOL);
    }
    output()->from(_op);

    set_minmax_to_non_const(g(), -1, 1);

    // Sync output dtype with graph's output dtype
    g()->outputs()->at(0)->dtype(output()->dtype());
  }

  loco::Node *x(void) const { return _op->x(); }
  loco::Node *y(void) const { return _op->y(); }

private:
  Op *_op = nullptr;
  luci::CircleConst *_y = nullptr;
};

class DivTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});

    _const = create_dummy_const<Type::FLOAT32>(g(), {32});
    _div = g()->nodes()->create<luci::CircleDiv>();
    {
      _div->x(input());
      _div->y(_const);
      _div->name("test");
    }
    output()->from(_div);

    set_minmax_to_non_const(g(), -1, 1);
  }

  loco::Node *x() { return _div->x(); }

  loco::Node *y() { return _div->y(); }

private:
  luci::CircleDiv *_div = nullptr;
  luci::CircleConst *_const = nullptr;
};

class FloorDivTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});

    _const = create_dummy_const<Type::FLOAT32>(g(), {32});
    _floor_div = g()->nodes()->create<luci::CircleFloorDiv>();
    {
      _floor_div->x(input());
      _floor_div->y(_const);
      _floor_div->name("test");
    }
    output()->from(_floor_div);

    set_minmax_to_non_const(g(), -1, 1);
  }

  loco::Node *x() { return _floor_div->x(); }

  loco::Node *y() { return _floor_div->y(); }

private:
  luci::CircleFloorDiv *_floor_div = nullptr;
  luci::CircleConst *_const = nullptr;
};

class RsqrtTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});
    _rsqrt = g()->nodes()->create<luci::CircleRsqrt>();
    {
      _rsqrt->x(input());
      _rsqrt->name("test");
    }
    output()->from(_rsqrt);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleRsqrt *_rsqrt = nullptr;
};

class SqrtTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});
    _sqrt = g()->nodes()->create<luci::CircleSqrt>();
    {
      _sqrt->x(input());
      _sqrt->name("test");
    }
    output()->from(_sqrt);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleSqrt *_sqrt = nullptr;
};

class EluTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});
    _elu = g()->nodes()->create<luci::CircleElu>();
    {
      _elu->features(input());
      _elu->name("test");
    }
    output()->from(_elu);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleElu *_elu = nullptr;
};

class PowTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});

    _const = create_dummy_const<Type::FLOAT32>(g(), {32});
    _pow = g()->nodes()->create<luci::CirclePow>();
    {
      _pow->x(input());
      _pow->y(_const);
      _pow->name("test");
    }
    output()->from(_pow);

    set_minmax_to_non_const(g(), -1, 1);
  }

  loco::Node *x() { return _pow->x(); }

  loco::Node *y() { return _pow->y(); }

private:
  luci::CirclePow *_pow = nullptr;
  luci::CircleConst *_const = nullptr;
};

class ReduceMaxTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({4, 3, 2}, {2});

    _axis = create_const<Type::S32, int32_t>(g(), {4}, {1, 0, -3, -3});
    _reduce_max = g()->nodes()->create<luci::CircleReduceMax>();
    {
      _reduce_max->input(input());
      _reduce_max->reduction_indices(_axis);
      _reduce_max->name("test");
      _reduce_max->keep_dims(false);
    }
    output()->from(_reduce_max);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleReduceMax *_reduce_max = nullptr;
  luci::CircleConst *_axis = nullptr;
};

class ResizeBilinearTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({1, 4, 4, 1}, {1, 8, 8, 1});

    _size = create_const<Type::S32, int32_t>(g(), {2}, {8, 8});
    _resize_bilinear = g()->nodes()->create<luci::CircleResizeBilinear>();
    {
      _resize_bilinear->input(input());
      _resize_bilinear->size(_size);
      _resize_bilinear->name("test");
    }
    output()->from(_resize_bilinear);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleResizeBilinear *_resize_bilinear = nullptr;
  luci::CircleConst *_size = nullptr;
};

class ResizeNearestNeighborTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({1, 4, 4, 1}, {1, 8, 8, 1});

    _size = create_const<Type::S32, int32_t>(g(), {2}, {8, 8});
    _resize_nearest_neighbor = g()->nodes()->create<luci::CircleResizeNearestNeighbor>();
    {
      _resize_nearest_neighbor->input(input());
      _resize_nearest_neighbor->size(_size);
      _resize_nearest_neighbor->name("test");
    }
    output()->from(_resize_nearest_neighbor);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleResizeNearestNeighbor *_resize_nearest_neighbor = nullptr;
  luci::CircleConst *_size = nullptr;
};

class UnpackTestGraph final : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({1, 32}, {32});
    _unpack = g()->nodes()->create<luci::CircleUnpack>();
    {
      _unpack->value(input());
      _unpack->axis(0);
      _unpack->num(1);
    }
    _unpack_o1 = g()->nodes()->create<luci::CircleUnpackOut>();
    {
      _unpack_o1->input(_unpack);
      _unpack_o1->index(0);
    }

    output()->from(_unpack_o1);

    set_minmax_to_non_const(g(), -1, 1);
  }

private:
  luci::CircleUnpack *_unpack = nullptr;
  luci::CircleUnpackOut *_unpack_o1 = nullptr;
  luci::CircleConst *_unpack_dim = nullptr;
};

class MulTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});

    _const = create_dummy_const<Type::FLOAT32>(g(), {32});
    _mul = g()->nodes()->create<luci::CircleMul>();
    {
      _mul->x(input());
      _mul->y(_const);
      _mul->fusedActivationFunction(luci::FusedActFunc::NONE);
      _mul->name("test");
    }
    output()->from(_mul);

    set_minmax_to_non_const(g(), -1, 1);
  }

  loco::Node *x() { return _mul->x(); }
  loco::Node *y() { return _mul->y(); }

private:
  luci::CircleMul *_mul = nullptr;
  luci::CircleConst *_const = nullptr;
};

template <Type T> class IntMulTestGraph final : public TypedTestGraph
{
public:
  void init(void) override
  {
    TypedTestGraph::init(T, {32}, {32});

    _const = create_dummy_const<T>(g(), {32});
    _mul = g()->nodes()->template create<luci::CircleMul>();
    {
      _mul->x(input());
      _mul->y(_const);
      _mul->fusedActivationFunction(luci::FusedActFunc::NONE);
      _mul->name("test");
      _mul->dtype(T);
    }
    output()->from(_mul);
  }

  loco::Node *x() { return _mul->x(); }
  loco::Node *y() { return _mul->y(); }

private:
  luci::CircleMul *_mul = nullptr;
  luci::CircleConst *_const = nullptr;
};

class AddTestGraph final : public SimpleTestGraph
{
public:
  void init(void) override
  {
    TestIOGraph::init({32}, {32});

    _const = create_dummy_const<Type::FLOAT32>(g(), {32});
    _add = g()->nodes()->create<luci::CircleAdd>();
    {
      _add->x(input());
      _add->y(_const);
      _add->fusedActivationFunction(luci::FusedActFunc::NONE);
      _add->name("test");
    }
    output()->from(_add);

    set_minmax_to_non_const(g(), -1, 1);
  }

  loco::Node *x() { return _add->x(); }
  loco::Node *y() { return _add->y(); }

private:
  luci::CircleAdd *_add = nullptr;
  luci::CircleConst *_const = nullptr;
};

template <Type T> class IntAddTestGraph final : public TypedTestGraph
{
public:
  void init(void) override
  {
    TypedTestGraph::init(T, {32}, {32});

    _const = create_dummy_const<T>(g(), {32});
    _add = g()->nodes()->template create<luci::CircleAdd>();
    {
      _add->x(input());
      _add->y(_const);
      _add->fusedActivationFunction(luci::FusedActFunc::NONE);
      _add->name("test");
      _add->dtype(T);
    }
    output()->from(_add);
  }

  loco::Node *x() { return _add->x(); }
  loco::Node *y() { return _add->y(); }

private:
  luci::CircleAdd *_add = nullptr;
  luci::CircleConst *_const = nullptr;
};

} // namespace

// Quantize and verify with given configurations
#define TEST_WITH_GRAPH(graph, type, granularity)                   \
  do                                                                \
  {                                                                 \
    graph g;                                                        \
    g.init();                                                       \
    EXPECT_NO_THROW(quantize_and_verify(g.g(), type, granularity)); \
  } while (0)

// Quantize and verify with layer info
#define TEST_WITH_LAYER_INFO(graph, type, granularity)                              \
  do                                                                                \
  {                                                                                 \
    graph g;                                                                        \
    g.init();                                                                       \
    EXPECT_NO_THROW(quantize_and_verify_with_layer_info(g.g(), type, granularity)); \
  } while (0)

// Quantize and verify with wrong type
#define TEST_WITH_WRONG_TYPE(graph, type, granularity, wrong_dtype)                            \
  do                                                                                           \
  {                                                                                            \
    graph g;                                                                                   \
    g.init();                                                                                  \
    EXPECT_ANY_THROW(quantize_and_verify_with_wrong_type(&g, type, granularity, wrong_dtype)); \
  } while (0)

// Quantize and verify with wrong granularity
#define TEST_WITH_WRONG_GRANULARITY(graph, type, granularity)                            \
  do                                                                                     \
  {                                                                                      \
    graph g;                                                                             \
    g.init();                                                                            \
    EXPECT_ANY_THROW(quantize_and_verify_with_wrong_granularity(&g, type, granularity)); \
  } while (0)

// Quantize and verify with wrong type
// Users can specify the test target
#define TEST_WITH_WRONG_TYPE_TARGET(graph, type, granularity_, wrong_dtype, target) \
  do                                                                                \
  {                                                                                 \
    graph g;                                                                        \
    g.init();                                                                       \
    auto node = loco::must_cast<luci::CircleNode *>(target);                        \
    run_phase(g.g(), type, granularity_);                                           \
    auto after_node = loco::must_cast<luci::CircleNode *>(target);                  \
    after_node->dtype(wrong_dtype);                                                 \
    auto ctx = std::make_unique<luci::QuantizedModelVerifier::Context>();           \
    {                                                                               \
      ctx->output_model_dtype = type;                                               \
      ctx->granularity = granularity_;                                              \
      ctx->input_types = {type};                                                    \
      ctx->output_types = {type};                                                   \
    }                                                                               \
    luci::QuantizedModelVerifier verifier(std::move(ctx));                          \
    EXPECT_ANY_THROW(verifier.verify(g.g()));                                       \
  } while (0)

// Quantize and verify with wrong granularity
// Users can specify the test target
#define TEST_WITH_WRONG_GRANULARITY_TARGET(graph, type, granularity_, target) \
  do                                                                          \
  {                                                                           \
    graph g;                                                                  \
    g.init();                                                                 \
    auto node = loco::must_cast<luci::CircleNode *>(target);                  \
    run_phase(g.g(), type, granularity_);                                     \
    auto after_node = loco::must_cast<luci::CircleNode *>(target);            \
    insert_scale_zp(after_node, 1.0, 1);                                      \
    auto ctx = std::make_unique<luci::QuantizedModelVerifier::Context>();     \
    {                                                                         \
      ctx->output_model_dtype = type;                                         \
      ctx->granularity = granularity_;                                        \
      ctx->input_types = {type};                                              \
      ctx->output_types = {type};                                             \
    }                                                                         \
    luci::QuantizedModelVerifier verifier(std::move(ctx));                    \
    EXPECT_ANY_THROW(verifier.verify(g.g()));                                 \
  } while (0)

// Test a local helper function
TEST(QuantizedModelVerifierTest, LocalCreateDummyConst)
{
  loco::Graph g;

  EXPECT_NO_THROW(create_dummy_const<Type::FLOAT32>(&g, {32, 32}));
}

TEST(QuantizedModelVerifierTest, LocalCreateConst)
{
  loco::Graph g;
  std::initializer_list<float> values = {0.1, 0, -5, 100};
  luci::CircleConst *node = create_const<Type::FLOAT32, float>(&g, {2, 2}, values);

  uint32_t index = 0;
  for (auto val : values)
  {
    EXPECT_EQ(node->at<Type::FLOAT32>(index++), val);
  }
}

TEST(QuantizedModelVerifierTest, InstanceNorm)
{
  TEST_WITH_GRAPH(InstanceNormTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(InstanceNormTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(InstanceNormTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(InstanceNormTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(InstanceNormTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(InstanceNormTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, InstanceNorm_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(InstanceNormTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(InstanceNormTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(InstanceNormTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, InstanceNorm_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(InstanceNormTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(InstanceNormTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(InstanceNormTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, LocalResponseNormalization)
{
  TEST_WITH_GRAPH(LocalResponseNormalizationTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(LocalResponseNormalizationTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(LocalResponseNormalizationTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(LocalResponseNormalizationTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(LocalResponseNormalizationTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(LocalResponseNormalizationTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, LocalResponseNormalization_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(LocalResponseNormalizationTestGraph, Type::U8, Granularity::LayerWise,
                       Type::S16);
  TEST_WITH_WRONG_TYPE(LocalResponseNormalizationTestGraph, Type::U8, Granularity::ChannelWise,
                       Type::S16);
  TEST_WITH_WRONG_TYPE(LocalResponseNormalizationTestGraph, Type::S16, Granularity::ChannelWise,
                       Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, LocalResponseNormalization_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(LocalResponseNormalizationTestGraph, Type::U8,
                              Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(LocalResponseNormalizationTestGraph, Type::U8,
                              Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(LocalResponseNormalizationTestGraph, Type::S16,
                              Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Logistic)
{
  TEST_WITH_GRAPH(LogisticTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(LogisticTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(LogisticTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(LogisticTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(LogisticTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(LogisticTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Logistic_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(LogisticTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(LogisticTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(LogisticTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Logistic_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(LogisticTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(LogisticTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(LogisticTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Softmax)
{
  TEST_WITH_GRAPH(SoftmaxTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(SoftmaxTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(SoftmaxTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(SoftmaxTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(SoftmaxTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(SoftmaxTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Softmax_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(SoftmaxTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SoftmaxTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SoftmaxTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Softmax_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(SoftmaxTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(SoftmaxTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(SoftmaxTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, SpaceToBatchND)
{
  TEST_WITH_GRAPH(SpaceToBatchNDTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(SpaceToBatchNDTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(SpaceToBatchNDTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(SpaceToBatchNDTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(SpaceToBatchNDTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(SpaceToBatchNDTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, SpaceToBatchND_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(SpaceToBatchNDTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SpaceToBatchNDTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SpaceToBatchNDTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, SpaceToBatchND_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(SpaceToBatchNDTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(SpaceToBatchNDTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(SpaceToBatchNDTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, SpaceToDepth)
{
  TEST_WITH_GRAPH(SpaceToDepthTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(SpaceToDepthTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(SpaceToDepthTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(SpaceToDepthTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(SpaceToDepthTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(SpaceToDepthTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, SpaceToDepth_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(SpaceToDepthTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SpaceToDepthTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SpaceToDepthTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, SpaceToDepth_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(SpaceToDepthTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(SpaceToDepthTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(SpaceToDepthTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Slice)
{
  TEST_WITH_GRAPH(SliceTestGraph<Type::S32>, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(SliceTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(SliceTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise);

  TEST_WITH_GRAPH(SliceTestGraph<Type::S64>, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(SliceTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(SliceTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(SliceTestGraph<Type::S32>, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(SliceTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(SliceTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(SliceTestGraph<Type::S64>, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(SliceTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(SliceTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Slice_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(SliceTestGraph<Type::S32>, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SliceTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SliceTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise, Type::U8);

  TEST_WITH_WRONG_TYPE(SliceTestGraph<Type::S64>, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SliceTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SliceTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Slice_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(SliceTestGraph<Type::S32>, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(SliceTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(SliceTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise);

  TEST_WITH_WRONG_GRANULARITY(SliceTestGraph<Type::S64>, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(SliceTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(SliceTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Split)
{
  TEST_WITH_GRAPH(SplitTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(SplitTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(SplitTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Split_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(SplitTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SplitTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SplitTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Split_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(SplitTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(SplitTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(SplitTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, SplitV)
{
  TEST_WITH_GRAPH(SplitVTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(SplitVTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(SplitVTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, SplitV_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(SplitVTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SplitVTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SplitVTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, SplitV_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(SplitVTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(SplitVTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(SplitVTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, StridedSlice)
{
  TEST_WITH_GRAPH(StridedSliceTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(StridedSliceTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(StridedSliceTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(StridedSliceTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(StridedSliceTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(StridedSliceTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, StridedSlice_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(StridedSliceTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(StridedSliceTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(StridedSliceTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, StridedSlice_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(StridedSliceTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(StridedSliceTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(StridedSliceTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Sum)
{
  TEST_WITH_GRAPH(SumTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(SumTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(SumTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(SumTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(SumTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(SumTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Sum_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(SumTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SumTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SumTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Sum_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(SumTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(SumTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(SumTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, ArgMax)
{
  TEST_WITH_GRAPH(ArgMaxTestGraph<Type::S32>, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(ArgMaxTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(ArgMaxTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise);

  TEST_WITH_GRAPH(ArgMaxTestGraph<Type::S64>, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(ArgMaxTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(ArgMaxTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, ArgMax_wrong_input_type_NEG)
{
  TEST_WITH_WRONG_TYPE(ArgMaxTestGraph<Type::S32>, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(ArgMaxTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(ArgMaxTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise, Type::U8);

  TEST_WITH_WRONG_TYPE(ArgMaxTestGraph<Type::S64>, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(ArgMaxTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(ArgMaxTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, ArgMax_wrong_dimension_type_NEG)
{
  TEST_WITH_WRONG_TYPE_TARGET(ArgMaxTestGraph<Type::S32>, Type::U8, Granularity::LayerWise,
                              Type::S16, g.dimension());
  TEST_WITH_WRONG_TYPE_TARGET(ArgMaxTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise,
                              Type::S16, g.dimension());
  TEST_WITH_WRONG_TYPE_TARGET(ArgMaxTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise,
                              Type::U8, g.dimension());

  TEST_WITH_WRONG_TYPE_TARGET(ArgMaxTestGraph<Type::S64>, Type::U8, Granularity::LayerWise,
                              Type::S16, g.dimension());
  TEST_WITH_WRONG_TYPE_TARGET(ArgMaxTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise,
                              Type::S16, g.dimension());
  TEST_WITH_WRONG_TYPE_TARGET(ArgMaxTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise,
                              Type::U8, g.dimension());
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, ArgMax_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY_TARGET(ArgMaxTestGraph<Type::S32>, Type::U8, Granularity::LayerWise,
                                     g.input_argmax());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ArgMaxTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise,
                                     g.input_argmax());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ArgMaxTestGraph<Type::S32>, Type::S16,
                                     Granularity::ChannelWise, g.input_argmax());

  TEST_WITH_WRONG_GRANULARITY_TARGET(ArgMaxTestGraph<Type::S64>, Type::U8, Granularity::LayerWise,
                                     g.input_argmax());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ArgMaxTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise,
                                     g.input_argmax());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ArgMaxTestGraph<Type::S64>, Type::S16,
                                     Granularity::ChannelWise, g.input_argmax());
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, BatchToSpaceND)
{
  TEST_WITH_GRAPH(BatchToSpaceNDTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(BatchToSpaceNDTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(BatchToSpaceNDTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(BatchToSpaceNDTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(BatchToSpaceNDTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(BatchToSpaceNDTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, BatchToSpaceND_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(BatchToSpaceNDTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(BatchToSpaceNDTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(BatchToSpaceNDTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, BatchToSpaceND_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(BatchToSpaceNDTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(BatchToSpaceNDTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(BatchToSpaceNDTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, DepthToSpace)
{
  TEST_WITH_GRAPH(DepthToSpaceTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(DepthToSpaceTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(DepthToSpaceTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(DepthToSpaceTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(DepthToSpaceTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(DepthToSpaceTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, DepthToSpace_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(DepthToSpaceTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(DepthToSpaceTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(DepthToSpaceTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, DepthToSpace_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(DepthToSpaceTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(DepthToSpaceTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(DepthToSpaceTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Concatenation)
{
  TEST_WITH_GRAPH(ConcatenationTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(ConcatenationTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(ConcatenationTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(ConcatenationTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(ConcatenationTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(ConcatenationTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Concatenation_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(ConcatenationTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(ConcatenationTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(ConcatenationTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Concatenation_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(ConcatenationTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(ConcatenationTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(ConcatenationTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, LogicalOr)
{
  TEST_WITH_GRAPH(BinaryLogicalOpTestGraph<luci::CircleLogicalOr>, Type::U8,
                  Granularity::LayerWise);
  TEST_WITH_GRAPH(BinaryLogicalOpTestGraph<luci::CircleLogicalOr>, Type::U8,
                  Granularity::ChannelWise);
  TEST_WITH_GRAPH(BinaryLogicalOpTestGraph<luci::CircleLogicalOr>, Type::S16,
                  Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, LogicalOr_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(BinaryLogicalOpTestGraph<luci::CircleLogicalOr>, Type::U8,
                       Granularity::LayerWise, Type::U8);
  TEST_WITH_WRONG_TYPE(BinaryLogicalOpTestGraph<luci::CircleLogicalOr>, Type::U8,
                       Granularity::ChannelWise, Type::U8);
  TEST_WITH_WRONG_TYPE(BinaryLogicalOpTestGraph<luci::CircleLogicalOr>, Type::S16,
                       Granularity::ChannelWise, Type::S16);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Reshape)
{
  TEST_WITH_GRAPH(ReshapeTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(ReshapeTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(ReshapeTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(ReshapeTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(ReshapeTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(ReshapeTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Reshape_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(ReshapeTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(ReshapeTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(ReshapeTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Reshape_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(ReshapeTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(ReshapeTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(ReshapeTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Tanh)
{
  TEST_WITH_GRAPH(TanhTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(TanhTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(TanhTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(TanhTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(TanhTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(TanhTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Tanh_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(TanhTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(TanhTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(TanhTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Tanh_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(TanhTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(TanhTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(TanhTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Pack)
{
  TEST_WITH_GRAPH(PackTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(PackTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(PackTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(PackTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(PackTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(PackTestGraph, Type::S16, Granularity::ChannelWise);

  // Test if Pack's qparam is propagated to the input
  {
    PackTestGraph g;
    g.init();
    quantize_and_verify(g.g(), Type::U8, Granularity::ChannelWise);
    auto input = loco::must_cast<luci::CircleNode *>(g.pack()->values(0));
    auto qp = input->quantparam();
    EXPECT_FLOAT_EQ(2.0 / 255.0, qp->scale[0]);
    EXPECT_FLOAT_EQ(128, qp->zerop[0]);
  }
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Pack_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(PackTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(PackTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(PackTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Pack_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(PackTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(PackTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(PackTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Pad)
{
  TEST_WITH_GRAPH(PadTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(PadTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(PadTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(PadTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(PadTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(PadTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Pad_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(PadTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(PadTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(PadTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Pad_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(PadTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(PadTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(PadTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, PadV2)
{
  TEST_WITH_GRAPH(PadV2TestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(PadV2TestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(PadV2TestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(PadV2TestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(PadV2TestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(PadV2TestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, PadV2_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(PadV2TestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(PadV2TestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(PadV2TestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, PadV2_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(PadV2TestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(PadV2TestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(PadV2TestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, MirrorPad)
{
  TEST_WITH_GRAPH(MirrorPadTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(MirrorPadTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(MirrorPadTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(MirrorPadTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(MirrorPadTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(MirrorPadTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, MirrorPad_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(MirrorPadTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(MirrorPadTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(MirrorPadTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, MirrorPad_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(MirrorPadTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(MirrorPadTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(MirrorPadTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Transpose)
{
  TEST_WITH_GRAPH(TransposeTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(TransposeTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(TransposeTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(TransposeTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(TransposeTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(TransposeTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Transpose_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(TransposeTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(TransposeTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(TransposeTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Transpose_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(TransposeTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(TransposeTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(TransposeTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Floor)
{
  TEST_WITH_GRAPH(FloorTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(FloorTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(FloorTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(FloorTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(FloorTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(FloorTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Floor_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(FloorTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(FloorTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(FloorTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Floor_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(FloorTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(FloorTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(FloorTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, GreaterEqual)
{
  TEST_WITH_GRAPH(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                  Granularity::LayerWise);
  TEST_WITH_GRAPH(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                  Granularity::ChannelWise);
  TEST_WITH_GRAPH(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::S16,
                  Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, GreaterEqual_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                       Granularity::LayerWise, Type::U8);
  TEST_WITH_WRONG_TYPE(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                       Granularity::ChannelWise, Type::U8);
  TEST_WITH_WRONG_TYPE(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::S16,
                       Granularity::ChannelWise, Type::S16);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, GreaterEqual_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                                     Granularity::LayerWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                                     Granularity::ChannelWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::S16,
                                     Granularity::ChannelWise, g.x());

  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                                     Granularity::LayerWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::U8,
                                     Granularity::ChannelWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreaterEqual>, Type::S16,
                                     Granularity::ChannelWise, g.y());
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Greater)
{
  TEST_WITH_GRAPH(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(ComparisonOpTestGraph<luci::CircleGreater>, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Greater_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8, Granularity::LayerWise,
                       Type::U8);
  TEST_WITH_WRONG_TYPE(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8,
                       Granularity::ChannelWise, Type::U8);
  TEST_WITH_WRONG_TYPE(ComparisonOpTestGraph<luci::CircleGreater>, Type::S16,
                       Granularity::ChannelWise, Type::S16);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Greater_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8,
                                     Granularity::LayerWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8,
                                     Granularity::ChannelWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreater>, Type::S16,
                                     Granularity::ChannelWise, g.x());

  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8,
                                     Granularity::LayerWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreater>, Type::U8,
                                     Granularity::ChannelWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleGreater>, Type::S16,
                                     Granularity::ChannelWise, g.y());
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, NotEqual)
{
  TEST_WITH_GRAPH(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, NotEqual_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8,
                       Granularity::LayerWise, Type::U8);
  TEST_WITH_WRONG_TYPE(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8,
                       Granularity::ChannelWise, Type::U8);
  TEST_WITH_WRONG_TYPE(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::S16,
                       Granularity::ChannelWise, Type::S16);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, NotEqual_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8,
                                     Granularity::LayerWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8,
                                     Granularity::ChannelWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::S16,
                                     Granularity::ChannelWise, g.x());

  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8,
                                     Granularity::LayerWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::U8,
                                     Granularity::ChannelWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(ComparisonOpTestGraph<luci::CircleNotEqual>, Type::S16,
                                     Granularity::ChannelWise, g.y());
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, OneHot)
{
  TEST_WITH_GRAPH(OneHotTestGraph<Type::S32>, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(OneHotTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(OneHotTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise);

  TEST_WITH_GRAPH(OneHotTestGraph<Type::S64>, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(OneHotTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(OneHotTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(OneHotTestGraph<Type::S32>, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(OneHotTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(OneHotTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(OneHotTestGraph<Type::S64>, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(OneHotTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(OneHotTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, OneHot_wrong_input_type_NEG)
{
  TEST_WITH_WRONG_TYPE(OneHotTestGraph<Type::S32>, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(OneHotTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(OneHotTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise, Type::U8);

  TEST_WITH_WRONG_TYPE(OneHotTestGraph<Type::S64>, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(OneHotTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(OneHotTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, OneHot_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(OneHotTestGraph<Type::S32>, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(OneHotTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(OneHotTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise);

  TEST_WITH_WRONG_GRANULARITY(OneHotTestGraph<Type::S64>, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(OneHotTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(OneHotTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Div)
{
  TEST_WITH_GRAPH(DivTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(DivTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(DivTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(DivTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(DivTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(DivTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Div_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(DivTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(DivTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(DivTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Div_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY_TARGET(DivTestGraph, Type::U8, Granularity::LayerWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(DivTestGraph, Type::U8, Granularity::ChannelWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(DivTestGraph, Type::S16, Granularity::ChannelWise, g.x());

  TEST_WITH_WRONG_GRANULARITY_TARGET(DivTestGraph, Type::U8, Granularity::LayerWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(DivTestGraph, Type::U8, Granularity::ChannelWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(DivTestGraph, Type::S16, Granularity::ChannelWise, g.y());
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, FloorDiv)
{
  TEST_WITH_GRAPH(FloorDivTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(FloorDivTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(FloorDivTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(FloorDivTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(FloorDivTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(FloorDivTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, FloorDiv_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(FloorDivTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(FloorDivTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(FloorDivTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, FloorDiv_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY_TARGET(FloorDivTestGraph, Type::U8, Granularity::LayerWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(FloorDivTestGraph, Type::U8, Granularity::ChannelWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(FloorDivTestGraph, Type::S16, Granularity::ChannelWise, g.x());

  TEST_WITH_WRONG_GRANULARITY_TARGET(FloorDivTestGraph, Type::U8, Granularity::LayerWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(FloorDivTestGraph, Type::U8, Granularity::ChannelWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(FloorDivTestGraph, Type::S16, Granularity::ChannelWise, g.y());
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Rsqrt)
{
  TEST_WITH_GRAPH(RsqrtTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(RsqrtTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(RsqrtTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(RsqrtTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(RsqrtTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(RsqrtTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Rsqrt_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(RsqrtTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(RsqrtTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(RsqrtTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Rsqrt_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(RsqrtTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(RsqrtTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(RsqrtTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Sqrt)
{
  TEST_WITH_GRAPH(SqrtTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(SqrtTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(SqrtTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(SqrtTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(SqrtTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(SqrtTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Sqrt_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(SqrtTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SqrtTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(SqrtTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Sqrt_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(SqrtTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(SqrtTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(SqrtTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Elu)
{
  TEST_WITH_GRAPH(EluTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(EluTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(EluTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(EluTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(EluTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(EluTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Elu_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(EluTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(EluTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(EluTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Elu_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(EluTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(EluTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(EluTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Pow)
{
  TEST_WITH_GRAPH(PowTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(PowTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(PowTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(PowTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(PowTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(PowTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Pow_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(PowTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(PowTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(PowTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Pow_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY_TARGET(PowTestGraph, Type::U8, Granularity::LayerWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(PowTestGraph, Type::U8, Granularity::ChannelWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(PowTestGraph, Type::S16, Granularity::ChannelWise, g.x());

  TEST_WITH_WRONG_GRANULARITY_TARGET(PowTestGraph, Type::U8, Granularity::LayerWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(PowTestGraph, Type::U8, Granularity::ChannelWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(PowTestGraph, Type::S16, Granularity::ChannelWise, g.y());
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, ReduceMax)
{
  TEST_WITH_GRAPH(ReduceMaxTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(ReduceMaxTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(ReduceMaxTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(ReduceMaxTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(ReduceMaxTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(ReduceMaxTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, ReduceMax_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(ReduceMaxTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(ReduceMaxTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(ReduceMaxTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, ReduceMax_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(ReduceMaxTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(ReduceMaxTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(ReduceMaxTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, ResizeBilinear)
{
  TEST_WITH_GRAPH(ResizeBilinearTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(ResizeBilinearTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(ResizeBilinearTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(ResizeBilinearTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(ResizeBilinearTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(ResizeBilinearTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, ResizeBilinear_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(ResizeBilinearTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(ResizeBilinearTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(ResizeBilinearTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, ResizeBilinear_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(ResizeBilinearTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(ResizeBilinearTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(ResizeBilinearTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, ResizeNearestNeighbor)
{
  TEST_WITH_GRAPH(ResizeNearestNeighborTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(ResizeNearestNeighborTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(ResizeNearestNeighborTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(ResizeBilinearTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(ResizeBilinearTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(ResizeBilinearTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, ResizeNearestNeighbor_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(ResizeNearestNeighborTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(ResizeNearestNeighborTestGraph, Type::U8, Granularity::ChannelWise,
                       Type::S16);
  TEST_WITH_WRONG_TYPE(ResizeNearestNeighborTestGraph, Type::S16, Granularity::ChannelWise,
                       Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, ResizeNearestNeighbor_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(ResizeNearestNeighborTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(ResizeNearestNeighborTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(ResizeNearestNeighborTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Unpack)
{
  TEST_WITH_GRAPH(UnpackTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(UnpackTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(UnpackTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Unpack_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(UnpackTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(UnpackTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(UnpackTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Unpack_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY(UnpackTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_WRONG_GRANULARITY(UnpackTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_WRONG_GRANULARITY(UnpackTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Add)
{
  TEST_WITH_GRAPH(AddTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(AddTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(AddTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(AddTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(AddTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(AddTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Add_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(AddTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(AddTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(AddTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Add_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY_TARGET(AddTestGraph, Type::U8, Granularity::LayerWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(AddTestGraph, Type::U8, Granularity::ChannelWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(AddTestGraph, Type::S16, Granularity::ChannelWise, g.x());

  TEST_WITH_WRONG_GRANULARITY_TARGET(AddTestGraph, Type::U8, Granularity::LayerWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(AddTestGraph, Type::U8, Granularity::ChannelWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(AddTestGraph, Type::S16, Granularity::ChannelWise, g.y());
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Add_inttype)
{
  // Tests for S32
  TEST_WITH_GRAPH(IntAddTestGraph<Type::S32>, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(IntAddTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(IntAddTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(IntAddTestGraph<Type::S32>, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(IntAddTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(IntAddTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise);

  // Tests for S64
  TEST_WITH_GRAPH(IntAddTestGraph<Type::S64>, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(IntAddTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(IntAddTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(IntAddTestGraph<Type::S64>, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(IntAddTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(IntAddTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise);

  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Mul)
{
  TEST_WITH_GRAPH(MulTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(MulTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(MulTestGraph, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(MulTestGraph, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(MulTestGraph, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(MulTestGraph, Type::S16, Granularity::ChannelWise);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Mul_wrong_type_NEG)
{
  TEST_WITH_WRONG_TYPE(MulTestGraph, Type::U8, Granularity::LayerWise, Type::S16);
  TEST_WITH_WRONG_TYPE(MulTestGraph, Type::U8, Granularity::ChannelWise, Type::S16);
  TEST_WITH_WRONG_TYPE(MulTestGraph, Type::S16, Granularity::ChannelWise, Type::U8);
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Mul_wrong_granularity_NEG)
{
  TEST_WITH_WRONG_GRANULARITY_TARGET(MulTestGraph, Type::U8, Granularity::LayerWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(MulTestGraph, Type::U8, Granularity::ChannelWise, g.x());
  TEST_WITH_WRONG_GRANULARITY_TARGET(MulTestGraph, Type::S16, Granularity::ChannelWise, g.x());

  TEST_WITH_WRONG_GRANULARITY_TARGET(MulTestGraph, Type::U8, Granularity::LayerWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(MulTestGraph, Type::U8, Granularity::ChannelWise, g.y());
  TEST_WITH_WRONG_GRANULARITY_TARGET(MulTestGraph, Type::S16, Granularity::ChannelWise, g.y());
  SUCCEED();
}

TEST(QuantizedModelVerifierTest, Mul_inttype)
{
  // Tests for S32
  TEST_WITH_GRAPH(IntMulTestGraph<Type::S32>, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(IntMulTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(IntMulTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(IntMulTestGraph<Type::S32>, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(IntMulTestGraph<Type::S32>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(IntMulTestGraph<Type::S32>, Type::S16, Granularity::ChannelWise);

  // Tests for S64
  TEST_WITH_GRAPH(IntMulTestGraph<Type::S64>, Type::U8, Granularity::LayerWise);
  TEST_WITH_GRAPH(IntMulTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_GRAPH(IntMulTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise);

  TEST_WITH_LAYER_INFO(IntMulTestGraph<Type::S64>, Type::U8, Granularity::LayerWise);
  TEST_WITH_LAYER_INFO(IntMulTestGraph<Type::S64>, Type::U8, Granularity::ChannelWise);
  TEST_WITH_LAYER_INFO(IntMulTestGraph<Type::S64>, Type::S16, Granularity::ChannelWise);

  SUCCEED();
}

// TODO Add following testcases
//
// CircleConv2D
//
// CircleDepthwiseConv2D
//
// CirclePRelu
//
// CircleTransposeConv
//
// CircleFullyConnected
//
// CircleAveragePool2D
//
// CircleMaxPool2D
//
// CircleMean
//
// CircleRelu
//
// CircleCast
//

#undef TEST_WITH_GRAPH
#undef TEST_WITH_WRONG_TYPE
#undef TEST_WITH_WRONG_GRANULARITY
