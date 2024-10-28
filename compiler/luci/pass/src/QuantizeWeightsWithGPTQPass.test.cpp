#include "luci/Pass/QuantizeWeightsWithGPTQPass.h"
#include <luci/IR/CircleNodes.h>
#include <gtest/gtest.h>
#include <unordered_map>

namespace
{
struct QuantizeWeightsWithGPTQPassTest : public ::testing::Test
{
  /**
   *  nconv graph
   *
   *        [CircleInput]
   *              |
   *              |
   *        [CircleConv2D]
   *              |
   *              |
   *        [CircleOutput]
   */
  void MakeGraph()
  {
    const int N = 1;
    const int H = 4;
    const int W = 4;
    const int C = 3; // IC = OC

    // graph input and output
    auto graph_input = _g.inputs()->create();
    auto graph_output = _g.outputs()->create();

    // CircleInput
    auto input = _g.nodes()->create<luci::CircleInput>();
    input->index(graph_input->index());
    input->shape({N, H, W, C});
    input->dtype(loco::DataType::FLOAT32);
    input->name("input");

    // CircleConv2D
    auto conv = _g.nodes()->create<luci::CircleConv2D>();
    conv->input(input);
    auto bias = _g.nodes()->create<luci::CircleConst>();
    bias->dtype(loco::DataType::FLOAT32);
    bias->shape({C});
    bias->name("conv_bias");
    conv->bias(bias);
    auto weight = _g.nodes()->create<luci::CircleConst>();
    weight->dtype(loco::DataType::FLOAT32);
    weight->shape({C, H, W, C});
    weight->size<loco::DataType::FLOAT32>(C * H * W * C);
    weight->name("nconv/filter");
    conv->filter(weight);
    conv->padding(luci::Padding::SAME);
    conv->fusedActivationFunction(luci::FusedActFunc::NONE);
    conv->dtype(loco::DataType::FLOAT32);
    conv->name("nconv");

    // CircleOutput
    auto output = _g.nodes()->create<luci::CircleOutput>();
    output->index(graph_output->index());
    output->from(conv);
    output->shape({N, H, W, C});
    output->dtype(loco::DataType::FLOAT32);
    output->name("output");
  }
  virtual void SetUp() { MakeGraph(); }
  loco::Graph _g;
};
} // namespace

TEST_F(QuantizeWeightsWithGPTQPassTest, name)
{
  luci::QuantizeWeightsWithGPTQPass pass(loco::DataType::FLOAT32, loco::DataType::U8,
                                         luci::QuantizationGranularity::ChannelWise);
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(QuantizeWeightsWithGPTQPassTest, name_ctx)
{
  auto ctx = std::make_unique<luci::QuantizeWeightsWithGPTQPass::Context>();
  {
    ctx->input_model_dtype = loco::DataType::FLOAT32;
    ctx->output_model_dtype = loco::DataType::U8;
    ctx->granularity = luci::QuantizationGranularity::ChannelWise;
  }

  luci::QuantizeWeightsWithGPTQPass pass(std::move(ctx));
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

// Negative test: Unsupported granularity - Invalid value
TEST_F(QuantizeWeightsWithGPTQPassTest, run_granularity_invalid_NEG)
{
  auto invalid_granularity = static_cast<luci::QuantizationGranularity>(999);
  luci::QuantizeWeightsWithGPTQPass pass(loco::DataType::FLOAT32, loco::DataType::U8, invalid_granularity);
  ASSERT_EXIT(pass.run(&_g),::testing::KilledBySignal(SIGSEGV),".*");
}

// Negative test: Unsupported output data type - FLOAT32
TEST_F(QuantizeWeightsWithGPTQPassTest, run_output_f32_NEG)
{
  luci::QuantizeWeightsWithGPTQPass pass(loco::DataType::FLOAT32, loco::DataType::FLOAT32,
                                         luci::QuantizationGranularity::ChannelWise);
  // Since output type is FLOAT32, an exception is expected
  EXPECT_THROW(pass.run(&_g), std::runtime_error);
}

// Negative test: Provide an empty hessian map
TEST_F(QuantizeWeightsWithGPTQPassTest, run_with_empty_hessian_map_NEG)
{
  std::unordered_map<const luci::CircleNode *, std::vector<float>> hessian_map;
  auto ctx = std::make_unique<luci::QuantizeWeightsWithGPTQPass::Context>();
  {
    ctx->input_model_dtype = loco::DataType::FLOAT32;
    ctx->output_model_dtype = loco::DataType::U8;
    ctx->granularity = luci::QuantizationGranularity::ChannelWise;
  }

  luci::QuantizeWeightsWithGPTQPass pass(std::move(ctx), &hessian_map);
  // Expect no exception, pass should handle empty hessian map gracefully
  EXPECT_NO_THROW(pass.run(&_g));
}

// Negative test: Weights are not of type FLOAT32
TEST_F(QuantizeWeightsWithGPTQPassTest, run_with_non_float_weights_NEG)
{
  // Change the weights to non-float32
  luci::CircleConst *weight = nullptr;
  for (auto node : loco::all_nodes(&_g))
  {
    if (auto const_node = dynamic_cast<luci::CircleConst *>(node))
    {
      if (const_node->name() == "nconv/filter")
      {
        weight = const_node;
        break;
      }
    }
  }
  ASSERT_NE(weight, nullptr);
  // Set dtype to INT32
  weight->dtype(loco::DataType::S32);

  luci::QuantizeWeightsWithGPTQPass pass(loco::DataType::FLOAT32, loco::DataType::U8,
                                         luci::QuantizationGranularity::ChannelWise);
  // The pass should skip this node without exception
  EXPECT_NO_THROW(pass.run(&_g));
}

// Positive test: Run pass with valid hessian map
TEST_F(QuantizeWeightsWithGPTQPassTest, run_with_valid_hessian)
{
  // Create a hessian map with valid data
  std::unordered_map<const luci::CircleNode *, std::vector<float>> hessian_map;
  // Find the conv node
  luci::CircleConv2D *conv = nullptr;
  for (auto node : loco::all_nodes(&_g))
  {
    if (auto conv_node = dynamic_cast<luci::CircleConv2D *>(node))
    {
      conv = conv_node;
      break;
    }
  }
  ASSERT_NE(conv, nullptr);
  const auto node_filter = loco::must_cast<luci::CircleConst *>(
        loco::must_cast<const luci::CircleConv2D *>(conv)->filter());
  // Create a dummy hessian vector
  size_t weight_size = node_filter->size<loco::DataType::FLOAT32>();
  std::vector<float> hessian(weight_size * weight_size, 0.0f);
  for (size_t i = 0; i < weight_size; ++i)
  {
    hessian[i * weight_size + i] = 1.0f; // Identity matrix
  }

  hessian_map[conv] = hessian;

  auto ctx = std::make_unique<luci::QuantizeWeightsWithGPTQPass::Context>();
  {
    ctx->input_model_dtype = loco::DataType::FLOAT32;
    ctx->output_model_dtype = loco::DataType::U8;
    ctx->granularity = luci::QuantizationGranularity::ChannelWise;
  }

  luci::QuantizeWeightsWithGPTQPass pass(std::move(ctx), &hessian_map);
  EXPECT_NO_THROW(pass.run(&_g));
}

// Negative test: Input model data type is U8 (unsupported)
TEST_F(QuantizeWeightsWithGPTQPassTest, run_input_U8_NEG)
{
  luci::QuantizeWeightsWithGPTQPass pass(loco::DataType::U8, loco::DataType::U8,
                                         luci::QuantizationGranularity::ChannelWise);
  EXPECT_THROW(pass.run(&_g), std::runtime_error);
}

// Negative test: Output model data type is S32 (unsupported)
TEST_F(QuantizeWeightsWithGPTQPassTest, run_output_S32_NEG)
{
  luci::QuantizeWeightsWithGPTQPass pass(loco::DataType::FLOAT32, loco::DataType::S32,
                                         luci::QuantizationGranularity::ChannelWise);
  EXPECT_THROW(pass.run(&_g), std::runtime_error);
}
