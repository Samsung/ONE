/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cker/eigen/EigenSupport.h>
#include <cker/operation/DepthwiseConv.h>

#include <gtest/gtest.h>
#include <vector>

namespace
{

template <typename T> class DepthwiseConvVerifier
{
public:
  DepthwiseConvVerifier() = default;

  void prepare(const nnfw::cker::Shape &output_shape, const nnfw::cker::Shape &filter_shape)
  {
    const int k_packet_size = nnfw::cker::eigen_support::kPacketSize<T>();
    const int batch = output_shape.Dims(0);
    const int out_depth = output_shape.Dims(3);
    const int filter_rows = filter_shape.Dims(1);
    const int filter_cols = filter_shape.Dims(2);
    const int filter_spatial_size = filter_rows * filter_cols;
    const int padded_filter_inner_dim_size =
      ((out_depth + k_packet_size - 1) / k_packet_size) * k_packet_size;

    _use_padded_filter = (out_depth % k_packet_size) == 0 ? false : true;
    {
      nnfw::cker::Shape padded_filter_shape(
        {batch, filter_spatial_size, padded_filter_inner_dim_size});
      _padded_filter.resize(padded_filter_shape.FlatSize());
    }

    {
      // NOTE The Eigen library uses both main thread as well as a thread pool.
      // Therefore, it needs to add an additional memory buffer for main thread.
      const int thread_count = nnfw::cker::eigen_support::getThreadCount() + 1;

      nnfw::cker::Shape filter_buffer_shape(
        {thread_count, filter_spatial_size, padded_filter_inner_dim_size});
      _filter_buffers.resize(filter_buffer_shape.FlatSize());
    }
  }

  void run(const nnfw::cker::DepthwiseConvParams &params, const nnfw::cker::Shape &input_shape,
           const T *input_data, const nnfw::cker::Shape &filter_shape, const T *filter_data,
           const nnfw::cker::Shape &bias_shape, const T *bias_data,
           const nnfw::cker::Shape &output_shape, const T *expected)
  {
    std::vector<T> output(output_shape.FlatSize());
    nnfw::cker::DepthwiseConvOp(params, input_shape, input_data, filter_shape, filter_data,
                                bias_shape, bias_data, _padded_filter.data(), _use_padded_filter,
                                _filter_buffers.data(), output_shape, output.data());

    for (size_t i = 0; i < output.size(); ++i)
      EXPECT_NEAR(output[i], expected[i], 1e-3f);
  }

  void checkException(const nnfw::cker::DepthwiseConvParams &params,
                      const nnfw::cker::Shape &input_shape, const T *input_data,
                      const nnfw::cker::Shape &filter_shape, const T *filter_data,
                      const nnfw::cker::Shape &bias_shape, const T *bias_data,
                      const nnfw::cker::Shape &output_shape, const T *expected)
  {
    std::vector<T> output(output_shape.FlatSize());
    EXPECT_ANY_THROW(
      nnfw::cker::DepthwiseConvOp(params, input_shape, input_data, filter_shape, filter_data,
                                  bias_shape, bias_data, _padded_filter.data(), _use_padded_filter,
                                  _filter_buffers.data(), output_shape, output.data()));
  }

private:
  bool _use_padded_filter;
  std::vector<T> _padded_filter;
  std::vector<T> _filter_buffers;
};

} // namespace

TEST(CKer_Operation, DepthwiseConv)
{
  {
    nnfw::cker::DepthwiseConvParams params{};
    params.padding_type = nnfw::cker::PaddingType::kValid;
    params.padding_values.width = 0;
    params.padding_values.height = 0;
    params.stride_width = 1;
    params.stride_height = 1;
    params.dilation_width_factor = 1;
    params.dilation_height_factor = 1;
    params.depth_multiplier = 1;
    params.float_activation_min = std::numeric_limits<float>::lowest();
    params.float_activation_max = std::numeric_limits<float>::max();

    nnfw::cker::Shape input_shape{1, 3, 2, 2}; // n, h, w, c
    std::vector<float> input = {1.0, 2.0, 7.0, 8.0, 3.0, 4.0, 9.0, 10.0, 5.0, 6.0, 11.0, 12.0};
    nnfw::cker::Shape filter_shape{1, 2, 2, 2}; // 1, h, w, c
    std::vector<float> filter = {1.0, 2.0, 3.0, 4.0, -9.0, 10.0, -11.0, 12.0};
    nnfw::cker::Shape bias_shape{2};
    std::vector<float> bias = {0.0, 0.0};
    nnfw::cker::Shape output_shape{1, 2, 1, 2}; // n, h, w, c
    std::vector<float> expected = {-104., 196.0, -136.0, 252.0};

    DepthwiseConvVerifier<float> verifier;
    verifier.prepare(output_shape, filter_shape);
    verifier.run(params, input_shape, input.data(), filter_shape, filter.data(), bias_shape,
                 bias.data(), output_shape, expected.data());
  }

  // Pad
  {
    nnfw::cker::DepthwiseConvParams params{};
    params.padding_type = nnfw::cker::PaddingType::kSame;
    params.padding_values.width = 0;
    params.padding_values.height = 1;
    params.stride_width = 1;
    params.stride_height = 1;
    params.dilation_width_factor = 1;
    params.dilation_height_factor = 1;
    params.depth_multiplier = 1;
    params.float_activation_min = std::numeric_limits<float>::lowest();
    params.float_activation_max = std::numeric_limits<float>::max();

    nnfw::cker::Shape input_shape{1, 2, 2, 2}; // n, h, w, c
    std::vector<float> input = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    nnfw::cker::Shape filter_shape{1, 3, 1, 2}; // 1, h, w, c
    std::vector<float> filter = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    nnfw::cker::Shape bias_shape{2};
    std::vector<float> bias = {0.0, 0.0};
    nnfw::cker::Shape output_shape{1, 2, 2, 2}; // n, h, w, c
    std::vector<float> expected = {16.0, 28.0, 28.0, 44.0, 8.0, 16.0, 12.0, 24.0};

    DepthwiseConvVerifier<float> verifier;
    verifier.prepare(output_shape, filter_shape);
    verifier.run(params, input_shape, input.data(), filter_shape, filter.data(), bias_shape,
                 bias.data(), output_shape, expected.data());
  }

  // Bias
  {
    nnfw::cker::DepthwiseConvParams params{};
    params.padding_type = nnfw::cker::PaddingType::kSame;
    params.padding_values.width = 0;
    params.padding_values.height = 1;
    params.stride_width = 1;
    params.stride_height = 1;
    params.dilation_width_factor = 1;
    params.dilation_height_factor = 1;
    params.depth_multiplier = 1;
    params.float_activation_min = std::numeric_limits<float>::lowest();
    params.float_activation_max = std::numeric_limits<float>::max();

    nnfw::cker::Shape input_shape{1, 2, 2, 2}; // n, h, w, c
    std::vector<float> input = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    nnfw::cker::Shape filter_shape{1, 3, 1, 2}; // 1, h, w, c
    std::vector<float> filter = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    nnfw::cker::Shape bias_shape{2};
    std::vector<float> bias = {0.5, -0.5};
    nnfw::cker::Shape output_shape{1, 2, 2, 2}; // n, h, w, c
    std::vector<float> expected = {16.5, 27.5, 28.5, 43.5, 8.5, 15.5, 12.5, 23.5};

    DepthwiseConvVerifier<float> verifier;
    verifier.prepare(output_shape, filter_shape);
    verifier.run(params, input_shape, input.data(), filter_shape, filter.data(), bias_shape,
                 bias.data(), output_shape, expected.data());
  }

  // Depth Multiplier
  {
    nnfw::cker::DepthwiseConvParams params{};
    params.padding_type = nnfw::cker::PaddingType::kSame;
    params.padding_values.width = 0;
    params.padding_values.height = 1;
    params.stride_width = 1;
    params.stride_height = 1;
    params.dilation_width_factor = 1;
    params.dilation_height_factor = 1;
    params.depth_multiplier = 2;
    params.float_activation_min = std::numeric_limits<float>::lowest();
    params.float_activation_max = std::numeric_limits<float>::max();

    nnfw::cker::Shape input_shape{1, 2, 2, 2}; // n, h, w, c
    std::vector<float> input = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    nnfw::cker::Shape filter_shape{1, 3, 1, 4}; // 1, h, w, c
    std::vector<float> filter = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0};
    nnfw::cker::Shape bias_shape{4};
    std::vector<float> bias = {0.5, -0.5, 0.3, -0.3};
    nnfw::cker::Shape output_shape{1, 2, 2, 4}; // n, h, w, c
    std::vector<float> expected = {-11.5, -8.5, -9.7,  -4.3,  -9.5, -2.5, -21.7, -12.3,
                                   16.5,  19.5, -22.7, -17.3, 24.5, 31.5, -28.7, -19.3};

    DepthwiseConvVerifier<float> verifier;
    verifier.prepare(output_shape, filter_shape);
    verifier.run(params, input_shape, input.data(), filter_shape, filter.data(), bias_shape,
                 bias.data(), output_shape, expected.data());
  }

  // ReLU6
  {
    nnfw::cker::DepthwiseConvParams params{};
    params.padding_type = nnfw::cker::PaddingType::kSame;
    params.padding_values.width = 0;
    params.padding_values.height = 1;
    params.stride_width = 1;
    params.stride_height = 1;
    params.dilation_width_factor = 1;
    params.dilation_height_factor = 1;
    params.depth_multiplier = 1;
    params.float_activation_min = 0.0;
    params.float_activation_max = 6.0;

    nnfw::cker::Shape input_shape{1, 2, 2, 2}; // n, h, w, c
    std::vector<float> input = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    nnfw::cker::Shape filter_shape{1, 3, 1, 2}; // 1, h, w, c
    std::vector<float> filter = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    nnfw::cker::Shape bias_shape{2};
    std::vector<float> bias = {0.5, -0.5};
    nnfw::cker::Shape output_shape{1, 2, 2, 2}; // n, h, w, c
    std::vector<float> expected = {6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0};

    DepthwiseConvVerifier<float> verifier;
    verifier.prepare(output_shape, filter_shape);
    verifier.run(params, input_shape, input.data(), filter_shape, filter.data(), bias_shape,
                 bias.data(), output_shape, expected.data());
  }

  // No bias
  {
    nnfw::cker::DepthwiseConvParams params{};
    params.padding_type = nnfw::cker::PaddingType::kSame;
    params.padding_values.width = 0;
    params.padding_values.height = 1;
    params.stride_width = 1;
    params.stride_height = 1;
    params.dilation_width_factor = 1;
    params.dilation_height_factor = 1;
    params.depth_multiplier = 1;
    params.float_activation_min = std::numeric_limits<float>::lowest();
    params.float_activation_max = std::numeric_limits<float>::max();

    nnfw::cker::Shape input_shape{1, 2, 2, 2}; // n, h, w, c
    std::vector<float> input = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    nnfw::cker::Shape filter_shape{1, 3, 1, 2}; // 1, h, w, c
    std::vector<float> filter = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    nnfw::cker::Shape bias_shape{2};
    nnfw::cker::Shape output_shape{1, 2, 2, 2}; // n, h, w, c
    std::vector<float> expected = {16.0, 28.0, 28.0, 44.0, 8.0, 16.0, 12.0, 24.0};

    DepthwiseConvVerifier<float> verifier;
    verifier.prepare(output_shape, filter_shape);
    verifier.run(params, input_shape, input.data(), filter_shape, filter.data(), bias_shape,
                 nullptr, output_shape, expected.data());
  }
}

TEST(CKer_Operation, neg_DepthwiseConv)
{
  // Not supported Dilation
  {
    nnfw::cker::DepthwiseConvParams params{};
    params.padding_type = nnfw::cker::PaddingType::kSame;
    params.padding_values.width = 0;
    params.padding_values.height = 1;
    params.stride_width = 1;
    params.stride_height = 1;
    params.dilation_width_factor = 2;
    params.dilation_height_factor = 2;
    params.depth_multiplier = 1;
    params.float_activation_min = std::numeric_limits<float>::lowest();
    params.float_activation_max = std::numeric_limits<float>::max();

    nnfw::cker::Shape input_shape{1, 6, 6, 1}; // n, h, w, c
    std::vector<float> input = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    nnfw::cker::Shape filter_shape{1, 2, 2, 1}; // 1, h, w, c
    std::vector<float> filter = {1.0, 2.0, 3.0, 4.0};
    nnfw::cker::Shape bias_shape{1};
    std::vector<float> bias = {0.0};
    nnfw::cker::Shape output_shape{1, 3, 3, 1}; // n, h, w, c
    std::vector<float> expected = {4.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0};

    DepthwiseConvVerifier<float> verifier;
    verifier.prepare(output_shape, filter_shape);
    verifier.checkException(params, input_shape, input.data(), filter_shape, filter.data(),
                            bias_shape, bias.data(), output_shape, expected.data());
  }
}
