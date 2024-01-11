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

#include <cker/train/operation/DepthwiseConv.h>

#include <gtest/gtest.h>
#include <vector>
#include <memory>

namespace
{

template <typename T> class DepthwiseConvVerifier
{
public:
  DepthwiseConvVerifier() : _dconv_kernel{new nnfw::cker::train::DepthwiseConv()}
  {
    _dconv_kernel = std::make_unique<nnfw::cker::train::DepthwiseConv>();
  }

  void prepare(const nnfw::cker::Shape &incoming_shape, const nnfw::cker::Shape &filter_shape)
  {
    const int kPacketSize = _dconv_kernel->kPacketSize<T>();
    const int batch = incoming_shape.Dims(0);
    const int out_depth = incoming_shape.Dims(3);
    const int filter_rows = filter_shape.Dims(1);
    const int filter_cols = filter_shape.Dims(2);
    const int filter_spatial_size = filter_rows * filter_cols;
    const int padded_filter_inner_dim_size =
      ((out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;

    _use_padded_filter = (out_depth % kPacketSize) == 0 ? false : true;
    {
      nnfw::cker::Shape padded_filter_shape(
        {batch, filter_spatial_size, padded_filter_inner_dim_size});
      _padded_filter.resize(padded_filter_shape.FlatSize());
    }

    {
      const int thread_count = _dconv_kernel->getThreadCount() + 1;
      nnfw::cker::Shape filter_buffer_shape(
        {thread_count, filter_spatial_size, padded_filter_inner_dim_size});
      _filter_buffers.resize(filter_buffer_shape.FlatSize());

      nnfw::cker::Shape filter_dim_buffer_shape({thread_count, padded_filter_inner_dim_size});
      _filter_dim_buffers.resize(filter_dim_buffer_shape.FlatSize());
      // for (auto i = 0; i < thread_count; ++i)
      // {
      //   _out_bprop.emplace_back(new uint8_t[padded_filter_shape.FlatSize() * sizeof(T)]);
      //   _in_bprop.emplace_back(new uint8_t[in_bprop_shape.FlatSize() * sizeof(T)]);
      // }
    }
  }

  void verifyInputGradExpected(const nnfw::cker::DepthwiseConvParams &params,
                               const nnfw::cker::Shape &incoming_shape, const T *incoming_data,
                               const nnfw::cker::Shape &filter_shape, const T *filter_data,
                               const nnfw::cker::Shape &grad_shape)
  {
    std::vector<T> gradient(grad_shape.FlatSize(), static_cast<T>(0));
    std::vector<T> expected(grad_shape.FlatSize(), static_cast<T>(0));

    calculateInputGradExpected(params, incoming_shape, incoming_data, filter_shape, filter_data,
                               grad_shape, expected.data());

    _dconv_kernel->backpropInput(params, incoming_shape, incoming_data, filter_shape, filter_data,
                                 _padded_filter.data(), grad_shape, gradient.data(),
                                 _use_padded_filter, _filter_buffers.data(),
                                 _filter_dim_buffers.data());

    for (size_t i = 0; i < gradient.size(); ++i)
      EXPECT_NEAR(gradient[i], expected[i], 1e-4f);
  }

  void throwInputGradExpected(const nnfw::cker::DepthwiseConvParams &params,
                              const nnfw::cker::Shape &incoming_shape, const T *incoming_data,
                              const nnfw::cker::Shape &filter_shape, const T *filter_data,
                              const nnfw::cker::Shape &grad_shape)
  {
    std::vector<T> gradient(grad_shape.FlatSize(), static_cast<T>(0));

    EXPECT_ANY_THROW(_dconv_kernel->backpropInput(
      params, incoming_shape, incoming_data, filter_shape, filter_data, _padded_filter.data(),
      grad_shape, gradient.data(), _use_padded_filter, _filter_buffers.data(),
      _filter_dim_buffers.data()));
  }

  void verifyFilterGradExpected(const nnfw::cker::DepthwiseConvParams &params,
                                const nnfw::cker::Shape &incoming_shape, const T *incoming_data,
                                const nnfw::cker::Shape &input_shape, const T *input_data,
                                const nnfw::cker::Shape &filter_grad_shape)
  {
    std::vector<T> gradient(filter_grad_shape.FlatSize(), static_cast<T>(0));
    std::vector<T> expected(filter_grad_shape.FlatSize(), static_cast<T>(0));

    calculateFilterGradExpected(params, incoming_shape, incoming_data, input_shape, input_data,
                                filter_grad_shape, expected.data());

    _dconv_kernel->backpropFilter(params, incoming_shape, incoming_data, input_shape, input_data,
                                  filter_grad_shape, gradient.data(), _padded_filter.data(),
                                  _filter_buffers.data());

    for (size_t i = 0; i < gradient.size(); ++i)
      EXPECT_NEAR(gradient[i], expected[i], 1e-3f);
  }

  void throwFilterGradExpected(const nnfw::cker::DepthwiseConvParams &params,
                               const nnfw::cker::Shape &incoming_shape, const T *incoming_data,
                               const nnfw::cker::Shape &input_shape, const T *input_data,
                               const nnfw::cker::Shape &filter_grad_shape)
  {
    std::vector<T> gradient(filter_grad_shape.FlatSize(), static_cast<T>(0));

    EXPECT_ANY_THROW(_dconv_kernel->backpropFilter(
      params, incoming_shape, incoming_data, input_shape, input_data, filter_grad_shape,
      gradient.data(), _padded_filter.data(), _filter_buffers.data()));
  }

private:
  void calculateInputGradExpected(const nnfw::cker::DepthwiseConvParams &params,
                                  const nnfw::cker::Shape &incoming_shape, const T *incoming_data,
                                  const nnfw::cker::Shape &filter_shape, const T *filter_data,
                                  const nnfw::cker::Shape &grad_shape, T *expected)
  {
    assert(incoming_shape.DimensionsCount() == 4);
    assert(filter_shape.DimensionsCount() == 4);
    assert(grad_shape.DimensionsCount() == 4);
    assert(params.stride_height == params.stride_width);

    const int batch = MatchingDim(incoming_shape, 0, grad_shape, 0);
    const int input_depth = grad_shape.Dims(3);
    const int output_depth = incoming_shape.Dims(3);
    const int incoming_height = incoming_shape.Dims(1);
    const int incoming_width = incoming_shape.Dims(2);
    const int grad_height = grad_shape.Dims(1);
    const int grad_width = grad_shape.Dims(2);
    const int stride = params.stride_height;
    const int depth_multiplier = params.depth_multiplier;
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int pad_height = params.padding_values.height;
    const int pad_width = params.padding_values.width;

    nnfw::cker::depthwise_conv_op::DepthwiseConvBackpropInputReference<T>(
      batch, grad_height, grad_width, input_depth, incoming_height, incoming_width, output_depth,
      stride, depth_multiplier, filter_height, filter_width, pad_height, pad_width, incoming_data,
      filter_data, expected);
  }

  void calculateFilterGradExpected(const nnfw::cker::DepthwiseConvParams &params,
                                   const nnfw::cker::Shape &incoming_shape, const T *incoming_data,
                                   const nnfw::cker::Shape &input_shape, const T *input_data,
                                   const nnfw::cker::Shape &filter_grad_shape, T *expected)
  {
    assert(incoming_shape.DimensionsCount() == 4);
    assert(input_shape.DimensionsCount() == 4);
    assert(filter_grad_shape.DimensionsCount() == 4);
    assert(params.stride_height == params.stride_width);

    const int batch = MatchingDim(incoming_shape, 0, input_shape, 0);
    const int input_depth = input_shape.Dims(3);
    const int output_depth = incoming_shape.Dims(3);
    const int incoming_height = incoming_shape.Dims(1);
    const int incoming_width = incoming_shape.Dims(2);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int stride = params.stride_height;
    const int depth_multiplier = params.depth_multiplier;
    const int filter_height = filter_grad_shape.Dims(1);
    const int filter_width = filter_grad_shape.Dims(2);
    const int pad_height = params.padding_values.height;
    const int pad_width = params.padding_values.width;

    nnfw::cker::depthwise_conv_op::DepthwiseConvBackpropFilterReference<T>(
      batch, input_height, input_width, input_depth, incoming_height, incoming_width, output_depth,
      stride, depth_multiplier, filter_height, filter_width, pad_height, pad_width, incoming_data,
      input_data, expected);
  }

private:
  std::unique_ptr<nnfw::cker::train::DepthwiseConv> _dconv_kernel;
  bool _use_padded_filter;
  std::vector<T> _padded_filter;
  std::vector<T> _filter_buffers;
  std::vector<T> _filter_dim_buffers;
  // std::vector<uint8_t *> _out_bprop;
  // std::vector<uint8_t *> _in_bprop;
};

} // namespace

TEST(CKer_Operation, DepthwiseConvGrad)
{
  // No pad, No stride
  // {
  //   nnfw::cker::DepthwiseConvParams params;
  //   params.padding_type = nnfw::cker::PaddingType::kNone;
  //   params.padding_values.width = 0;
  //   params.padding_values.height = 0;
  //   params.stride_width = 1;
  //   params.stride_height = 1;
  //   params.dilation_width_factor = 1;
  //   params.dilation_height_factor = 1;
  //   params.depth_multiplier = 1;

  //   nnfw::cker::Shape incoming_shape{1, 2, 2, 3}; // n, h, w, c
  //   std::vector<float> incoming = {-0.1, 0.2, -0.3, 0.4,  0.5, -0.6,
  //                                  -0.7, 0.8, 0.9,  -1.0, 1.1, -1.2};
  //   nnfw::cker::Shape filter_shape{1, 2, 2, 3}; // 1, h, w, c
  //   std::vector<float> filter = {-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12};
  //   nnfw::cker::Shape input_shape{1, 3, 3, 3}; // n, h, w, c
  //   std::vector<float> input = {-1, 2,   -3,  4,  5,  -6,  -7, 8,   -9,  -10, 11,  -12, 13, -14,
  //                               15, -16, -17, 18, 19, -20, 21, -22, -23, 24,  -25, -26, -27};

  //   DepthwiseConvVerifier<float> verifier;
  //   verifier.prepare(incoming_shape, filter_shape);
  //   verifier.verifyInputGradExpected(params, incoming_shape, incoming.data(), filter_shape,
  //                                    filter.data(), input_shape);
  //   verifier.verifyFilterGradExpected(params, incoming_shape, incoming.data(), input_shape,
  //                                     input.data(), filter_shape);
  // }

  // // 2 depth_multiplier, use_padded_filter false
  // {
  //   nnfw::cker::DepthwiseConvParams params;
  //   params.padding_type = nnfw::cker::PaddingType::kNone;
  //   params.padding_values.width = 0;
  //   params.padding_values.height = 0;
  //   params.stride_width = 1;
  //   params.stride_height = 1;
  //   params.dilation_width_factor = 1;
  //   params.dilation_height_factor = 1;
  //   params.depth_multiplier = 2;

  //   nnfw::cker::Shape incoming_shape{1, 2, 2, 4}; // n, h, w, c
  //   std::vector<float> incoming = {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8,
  //                                  -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8};
  //   nnfw::cker::Shape filter_shape{1, 2, 2, 4}; // 1, h, w, c * depth_multiplier
  //   std::vector<float> filter = {-1, 2, -3, 4, 5, -6, 7, -8, 9, -10, -11, 12, -13, 14, -15, 16};
  //   nnfw::cker::Shape input_shape{1, 3, 3, 2}; // n, h, w, c
  //   std::vector<float> input = {-1,  2,  -3,  4,  5,   -6, -7,  8,   -9,
  //                               -10, 11, -12, 13, -14, 15, -16, -17, 18};

  //   DepthwiseConvVerifier<float> verifier;
  //   verifier.prepare(incoming_shape, filter_shape);
  //   verifier.verifyInputGradExpected(params, incoming_shape, incoming.data(), filter_shape,
  //                                    filter.data(), input_shape);
  //   verifier.verifyFilterGradExpected(params, incoming_shape, incoming.data(), input_shape,
  //                                     input.data(), filter_shape);
  // }

  // // pad valid, stride 2
  // {
  //   nnfw::cker::DepthwiseConvParams params;
  //   params.padding_type = nnfw::cker::PaddingType::kValid;
  //   params.stride_width = 2;
  //   params.stride_height = 2;
  //   params.dilation_width_factor = 1;
  //   params.dilation_height_factor = 1;
  //   params.depth_multiplier = 1;

  //   nnfw::cker::Shape incoming_shape{1, 3, 3, 2}; // n, h, w, c
  //   std::vector<float> incoming = {-0.1, 0.2, -0.3, 0.4, 0.5,  -0.6, -0.7, 0.8, 0.9,
  //                                  -1.0, 1.1, -1.2, 1.3, -1.4, -1.5, 1.6,  1.7, -1.8};
  //   nnfw::cker::Shape filter_shape{1, 3, 3, 2}; // 1, h, w, c
  //   std::vector<float> filter = {-1,  2,   -3, 4,   5,  -6,  -7, 8,  9,
  //                                -10, -11, 12, -13, 14, -15, 16, 17, -18};
  //   nnfw::cker::Shape input_shape{1, 3, 3, 2}; // n, h, w, c
  //   std::vector<float> input = {-1,  2,  -3,  4,  5,   -6, -7,  8,   -9,
  //                               -10, 11, -12, 13, -14, 15, -16, -17, 18};

  //   DepthwiseConvVerifier<float> verifier;
  //   verifier.prepare(incoming_shape, filter_shape);
  //   verifier.verifyInputGradExpected(params, incoming_shape, incoming.data(), filter_shape,
  //                                    filter.data(), input_shape);
  //   verifier.verifyFilterGradExpected(params, incoming_shape, incoming.data(), input_shape,
  //                                     input.data(), filter_shape);
  // }

  // // pad same, stride 2
  // {
  //   nnfw::cker::DepthwiseConvParams params;
  //   params.padding_type = nnfw::cker::PaddingType::kSame;
  //   params.stride_width = 2;
  //   params.stride_height = 2;
  //   params.dilation_width_factor = 1;
  //   params.dilation_height_factor = 1;
  //   params.depth_multiplier = 1;

  //   nnfw::cker::Shape incoming_shape{1, 1, 1, 2}; // n, h, w, c
  //   std::vector<float> incoming = {-0.1, 0.2};
  //   nnfw::cker::Shape filter_shape{1, 2, 2, 2}; // 1, h, w, c
  //   std::vector<float> filter = {-1, 2, -3, 4, 5, -6, -7, 8};
  //   nnfw::cker::Shape input_shape{1, 3, 3, 2}; // n, h, w, c
  //   std::vector<float> input = {-1,  2,  -3,  4,  5,   -6, -7,  8,   -9,
  //                               -10, 11, -12, 13, -14, 15, -16, -17, 18};

  //   DepthwiseConvVerifier<float> verifier;
  //   verifier.prepare(incoming_shape, filter_shape);
  //   verifier.verifyInputGradExpected(params, incoming_shape, incoming.data(), filter_shape,
  //                                    filter.data(), input_shape);
  //   verifier.verifyFilterGradExpected(params, incoming_shape, incoming.data(), input_shape,
  //                                     input.data(), filter_shape);
  // }

  // multi thread case
  {
    nnfw::cker::DepthwiseConvParams params;
    params.padding_type = nnfw::cker::PaddingType::kNone;
    params.padding_values.width = 0;
    params.padding_values.height = 0;
    params.stride_width = 1;
    params.stride_height = 1;
    params.dilation_width_factor = 1;
    params.dilation_height_factor = 1;
    params.depth_multiplier = 1;

    nnfw::cker::Shape incoming_shape{10, 112, 112, 32}; // n, h, w, c
    std::vector<float> incoming;
    for (int i = 0; i < incoming_shape.FlatSize(); ++i)
    {
      incoming.push_back(static_cast<float>(i) / static_cast<float>(RAND_MAX));
    }
    nnfw::cker::Shape filter_shape{1, 3, 3, 32}; // 1, h, w, c
    std::vector<float> filter;
    for (int i = 0; i < filter_shape.FlatSize(); ++i)
    {
      filter.push_back(static_cast<float>(i) / static_cast<float>(RAND_MAX / 100));
    }
    nnfw::cker::Shape input_shape{10, 112, 112, 32}; // n, h, w, c
    std::vector<float> input;
    const int input_size = input_shape.FlatSize();
    for (int i = 0; i < input_size; ++i)
    {
      input.push_back(static_cast<float>(input_size - i) / static_cast<float>(RAND_MAX / 100));
    }

    DepthwiseConvVerifier<float> verifier;
    verifier.prepare(incoming_shape, filter_shape);
    verifier.verifyInputGradExpected(params, incoming_shape, incoming.data(), filter_shape,
                                     filter.data(), input_shape);
    verifier.verifyFilterGradExpected(params, incoming_shape, incoming.data(), input_shape,
                                      input.data(), filter_shape);
  }
}

TEST(CKer_Operation, neg_DepthwiseConvGrad)
{
  // // Not matched stride, InputGrad test case
  // {
  //   nnfw::cker::DepthwiseConvParams params;
  //   params.padding_type = nnfw::cker::PaddingType::kNone;
  //   params.padding_values.width = 0;
  //   params.padding_values.height = 0;
  //   params.stride_width = 1;
  //   params.stride_height = 2;
  //   params.dilation_width_factor = 1;
  //   params.dilation_height_factor = 1;
  //   params.depth_multiplier = 1;

  //   nnfw::cker::Shape incoming_shape{1, 2, 2, 4}; // n, h, w, c
  //   std::vector<float> incoming = {-0.1, 0.2,  -0.3, 0.4,  0.5,  -0.6, -0.7, 0.8,
  //                                  0.9,  -1.0, 1.1,  -1.2, -1.3, 1.4,  1.5,  -1.6};
  //   nnfw::cker::Shape filter_shape{1, 2, 2, 4}; // 1, h, w, c
  //   std::vector<float> filter = {-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12};
  //   nnfw::cker::Shape input_shape{1, 3, 3, 4}; // n, h, w, c
  //   std::vector<float> input = {-1,  2,   -3,  4,   5,   -6,  -7, 8,   -9,  -10, 11,  -12,
  //                               13,  -14, 15,  -16, -17, 18,  19, -20, 21,  -22, -23, 24,
  //                               -25, -26, -27, 28,  -29, -30, 31, 32,  -33, -34, -35, 36};

  //   DepthwiseConvVerifier<float> verifier;
  //   verifier.prepare(incoming_shape, filter_shape);
  //   verifier.throwInputGradExpected(params, incoming_shape, incoming.data(), filter_shape,
  //                                   filter.data(), input_shape);
  // }

  // // Not matched stride, FilterGrad test case
  // {
  //   nnfw::cker::DepthwiseConvParams params;
  //   params.padding_type = nnfw::cker::PaddingType::kNone;
  //   params.padding_values.width = 0;
  //   params.padding_values.height = 0;
  //   params.stride_width = 1;
  //   params.stride_height = 2;
  //   params.dilation_width_factor = 1;
  //   params.dilation_height_factor = 1;
  //   params.depth_multiplier = 1;

  //   nnfw::cker::Shape incoming_shape{1, 2, 2, 4}; // n, h, w, c
  //   std::vector<float> incoming = {-0.1, 0.2,  -0.3, 0.4,  0.5,  -0.6, -0.7, 0.8,
  //                                  0.9,  -1.0, 1.1,  -1.2, -1.3, 1.4,  1.5,  -1.6};
  //   nnfw::cker::Shape filter_shape{1, 2, 2, 4}; // 1, h, w, c
  //   std::vector<float> filter = {-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12};
  //   nnfw::cker::Shape input_shape{1, 3, 3, 4}; // n, h, w, c
  //   std::vector<float> input = {-1,  2,   -3,  4,   5,   -6,  -7, 8,   -9,  -10, 11,  -12,
  //                               13,  -14, 15,  -16, -17, 18,  19, -20, 21,  -22, -23, 24,
  //                               -25, -26, -27, 28,  -29, -30, 31, 32,  -33, -34, -35, 36};

  //   DepthwiseConvVerifier<float> verifier;
  //   verifier.prepare(incoming_shape, filter_shape);
  //   verifier.throwFilterGradExpected(params, incoming_shape, incoming.data(), filter_shape,
  //                                    filter.data(), input_shape);
  // }
}
