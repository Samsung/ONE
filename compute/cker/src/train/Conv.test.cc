/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <cker/train/operation/Conv.h>

#include <gtest/gtest.h>
#include <vector>

namespace
{

template <typename T> class ConvVerifier
{
public:
  static void verifyFilterGradExpected(const nnfw::cker::ConvParams &params,
                                       const nnfw::cker::Shape &incoming_shape,
                                       const T *incoming_data, const nnfw::cker::Shape &input_shape,
                                       const T *input_data, int padding_bottom, int padding_right,
                                       const nnfw::cker::Shape &filter_shape)
  {
    std::vector<T> gradient(filter_shape.FlatSize(), static_cast<T>(0));
    std::vector<T> expected(filter_shape.FlatSize(), static_cast<T>(0));

    calculateFilterGradExpected(params, incoming_shape, incoming_data, input_shape, input_data,
                                filter_shape, expected.data());

    nnfw::cker::train::ConvFilterGrad(params, incoming_shape, incoming_data, input_shape,
                                      input_data, padding_bottom, padding_right, filter_shape,
                                      gradient.data());

    for (size_t i = 0; i < gradient.size(); ++i)
      EXPECT_NEAR(gradient[i], expected[i], 1e-4f);
  }

  static void verifyInputGradExpected(const nnfw::cker::ConvParams &params,
                                      const nnfw::cker::Shape &incoming_shape,
                                      const T *incoming_data, const nnfw::cker::Shape &filter_shape,
                                      const T *filter_data, int padding_bottom, int padding_right,
                                      const nnfw::cker::Shape &input_shape)
  {
    std::vector<T> gradient(input_shape.FlatSize(), static_cast<T>(0));
    std::vector<T> expected(input_shape.FlatSize(), static_cast<T>(0));

    calculateInputGradExpected(params, incoming_shape, incoming_data, filter_shape, filter_data,
                               input_shape, expected.data());

    nnfw::cker::train::ConvInputGrad(params, incoming_shape, incoming_data, filter_shape,
                                     filter_data, padding_bottom, padding_right, input_shape,
                                     gradient.data());

    for (size_t i = 0; i < gradient.size(); ++i)
      EXPECT_NEAR(gradient[i], expected[i], 1e-4f);
  }

private:
  static void calculateFilterGradExpected(const nnfw::cker::ConvParams &params,
                                          const nnfw::cker::Shape &incoming_shape,
                                          const T *incoming_data,
                                          const nnfw::cker::Shape &input_shape, const T *input_data,
                                          const nnfw::cker::Shape &filter_shape, T *expected)
  {
    assert(incoming_shape.DimensionsCount() == 4);
    assert(input_shape.DimensionsCount() == 4);
    assert(filter_shape.DimensionsCount() == 4);

    const int batches = MatchingDim(incoming_shape, 0, input_shape, 0);
    const int input_channel = MatchingDim(filter_shape, 2, input_shape, 3);
    const int output_channel = MatchingDim(filter_shape, 3, incoming_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int filter_height = filter_shape.Dims(0);
    const int filter_width = filter_shape.Dims(1);
    const int incoming_height = incoming_shape.Dims(1);
    const int incoming_width = incoming_shape.Dims(2);

    int pad_top = params.padding_values.height;
    int pad_left = params.padding_values.width;
    const int stride_height = params.stride_height;
    const int stride_width = params.stride_width;

    if (params.padding_type == nnfw::cker::PaddingType::kValid)
    {
      pad_top = 0;
      pad_left = 0;
    }
    else if (params.padding_type == nnfw::cker::PaddingType::kSame)
    {
      pad_top = ((incoming_height - 1) * stride_height + filter_height - input_height) / 2;
      pad_left = ((incoming_width - 1) * stride_width + filter_width - input_width) / 2;
    }

    assert(pad_top >= 0);
    assert(pad_left >= 0);
    assert(stride_height > 0);
    assert(stride_width > 0);

    for (int m = 0; m < batches; ++m)
    {
      for (int i = 0; i < incoming_height; ++i)
      {
        for (int j = 0; j < incoming_width; ++j)
        {
          for (int a = 0; a < filter_height; ++a)
          {
            for (int b = 0; b < filter_width; ++b)
            {
              for (int c = 0; c < input_channel; ++c)
              {
                for (int k = 0; k < output_channel; ++k)
                {
                  const auto input_i = stride_height * i + a - pad_top;
                  const auto input_j = stride_width * j + b - pad_left;
                  if (input_i < 0 || input_i >= input_height || input_j < 0 ||
                      input_j >= input_width)
                    continue;

                  const auto filter_offset = Offset(filter_shape, a, b, c, k);
                  const auto incoming_offset = Offset(incoming_shape, m, i, j, k);
                  const auto input_offset = Offset(input_shape, m, input_i, input_j, c);

                  expected[filter_offset] +=
                    incoming_data[incoming_offset] * input_data[input_offset];
                }
              }
            }
          }
        }
      }
    }
  }

  static void calculateInputGradExpected(const nnfw::cker::ConvParams &params,
                                         const nnfw::cker::Shape &incoming_shape,
                                         const T *incoming_data,
                                         const nnfw::cker::Shape &filter_shape,
                                         const T *filter_data, const nnfw::cker::Shape &input_shape,
                                         T *expected)
  {
    assert(incoming_shape.DimensionsCount() == 4);
    assert(filter_shape.DimensionsCount() == 4);
    assert(input_shape.DimensionsCount() == 4);

    const int batches = MatchingDim(incoming_shape, 0, input_shape, 0);
    const int input_channel = MatchingDim(filter_shape, 2, input_shape, 3);
    const int output_channel = MatchingDim(filter_shape, 3, incoming_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int filter_height = filter_shape.Dims(0);
    const int filter_width = filter_shape.Dims(1);
    const int incoming_height = incoming_shape.Dims(1);
    const int incoming_width = incoming_shape.Dims(2);

    int pad_top = params.padding_values.height;
    int pad_left = params.padding_values.width;
    const int stride_height = params.stride_height;
    const int stride_width = params.stride_width;

    if (params.padding_type == nnfw::cker::PaddingType::kValid)
    {
      pad_top = 0;
      pad_left = 0;
    }
    else if (params.padding_type == nnfw::cker::PaddingType::kSame)
    {
      pad_top = ((incoming_height - 1) * stride_height + filter_height - input_height) / 2;
      pad_left = ((incoming_width - 1) * stride_width + filter_width - input_width) / 2;
    }

    assert(pad_top >= 0);
    assert(pad_left >= 0);
    assert(stride_height > 0);
    assert(stride_width > 0);

    for (int m = 0; m < batches; ++m)
    {
      for (int i = 0; i < incoming_height; ++i)
      {
        for (int j = 0; j < incoming_width; ++j)
        {
          for (int a = 0; a < filter_height; ++a)
          {
            for (int b = 0; b < filter_width; ++b)
            {
              for (int c = 0; c < input_channel; ++c)
              {
                for (int k = 0; k < output_channel; ++k)
                {
                  const auto input_i = stride_height * i + a - pad_top;
                  const auto input_j = stride_width * j + b - pad_left;
                  if (input_i < 0 || input_i >= input_height || input_j < 0 ||
                      input_j >= input_width)
                    continue;

                  const auto filter_offset = Offset(filter_shape, a, b, c, k);
                  const auto incoming_offset = Offset(incoming_shape, m, i, j, k);
                  const auto input_offset = Offset(input_shape, m, input_i, input_j, c);

                  expected[input_offset] +=
                    incoming_data[incoming_offset] * filter_data[filter_offset];
                }
              }
            }
          }
        }
      }
    }
  }
};

} // namespace

TEST(CKer_Operation, ConvGrad)
{
  // No pad, No stride
  {
    nnfw::cker::ConvParams params;
    params.padding_type = nnfw::cker::PaddingType::kNone;
    params.padding_values.width = 0;
    params.padding_values.height = 0;
    params.stride_width = 1;
    params.stride_height = 1;
    params.dilation_width_factor = 1;
    params.dilation_height_factor = 1;

    nnfw::cker::Shape incoming_shape{1, 2, 2, 3}; // n, h, w, c
    std::vector<float> incoming = {-0.1, 0.2, -0.3, 0.4,  0.5, -0.6,
                                   -0.7, 0.8, 0.9,  -1.0, 1.1, -1.2};
    nnfw::cker::Shape filter_shape{2, 2, 2, 3}; // h, w, i, o
    std::vector<float> filter = {-1,  2,  -3,  4,  5,  -6,  -7,  8,  9,   -10, -11, 12,
                                 -13, 14, -15, 16, 17, -18, -19, 20, -21, 22,  23,  -24};
    nnfw::cker::Shape input_shape{1, 3, 3, 2}; // n, h, w, c
    std::vector<float> input = {-1,  2,  -3,  4,  5,   -6, -7,  8,   -9,
                                -10, 11, -12, 13, -14, 15, -16, -17, 18};

    const auto padding_bottom = 0;
    const auto padding_right = 0;

    ConvVerifier<float>::verifyFilterGradExpected(params, incoming_shape, incoming.data(),
                                                  input_shape, input.data(), padding_bottom,
                                                  padding_right, filter_shape);
    ConvVerifier<float>::verifyInputGradExpected(params, incoming_shape, incoming.data(),
                                                 filter_shape, filter.data(), padding_bottom,
                                                 padding_right, input_shape);
  }

  // pad top 1, pad left 1, No stride
  {
    nnfw::cker::ConvParams params;
    params.padding_type = nnfw::cker::PaddingType::kNone;
    params.padding_values.width = 1;
    params.padding_values.height = 1;
    params.stride_width = 1;
    params.stride_height = 1;
    params.dilation_width_factor = 1;
    params.dilation_height_factor = 1;

    nnfw::cker::Shape incoming_shape{1, 3, 3, 3}; // n, h, w, c
    std::vector<float> incoming = {-0.1, 0.2, -0.3, 0.4, 0.5,  -0.6, -0.7, 0.8,  0.9,
                                   -1.0, 1.1, -1.2, 1.3, -1.4, 1.5,  -1.6, 1.7,  -1.8,
                                   -1.9, 2.0, -2.1, 2.2, 2.3,  -2.4, 2.5,  -2.6, -2.7};
    nnfw::cker::Shape filter_shape{2, 2, 2, 3}; // h, w, i, o
    std::vector<float> filter = {-1,  2,  -3,  4,  5,  -6,  -7,  8,  9,   -10, -11, 12,
                                 -13, 14, -15, 16, 17, -18, -19, 20, -21, 22,  23,  -24};
    nnfw::cker::Shape input_shape{1, 3, 3, 2}; // n, h, w, c
    std::vector<float> input = {-1,  2,  -3,  4,  5,   -6, -7,  8,   -9,
                                -10, 11, -12, 13, -14, 15, -16, -17, 18};

    const auto padding_bottom = 0;
    const auto padding_right = 0;

    ConvVerifier<float>::verifyFilterGradExpected(params, incoming_shape, incoming.data(),
                                                  input_shape, input.data(), padding_bottom,
                                                  padding_right, filter_shape);
    ConvVerifier<float>::verifyInputGradExpected(params, incoming_shape, incoming.data(),
                                                 filter_shape, filter.data(), padding_bottom,
                                                 padding_right, input_shape);
  }

  // pad top 1, pad left 1, pad bottom 2, pad right 2, stride 2
  {
    nnfw::cker::ConvParams params;
    params.padding_type = nnfw::cker::PaddingType::kNone;
    params.padding_values.width = 1;
    params.padding_values.height = 1;
    params.stride_width = 2;
    params.stride_height = 2;
    params.dilation_width_factor = 1;
    params.dilation_height_factor = 1;

    nnfw::cker::Shape incoming_shape{1, 3, 3, 3}; // n, h, w, c
    std::vector<float> incoming = {-0.1, 0.2, -0.3, 0.4, 0.5,  -0.6, -0.7, 0.8,  0.9,
                                   -1.0, 1.1, -1.2, 1.3, -1.4, -1.5, 1.6,  1.7,  -1.8,
                                   1.9,  2.0, -2.1, 2.2, -2.3, 2.4,  2.5,  -2.6, -2.7};
    nnfw::cker::Shape filter_shape{3, 3, 2, 3}; // h, w, i, o
    std::vector<float> filter = {
      -1,  2,   -3,  4,   5,  -6,  -7,  8,   9,  -10, -11, 12,  -13, 14,  -15, 16,  17,  -18,
      -19, 20,  -21, 22,  23, -24, -25, 26,  27, -28, -29, -30, 31,  -32, 33,  -34, 35,  36,
      -37, -38, 39,  -40, 41, 42,  43,  -44, 45, -46, 47,  48,  -49, -50, -51, 52,  -53, 54};
    nnfw::cker::Shape input_shape{1, 3, 3, 2}; // n, h, w, c
    std::vector<float> input = {-1,  2,  -3,  4,  5,   -6, -7,  8,   -9,
                                -10, 11, -12, 13, -14, 15, -16, -17, 18};

    const auto padding_bottom = 2;
    const auto padding_right = 2;

    ConvVerifier<float>::verifyFilterGradExpected(params, incoming_shape, incoming.data(),
                                                  input_shape, input.data(), padding_bottom,
                                                  padding_right, filter_shape);
    ConvVerifier<float>::verifyInputGradExpected(params, incoming_shape, incoming.data(),
                                                 filter_shape, filter.data(), padding_bottom,
                                                 padding_right, input_shape);
  }

  // pad same, stride 2
  {
    nnfw::cker::ConvParams params;
    params.padding_type = nnfw::cker::PaddingType::kSame;
    params.stride_width = 2;
    params.stride_height = 2;
    params.dilation_width_factor = 1;
    params.dilation_height_factor = 1;

    nnfw::cker::Shape incoming_shape{1, 3, 3, 3}; // n, h, w, c
    std::vector<float> incoming = {-0.1, 0.2, -0.3, 0.4, 0.5,  -0.6, -0.7, 0.8,  0.9,
                                   -1.0, 1.1, -1.2, 1.3, -1.4, -1.5, 1.6,  1.7,  -1.8,
                                   1.9,  2.0, -2.1, 2.2, -2.3, 2.4,  2.5,  -2.6, -2.7};
    nnfw::cker::Shape filter_shape{3, 3, 2, 3}; // h, w, i, o
    std::vector<float> filter = {
      -1,  2,   -3,  4,   5,  -6,  -7,  8,   9,  -10, -11, 12,  -13, 14,  -15, 16,  17,  -18,
      -19, 20,  -21, 22,  23, -24, -25, 26,  27, -28, -29, -30, 31,  -32, 33,  -34, 35,  36,
      -37, -38, 39,  -40, 41, 42,  43,  -44, 45, -46, 47,  48,  -49, -50, -51, 52,  -53, 54};
    nnfw::cker::Shape input_shape{1, 3, 3, 2}; // n, h, w, c
    std::vector<float> input = {-1,  2,  -3,  4,  5,   -6, -7,  8,   -9,
                                -10, 11, -12, 13, -14, 15, -16, -17, 18};

    const auto padding_bottom = 0;
    const auto padding_right = 0;

    ConvVerifier<float>::verifyFilterGradExpected(params, incoming_shape, incoming.data(),
                                                  input_shape, input.data(), padding_bottom,
                                                  padding_right, filter_shape);
    ConvVerifier<float>::verifyInputGradExpected(params, incoming_shape, incoming.data(),
                                                 filter_shape, filter.data(), padding_bottom,
                                                 padding_right, input_shape);
  }

  // pad valid, stride 2
  {
    nnfw::cker::ConvParams params;
    params.padding_type = nnfw::cker::PaddingType::kValid;
    params.stride_width = 2;
    params.stride_height = 2;
    params.dilation_width_factor = 1;
    params.dilation_height_factor = 1;

    nnfw::cker::Shape incoming_shape{1, 1, 1, 3}; // n, h, w, c
    std::vector<float> incoming = {-0.1, 0.2, -0.3};
    nnfw::cker::Shape filter_shape{2, 2, 2, 3}; // h, w, i, o
    std::vector<float> filter = {-1,  2,  -3,  4,  5,  -6,  -7,  8,  9,   -10, -11, 12,
                                 -13, 14, -15, 16, 17, -18, -19, 20, -21, 22,  23,  -24};
    nnfw::cker::Shape input_shape{1, 3, 3, 2}; // n, h, w, c
    std::vector<float> input = {-1,  2,  -3,  4,  5,   -6, -7,  8,   -9,
                                -10, 11, -12, 13, -14, 15, -16, -17, 18};

    const auto padding_bottom = 0;
    const auto padding_right = 0;

    ConvVerifier<float>::verifyFilterGradExpected(params, incoming_shape, incoming.data(),
                                                  input_shape, input.data(), padding_bottom,
                                                  padding_right, filter_shape);
    ConvVerifier<float>::verifyInputGradExpected(params, incoming_shape, incoming.data(),
                                                 filter_shape, filter.data(), padding_bottom,
                                                 padding_right, input_shape);
  }
}

TEST(CKer_Operation, neg_ConvGradUnsupportedDilation)
{
  // Unsupported dilation
  {
    nnfw::cker::ConvParams params;
    params.padding_type = nnfw::cker::PaddingType::kNone;
    params.padding_values.width = 0;
    params.padding_values.height = 0;
    params.stride_width = 1;
    params.stride_height = 1;
    params.dilation_width_factor = 2;
    params.dilation_height_factor = 2;

    nnfw::cker::Shape incoming_shape{1, 2, 2, 3}; // n, h, w, c
    std::vector<float> incoming = {-0.1, 0.2, -0.3, 0.4,  0.5, -0.6,
                                   -0.7, 0.8, 0.9,  -1.0, 1.1, -1.2};
    nnfw::cker::Shape filter_shape{2, 2, 2, 3}; // h, w, i, o
    std::vector<float> filter = {-1,  2,  -3,  4,  5,  -6,  -7,  8,  9,   -10, -11, 12,
                                 -13, 14, -15, 16, 17, -18, -19, 20, -21, 22,  23,  -24};
    nnfw::cker::Shape input_shape{1, 3, 3, 2}; // n, h, w, c
    std::vector<float> input = {-1,  2,  -3,  4,  5,   -6, -7,  8,   -9,
                                -10, 11, -12, 13, -14, 15, -16, -17, 18};

    EXPECT_ANY_THROW(nnfw::cker::train::ConvFilterGrad(params, incoming_shape, incoming.data(),
                                                       input_shape, input.data(), 0, 0,
                                                       filter_shape, filter.data()));

    EXPECT_ANY_THROW(nnfw::cker::train::ConvInputGrad(params, incoming_shape, incoming.data(),
                                                      filter_shape, filter.data(), 0, 0,
                                                      input_shape, input.data()));
  }
}
