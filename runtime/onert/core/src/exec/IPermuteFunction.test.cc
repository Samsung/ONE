/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "IPermuteFunction.h"

#include <ir/Layout.h>
#include <ir/Shape.h>
#include <ir/TypeInfo.h>

#include <cmath>
#include <gtest/gtest.h>

namespace
{
using namespace onert;
using namespace ir;
using namespace backend;
using namespace exec;

class MockUpTensor : public ITensor
{
public:
  MockUpTensor(const Shape &shape, const TypeInfo &type_info, Layout layout, size_t pad)
    : _shape(shape), _type_info(type_info), _data(nullptr), _layout(layout)
  {
    _strides.resize(shape.rank());

    std::vector<size_t> pads(shape.rank(), 0);
    pads[shape.rank() - 1] = pad;
    size_t stride = 1;
    for (int32_t i = _shape.rank() - 1; i >= 0; --i)
    {
      _strides.at(i) = stride;
      stride = stride * (_shape.dim(i) + pads.at(i));
    }
  }
  virtual ~MockUpTensor() {}

  void setBuffer(uint8_t *data) { _data = data; }

  size_t total_size() const override
  {
    size_t total_size = _strides[0] * _shape.dim(0);
    total_size *= sizeOfDataType(data_type());
    return total_size;
  }

  size_t calcOffset(const ir::Coordinates &coords) const override
  {
    size_t offset = 0;
    for (size_t i = 0; i < _shape.rank(); ++i)
    {
      offset += (_strides[i] * coords[i]);
    }
    offset *= sizeOfDataType(data_type());
    return offset;
  }

  uint8_t *buffer() const override { return _data; }

  ir::Layout layout() const override { return _layout; }
  ir::DataType data_type() const override { return _type_info.type(); }
  float data_scale() const override { return _type_info.scale(); }
  int32_t data_zero_point() const override { return _type_info.zero_point(); }
  const std::vector<float> &data_scales() const override { return _type_info.scales(); }
  const std::vector<int32_t> &data_zero_points() const override { return _type_info.zero_points(); }
  bool has_padding() const override
  {
    return total_size() / sizeOfDataType(data_type()) != _shape.num_elements();
  }
  void access(const std::function<void(ITensor &tensor)> &fn) final { fn(*this); }

  bool is_dynamic() const override { return false; }
  Shape getShape() const override { return _shape; }

private:
  Shape _shape;
  TypeInfo _type_info;
  Layout _layout;
  uint8_t *_data;
  std::vector<size_t> _strides;
};

class MockUpLayer : public IPermuteFunction
{
public:
  MockUpLayer(const std::vector<ITensor *> &inputs, const std::vector<ITensor *> &outputs)
  {
    assert(inputs.size() == outputs.size());
    _src_tensors = inputs;
    _dst_tensors = outputs;
  }
  virtual ~MockUpLayer() {}
  void optimize() override {}
};

TEST(IPermuteFunction, float_to_float)
{
  // rank 1
  {
    const size_t input_pads[4] = {0, 1, 0, 2};
    const size_t output_pads[4] = {0, 0, 2, 1};
    const std::vector<Shape> shapes{{1}, {4}, {5}, {2}};
    float expected_buffer[] = {1, 0, -1, -2, 3};
    const auto type_info = TypeInfo(DataType::FLOAT32);

    std::vector<std::unique_ptr<MockUpTensor>> inputs(4);
    std::vector<std::unique_ptr<MockUpTensor>> outputs(4);

    std::vector<std::unique_ptr<uint8_t[]>> output_buffers(4);
    for (size_t i = 0; i < 4; ++i)
    {
      inputs[i] = std::make_unique<MockUpTensor>(shapes[i], type_info, Layout::NHWC, input_pads[i]);
      inputs[i]->setBuffer(reinterpret_cast<uint8_t *>(expected_buffer));

      outputs[i] =
        std::make_unique<MockUpTensor>(shapes[i], type_info, Layout::NHWC, output_pads[i]);
      output_buffers[i] = std::make_unique<uint8_t[]>(outputs[i]->total_size());
      outputs[i]->setBuffer(output_buffers[i].get());
    }

    auto mockup_layer = std::make_unique<MockUpLayer>(
      std::vector<ITensor *>{inputs[0].get(), inputs[1].get(), inputs[2].get(), inputs[3].get()},
      std::vector<ITensor *>{outputs[0].get(), outputs[1].get(), outputs[2].get(),
                             outputs[3].get()});
    mockup_layer->run();

    for (size_t i = 0; i < 4; ++i)
    {
      for (int32_t j = 0; j < shapes[i].dim(0); ++j)
      {
        Coordinates coords{j};
        float result =
          *reinterpret_cast<float *>(outputs[i]->buffer() + outputs[i]->calcOffset(coords));
        float expected =
          *reinterpret_cast<float *>(inputs[i]->buffer() + inputs[i]->calcOffset(coords));
        EXPECT_EQ(result, expected);
      }
    }
  }

  // rank 2
  {
    const size_t input_pads[4] = {0, 1, 0, 2};
    const size_t output_pads[4] = {0, 0, 2, 1};
    const std::vector<Shape> shapes{{1, 4}, {2, 2}, {1, 5}, {2, 3}};
    float expected_buffer[] = {1, 0, -1, -2, 3, -4, 5, -6, 7, -8};
    const auto type_info = TypeInfo(DataType::FLOAT32);

    std::vector<std::unique_ptr<MockUpTensor>> inputs(4);
    std::vector<std::unique_ptr<MockUpTensor>> outputs(4);
    std::vector<std::unique_ptr<uint8_t[]>> output_buffers(4);
    for (size_t i = 0; i < 4; ++i)
    {
      inputs[i] = std::make_unique<MockUpTensor>(shapes[i], type_info, Layout::NHWC, input_pads[i]);
      inputs[i]->setBuffer(reinterpret_cast<uint8_t *>(expected_buffer));

      outputs[i] =
        std::make_unique<MockUpTensor>(shapes[i], type_info, Layout::NHWC, output_pads[i]);
      output_buffers[i] = std::make_unique<uint8_t[]>(outputs[i]->total_size());
      outputs[i]->setBuffer(output_buffers[i].get());
    }

    auto mockup_layer = std::make_unique<MockUpLayer>(
      std::vector<ITensor *>{inputs[0].get(), inputs[1].get(), inputs[2].get(), inputs[3].get()},
      std::vector<ITensor *>{outputs[0].get(), outputs[1].get(), outputs[2].get(),
                             outputs[3].get()});
    mockup_layer->run();

    for (size_t i = 0; i < 4; ++i)
    {
      for (int32_t j = 0; j < shapes[i].dim(0); ++j)
      {
        for (int32_t k = 0; k < shapes[i].dim(1); ++k)
        {
          Coordinates coords{j, k};
          float result =
            *reinterpret_cast<float *>(outputs[i]->buffer() + outputs[i]->calcOffset(coords));
          float expected =
            *reinterpret_cast<float *>(inputs[i]->buffer() + inputs[i]->calcOffset(coords));
          EXPECT_EQ(result, expected);
        }
      }
    }
  }

  // rank 3
  {
    const size_t input_pads[4] = {0, 5, 0, 2};
    const size_t output_pads[4] = {0, 3, 2, 1};
    const std::vector<Shape> shapes{{1, 4, 1}, {1, 2, 1}, {2, 1, 5}, {1, 2, 3}};
    float expected_buffer[] = {1, 0, -1, -2, 3, -4, 5, -6, 7, -8, 9, -10};
    const auto type_info = TypeInfo(DataType::FLOAT32);

    std::vector<std::unique_ptr<MockUpTensor>> inputs(4);
    std::vector<std::unique_ptr<MockUpTensor>> outputs(4);
    std::vector<std::unique_ptr<uint8_t[]>> output_buffers(4);
    for (size_t i = 0; i < 4; ++i)
    {
      inputs[i] = std::make_unique<MockUpTensor>(shapes[i], type_info, Layout::NHWC, input_pads[i]);
      inputs[i]->setBuffer(reinterpret_cast<uint8_t *>(expected_buffer));

      outputs[i] =
        std::make_unique<MockUpTensor>(shapes[i], type_info, Layout::NHWC, output_pads[i]);
      output_buffers[i] = std::make_unique<uint8_t[]>(outputs[i]->total_size());
      outputs[i]->setBuffer(output_buffers[i].get());
    }

    auto mockup_layer = std::make_unique<MockUpLayer>(
      std::vector<ITensor *>{inputs[0].get(), inputs[1].get(), inputs[2].get(), inputs[3].get()},
      std::vector<ITensor *>{outputs[0].get(), outputs[1].get(), outputs[2].get(),
                             outputs[3].get()});
    mockup_layer->run();

    for (size_t i = 0; i < 4; ++i)
    {
      for (int32_t j = 0; j < shapes[i].dim(0); ++j)
      {
        for (int32_t k = 0; k < shapes[i].dim(1); ++k)
        {
          for (int32_t l = 0; l < shapes[i].dim(2); ++l)
          {
            Coordinates coords{j, k, l};
            float result =
              *reinterpret_cast<float *>(outputs[i]->buffer() + outputs[i]->calcOffset(coords));
            float expected =
              *reinterpret_cast<float *>(inputs[i]->buffer() + inputs[i]->calcOffset(coords));
            EXPECT_EQ(result, expected);
          }
        }
      }
    }
  }

  // rank 4
  {
    const size_t input_pads[4] = {0, 0, 1, 2};
    const size_t output_pads[4] = {0, 3, 2, 1};
    const std::vector<Shape> shapes{{1, 1, 4, 1}, {2, 1, 2, 3}, {1, 2, 1, 2}, {1, 1, 2, 3}};
    float expected_buffer[] = {1, 0, -1, -2, 3, -4, 5, -6, 7, -8, 9, -10};
    const auto type_info = TypeInfo(DataType::FLOAT32);

    std::vector<std::unique_ptr<MockUpTensor>> inputs(4);
    std::vector<std::unique_ptr<MockUpTensor>> outputs(4);
    std::vector<std::unique_ptr<uint8_t[]>> output_buffers(4);
    for (size_t i = 0; i < 4; ++i)
    {
      inputs[i] = std::make_unique<MockUpTensor>(shapes[i], type_info, Layout::NHWC, input_pads[i]);
      inputs[i]->setBuffer(reinterpret_cast<uint8_t *>(expected_buffer));

      outputs[i] =
        std::make_unique<MockUpTensor>(shapes[i], type_info, Layout::NHWC, output_pads[i]);
      output_buffers[i] = std::make_unique<uint8_t[]>(outputs[i]->total_size());
      outputs[i]->setBuffer(output_buffers[i].get());
    }

    auto mockup_layer = std::make_unique<MockUpLayer>(
      std::vector<ITensor *>{inputs[0].get(), inputs[1].get(), inputs[2].get(), inputs[3].get()},
      std::vector<ITensor *>{outputs[0].get(), outputs[1].get(), outputs[2].get(),
                             outputs[3].get()});
    mockup_layer->run();

    for (size_t i = 0; i < 4; ++i)
    {
      for (int32_t j = 0; j < shapes[i].dim(0); ++j)
      {
        for (int32_t k = 0; k < shapes[i].dim(1); ++k)
        {
          for (int32_t l = 0; l < shapes[i].dim(2); ++l)
          {
            for (int32_t m = 0; m < shapes[i].dim(3); ++m)
            {
              Coordinates coords{j, k, l, m};
              float result =
                *reinterpret_cast<float *>(outputs[i]->buffer() + outputs[i]->calcOffset(coords));
              float expected =
                *reinterpret_cast<float *>(inputs[i]->buffer() + inputs[i]->calcOffset(coords));
              EXPECT_EQ(result, expected);
            }
          }
        }
      }
    }
  }

  // rank4 layout
  {
    const size_t input_pads[4] = {0, 0, 1, 2};
    const size_t output_pads[4] = {0, 3, 2, 1};
    const std::vector<Shape> shapes{{1, 1, 4, 1}, {2, 1, 2, 3}, {1, 2, 1, 2}, {1, 1, 2, 3}};
    float expected_buffer[] = {1,  0, -1,  -2, 3,   -4, 5,   -6, 7,
                               -8, 9, -10, 11, -12, 13, -14, 15, -16};
    const auto type_info = TypeInfo(DataType::FLOAT32);

    std::vector<std::unique_ptr<MockUpTensor>> inputs(4);
    std::vector<std::unique_ptr<MockUpTensor>> outputs(4);
    std::vector<std::unique_ptr<uint8_t[]>> output_buffers(4);
    for (size_t i = 0; i < 4; ++i)
    {
      Layout layout = Layout::NHWC;
      Shape shape = shapes[i];
      if (i % 2 == 1)
      {
        layout = Layout::NCHW;
        shape = Shape{shapes[i].dim(0), shapes[i].dim(3), shapes[i].dim(1), shapes[i].dim(2)};
      }
      inputs[i] = std::make_unique<MockUpTensor>(shape, type_info, layout, input_pads[i]);
      inputs[i]->setBuffer(reinterpret_cast<uint8_t *>(expected_buffer));

      if (layout == Layout::NHWC)
      {
        layout = Layout::NCHW;
        shape = Shape{shapes[i].dim(0), shapes[i].dim(3), shapes[i].dim(1), shapes[i].dim(2)};
      }
      else
      {
        layout = Layout::NHWC;
        shape = shapes[i];
      }
      outputs[i] = std::make_unique<MockUpTensor>(shape, type_info, layout, output_pads[i]);
      output_buffers[i] = std::make_unique<uint8_t[]>(outputs[i]->total_size());
      outputs[i]->setBuffer(output_buffers[i].get());
    }

    auto mockup_layer = std::make_unique<MockUpLayer>(
      std::vector<ITensor *>{inputs[0].get(), inputs[1].get(), inputs[2].get(), inputs[3].get()},
      std::vector<ITensor *>{outputs[0].get(), outputs[1].get(), outputs[2].get(),
                             outputs[3].get()});
    mockup_layer->run();

    for (size_t i = 0; i < 4; ++i)
    {
      for (int32_t j = 0; j < shapes[i].dim(0); ++j)
      {
        for (int32_t k = 0; k < shapes[i].dim(1); ++k)
        {
          for (int32_t l = 0; l < shapes[i].dim(2); ++l)
          {
            for (int32_t m = 0; m < shapes[i].dim(3); ++m)
            {
              Coordinates input_coords;
              Coordinates output_coords;
              if (inputs[i]->layout() == Layout::NHWC)
              {
                input_coords = Coordinates{j, k, l, m};
              }
              else
              {
                input_coords = Coordinates{j, m, k, l};
              }
              if (outputs[i]->layout() == Layout::NHWC)
              {
                output_coords = Coordinates{j, k, l, m};
              }
              else
              {
                output_coords = Coordinates{j, m, k, l};
              }
              float result = *reinterpret_cast<float *>(outputs[i]->buffer() +
                                                        outputs[i]->calcOffset(output_coords));
              float expected = *reinterpret_cast<float *>(inputs[i]->buffer() +
                                                          inputs[i]->calcOffset(input_coords));
              EXPECT_EQ(result, expected);
            }
          }
        }
      }
    }
  }
}

TEST(IPermuteFunction, float_to_qasymm8)
{
  const size_t input_pads[4] = {0, 0, 1, 2};
  const size_t output_pads[4] = {0, 3, 2, 1};
  const std::vector<Shape> shapes{{1, 1, 4, 1}, {2, 1, 2, 3}, {1, 2, 1, 2}, {1, 1, 2, 3}};
  float expected_buffer[] = {10, 0, -10, -20, 30, -40, 50, -60, 70, -80, 90, -100};
  float scale = 10;
  int32_t zero_point = 128;

  std::vector<std::unique_ptr<MockUpTensor>> inputs(4);
  std::vector<std::unique_ptr<MockUpTensor>> outputs(4);
  std::vector<std::unique_ptr<uint8_t[]>> output_buffers(4);
  for (size_t i = 0; i < 4; ++i)
  {
    inputs[i] = std::make_unique<MockUpTensor>(shapes[i], TypeInfo(DataType::FLOAT32), Layout::NHWC,
                                               input_pads[i]);
    inputs[i]->setBuffer(reinterpret_cast<uint8_t *>(expected_buffer));

    TypeInfo type_info{DataType::QUANT_UINT8_ASYMM, scale, zero_point};
    outputs[i] = std::make_unique<MockUpTensor>(shapes[i], type_info, Layout::NHWC, output_pads[i]);
    output_buffers[i] = std::make_unique<uint8_t[]>(outputs[i]->total_size());
    outputs[i]->setBuffer(output_buffers[i].get());
  }

  auto mockup_layer = std::make_unique<MockUpLayer>(
    std::vector<ITensor *>{inputs[0].get(), inputs[1].get(), inputs[2].get(), inputs[3].get()},
    std::vector<ITensor *>{outputs[0].get(), outputs[1].get(), outputs[2].get(), outputs[3].get()});
  mockup_layer->run();

  for (size_t i = 0; i < 4; ++i)
  {
    for (int32_t j = 0; j < shapes[i].dim(0); ++j)
    {
      for (int32_t k = 0; k < shapes[i].dim(1); ++k)
      {
        for (int32_t l = 0; l < shapes[i].dim(2); ++l)
        {
          for (int32_t m = 0; m < shapes[i].dim(3); ++m)
          {
            Coordinates coords{j, k, l, m};
            uint8_t qasymm8 =
              *reinterpret_cast<uint8_t *>(outputs[i]->buffer() + outputs[i]->calcOffset(coords));
            float result = (qasymm8 - zero_point) * scale;
            float expected =
              *reinterpret_cast<float *>(inputs[i]->buffer() + inputs[i]->calcOffset(coords));
            EXPECT_EQ(result, expected);
          }
        }
      }
    }
  }
}

TEST(IPermuteFunction, float_to_qsymm8)
{
  const size_t input_pads[4] = {0, 0, 1, 2};
  const size_t output_pads[4] = {0, 3, 2, 1};
  const std::vector<Shape> shapes{{1, 1, 4, 1}, {2, 1, 2, 3}, {1, 2, 1, 2}, {1, 1, 2, 3}};
  float expected_buffer[] = {10, 0, -10, -20, 30, -40, 50, -60, 70, -80, 90, -100};
  float scale = 10;
  int32_t zero_point = 0;

  std::vector<std::unique_ptr<MockUpTensor>> inputs(4);
  std::vector<std::unique_ptr<MockUpTensor>> outputs(4);
  std::vector<std::unique_ptr<uint8_t[]>> output_buffers(4);
  for (size_t i = 0; i < 4; ++i)
  {
    inputs[i] = std::make_unique<MockUpTensor>(shapes[i], TypeInfo(DataType::FLOAT32), Layout::NHWC,
                                               input_pads[i]);
    inputs[i]->setBuffer(reinterpret_cast<uint8_t *>(expected_buffer));

    TypeInfo type_info{DataType::QUANT_INT8_SYMM, scale, zero_point};
    outputs[i] = std::make_unique<MockUpTensor>(shapes[i], type_info, Layout::NHWC, output_pads[i]);
    output_buffers[i] = std::make_unique<uint8_t[]>(outputs[i]->total_size());
    outputs[i]->setBuffer(output_buffers[i].get());
  }

  auto mockup_layer = std::make_unique<MockUpLayer>(
    std::vector<ITensor *>{inputs[0].get(), inputs[1].get(), inputs[2].get(), inputs[3].get()},
    std::vector<ITensor *>{outputs[0].get(), outputs[1].get(), outputs[2].get(), outputs[3].get()});
  mockup_layer->run();

  for (size_t i = 0; i < 4; ++i)
  {
    for (int32_t j = 0; j < shapes[i].dim(0); ++j)
    {
      for (int32_t k = 0; k < shapes[i].dim(1); ++k)
      {
        for (int32_t l = 0; l < shapes[i].dim(2); ++l)
        {
          for (int32_t m = 0; m < shapes[i].dim(3); ++m)
          {
            Coordinates coords{j, k, l, m};
            int8_t qsymm8 =
              *reinterpret_cast<int8_t *>(outputs[i]->buffer() + outputs[i]->calcOffset(coords));
            float result = (qsymm8 - zero_point) * scale;
            float expected =
              *reinterpret_cast<float *>(inputs[i]->buffer() + inputs[i]->calcOffset(coords));
            EXPECT_EQ(result, expected);
          }
        }
      }
    }
  }
}

TEST(IPermuteFunction, float_to_qsymm16)
{
  const size_t input_pads[4] = {0, 0, 1, 2};
  const size_t output_pads[4] = {0, 3, 2, 1};
  const std::vector<Shape> shapes{{1, 1, 4, 1}, {2, 1, 2, 3}, {1, 2, 1, 2}, {1, 1, 2, 3}};
  float expected_buffer[] = {10, 0, -10, -20, 30, -40, 50, -60, 70, -80, 90, -100};
  float scale = 10;
  int32_t zero_point = 0;

  std::vector<std::unique_ptr<MockUpTensor>> inputs(4);
  std::vector<std::unique_ptr<MockUpTensor>> outputs(4);
  std::vector<std::unique_ptr<uint8_t[]>> output_buffers(4);
  for (size_t i = 0; i < 4; ++i)
  {
    inputs[i] = std::make_unique<MockUpTensor>(shapes[i], TypeInfo(DataType::FLOAT32), Layout::NHWC,
                                               input_pads[i]);
    inputs[i]->setBuffer(reinterpret_cast<uint8_t *>(expected_buffer));

    TypeInfo type_info{DataType::QUANT_INT16_SYMM, scale, zero_point};
    outputs[i] = std::make_unique<MockUpTensor>(shapes[i], type_info, Layout::NHWC, output_pads[i]);
    output_buffers[i] = std::make_unique<uint8_t[]>(outputs[i]->total_size());
    outputs[i]->setBuffer(output_buffers[i].get());
  }

  auto mockup_layer = std::make_unique<MockUpLayer>(
    std::vector<ITensor *>{inputs[0].get(), inputs[1].get(), inputs[2].get(), inputs[3].get()},
    std::vector<ITensor *>{outputs[0].get(), outputs[1].get(), outputs[2].get(), outputs[3].get()});
  mockup_layer->run();

  for (size_t i = 0; i < 4; ++i)
  {
    for (int32_t j = 0; j < shapes[i].dim(0); ++j)
    {
      for (int32_t k = 0; k < shapes[i].dim(1); ++k)
      {
        for (int32_t l = 0; l < shapes[i].dim(2); ++l)
        {
          for (int32_t m = 0; m < shapes[i].dim(3); ++m)
          {
            Coordinates coords{j, k, l, m};
            int16_t qsymm16 =
              *reinterpret_cast<int16_t *>(outputs[i]->buffer() + outputs[i]->calcOffset(coords));
            float result = (qsymm16 - zero_point) * scale;
            float expected =
              *reinterpret_cast<float *>(inputs[i]->buffer() + inputs[i]->calcOffset(coords));
            EXPECT_EQ(result, expected);
          }
        }
      }
    }
  }
}

TEST(IPermuteFunction, qasymm8_to_float)
{
  const size_t input_pads[4] = {0, 0, 1, 2};
  const size_t output_pads[4] = {0, 3, 2, 1};
  const std::vector<Shape> shapes{{1, 1, 4, 1}, {2, 1, 2, 3}, {1, 2, 1, 2}, {1, 1, 2, 3}};
  float expected_buffer[] = {10, 0, -10, -20, 30, -40, 50, -60, 70, -80, 90, -100};
  float scale = 10;
  int32_t zero_point = 128;
  uint8_t input_buffer[12];

  int32_t min_val = std::numeric_limits<uint8_t>::min();
  int32_t max_val = std::numeric_limits<uint8_t>::max();
  for (int32_t i = 0; i < sizeof(expected_buffer) / sizeof(float); ++i)
  {
    int32_t unclamped = static_cast<int32_t>(std::round(expected_buffer[i] / scale)) + zero_point;
    input_buffer[i] = std::min(std::max(unclamped, min_val), max_val);
  }

  std::vector<std::unique_ptr<MockUpTensor>> inputs(4);
  std::vector<std::unique_ptr<MockUpTensor>> outputs(4);
  std::vector<std::unique_ptr<uint8_t[]>> output_buffers(4);
  for (size_t i = 0; i < 4; ++i)
  {
    TypeInfo type_info{DataType::QUANT_UINT8_ASYMM, scale, zero_point};
    inputs[i] = std::make_unique<MockUpTensor>(shapes[i], type_info, Layout::NHWC, input_pads[i]);
    inputs[i]->setBuffer(reinterpret_cast<uint8_t *>(input_buffer));

    outputs[i] = std::make_unique<MockUpTensor>(shapes[i], TypeInfo(DataType::FLOAT32),
                                                Layout::NHWC, output_pads[i]);
    output_buffers[i] = std::make_unique<uint8_t[]>(outputs[i]->total_size());
    outputs[i]->setBuffer(output_buffers[i].get());
  }

  auto mockup_layer = std::make_unique<MockUpLayer>(
    std::vector<ITensor *>{inputs[0].get(), inputs[1].get(), inputs[2].get(), inputs[3].get()},
    std::vector<ITensor *>{outputs[0].get(), outputs[1].get(), outputs[2].get(), outputs[3].get()});
  mockup_layer->run();

  for (size_t i = 0; i < 4; ++i)
  {
    for (int32_t j = 0; j < shapes[i].dim(0); ++j)
    {
      for (int32_t k = 0; k < shapes[i].dim(1); ++k)
      {
        for (int32_t l = 0; l < shapes[i].dim(2); ++l)
        {
          for (int32_t m = 0; m < shapes[i].dim(3); ++m)
          {
            Coordinates coords{j, k, l, m};
            float result =
              *reinterpret_cast<float *>(outputs[i]->buffer() + outputs[i]->calcOffset(coords));
            uint8_t qasymm8 =
              *reinterpret_cast<uint8_t *>(inputs[i]->buffer() + inputs[i]->calcOffset(coords));
            float expected = (qasymm8 - zero_point) * scale;
            EXPECT_EQ(result, expected);
          }
        }
      }
    }
  }
}

TEST(IPermuteFunction, qsymm8_to_float)
{
  const size_t input_pads[4] = {0, 0, 1, 2};
  const size_t output_pads[4] = {0, 3, 2, 1};
  const std::vector<Shape> shapes{{1, 1, 4, 1}, {2, 1, 2, 3}, {1, 2, 1, 2}, {1, 1, 2, 3}};
  float expected_buffer[] = {10, 0, -10, -20, 30, -40, 50, -60, 70, -80, 90, -100};
  float scale = 10;
  int32_t zero_point = 0;
  uint8_t input_buffer[12];

  int32_t min_val = std::numeric_limits<int8_t>::min();
  int32_t max_val = std::numeric_limits<int8_t>::max();
  for (int32_t i = 0; i < sizeof(expected_buffer) / sizeof(float); ++i)
  {
    int32_t unclamped = static_cast<int32_t>(std::round(expected_buffer[i] / scale)) + zero_point;
    input_buffer[i] = std::min(std::max(unclamped, min_val), max_val);
  }

  std::vector<std::unique_ptr<MockUpTensor>> inputs(4);
  std::vector<std::unique_ptr<MockUpTensor>> outputs(4);
  std::vector<std::unique_ptr<uint8_t[]>> output_buffers(4);
  for (size_t i = 0; i < 4; ++i)
  {
    TypeInfo type_info{DataType::QUANT_INT8_SYMM, scale, zero_point};
    inputs[i] = std::make_unique<MockUpTensor>(shapes[i], type_info, Layout::NHWC, input_pads[i]);
    inputs[i]->setBuffer(reinterpret_cast<uint8_t *>(input_buffer));

    outputs[i] = std::make_unique<MockUpTensor>(shapes[i], TypeInfo(DataType::FLOAT32),
                                                Layout::NHWC, output_pads[i]);
    output_buffers[i] = std::make_unique<uint8_t[]>(outputs[i]->total_size());
    outputs[i]->setBuffer(output_buffers[i].get());
  }

  auto mockup_layer = std::make_unique<MockUpLayer>(
    std::vector<ITensor *>{inputs[0].get(), inputs[1].get(), inputs[2].get(), inputs[3].get()},
    std::vector<ITensor *>{outputs[0].get(), outputs[1].get(), outputs[2].get(), outputs[3].get()});
  mockup_layer->run();

  for (size_t i = 0; i < 4; ++i)
  {
    for (int32_t j = 0; j < shapes[i].dim(0); ++j)
    {
      for (int32_t k = 0; k < shapes[i].dim(1); ++k)
      {
        for (int32_t l = 0; l < shapes[i].dim(2); ++l)
        {
          for (int32_t m = 0; m < shapes[i].dim(3); ++m)
          {
            Coordinates coords{j, k, l, m};
            float result =
              *reinterpret_cast<float *>(outputs[i]->buffer() + outputs[i]->calcOffset(coords));
            int8_t qasymm8 =
              *reinterpret_cast<int8_t *>(inputs[i]->buffer() + inputs[i]->calcOffset(coords));
            float expected = (qasymm8 - zero_point) * scale;
            EXPECT_EQ(result, expected);
          }
        }
      }
    }
  }
}

TEST(IPermuteFunction, qsymm16_to_float)
{
  const size_t input_pads[4] = {0, 0, 1, 2};
  const size_t output_pads[4] = {0, 3, 2, 1};
  const std::vector<Shape> shapes{{1, 1, 4, 1}, {2, 1, 2, 3}, {1, 2, 1, 2}, {1, 1, 2, 3}};
  float expected_buffer[] = {10, 0, -10, -20, 30, -40, 50, -60, 70, -80, 90, -100};
  float scale = 10;
  int32_t zero_point = 0;
  uint8_t input_buffer[12];

  int32_t min_val = std::numeric_limits<int16_t>::min();
  int32_t max_val = std::numeric_limits<int16_t>::max();
  for (int32_t i = 0; i < sizeof(expected_buffer) / sizeof(float); ++i)
  {
    int32_t unclamped = static_cast<int32_t>(std::round(expected_buffer[i] / scale)) + zero_point;
    input_buffer[i] = std::min(std::max(unclamped, min_val), max_val);
  }

  std::vector<std::unique_ptr<MockUpTensor>> inputs(4);
  std::vector<std::unique_ptr<MockUpTensor>> outputs(4);
  std::vector<std::unique_ptr<uint8_t[]>> output_buffers(4);
  for (size_t i = 0; i < 4; ++i)
  {
    TypeInfo type_info{DataType::QUANT_INT16_SYMM, scale, zero_point};
    inputs[i] = std::make_unique<MockUpTensor>(shapes[i], type_info, Layout::NHWC, input_pads[i]);
    inputs[i]->setBuffer(reinterpret_cast<uint8_t *>(input_buffer));

    outputs[i] = std::make_unique<MockUpTensor>(shapes[i], TypeInfo(DataType::FLOAT32),
                                                Layout::NHWC, output_pads[i]);
    output_buffers[i] = std::make_unique<uint8_t[]>(outputs[i]->total_size());
    outputs[i]->setBuffer(output_buffers[i].get());
  }

  auto mockup_layer = std::make_unique<MockUpLayer>(
    std::vector<ITensor *>{inputs[0].get(), inputs[1].get(), inputs[2].get(), inputs[3].get()},
    std::vector<ITensor *>{outputs[0].get(), outputs[1].get(), outputs[2].get(), outputs[3].get()});
  mockup_layer->run();

  for (size_t i = 0; i < 4; ++i)
  {
    for (int32_t j = 0; j < shapes[i].dim(0); ++j)
    {
      for (int32_t k = 0; k < shapes[i].dim(1); ++k)
      {
        for (int32_t l = 0; l < shapes[i].dim(2); ++l)
        {
          for (int32_t m = 0; m < shapes[i].dim(3); ++m)
          {
            Coordinates coords{j, k, l, m};
            float result =
              *reinterpret_cast<float *>(outputs[i]->buffer() + outputs[i]->calcOffset(coords));
            int16_t qasymm8 =
              *reinterpret_cast<int16_t *>(inputs[i]->buffer() + inputs[i]->calcOffset(coords));
            float expected = (qasymm8 - zero_point) * scale;
            EXPECT_EQ(result, expected);
          }
        }
      }
    }
  }
}

TEST(IPermuteFunction, float_qasymm8_layout)
{
  // float -> quasymm8
  {
    const size_t input_pads[4] = {0, 0, 1, 2};
    const size_t output_pads[4] = {0, 3, 2, 1};
    const std::vector<Shape> shapes{{1, 1, 4, 1}, {2, 1, 2, 3}, {1, 2, 1, 2}, {1, 1, 2, 3}};
    float expected_buffer[] = {10,  0,  -10,  -20, 30,   -40, 50,   -60, 70,
                               -80, 90, -100, 110, -120, 130, -140, 150, -160};
    float scale = 10;
    int32_t zero_point = 128;

    std::vector<std::unique_ptr<MockUpTensor>> inputs(4);
    std::vector<std::unique_ptr<MockUpTensor>> outputs(4);
    std::vector<std::unique_ptr<uint8_t[]>> output_buffers(4);
    for (size_t i = 0; i < 4; ++i)
    {
      Layout layout = Layout::NHWC;
      Shape shape = shapes[i];
      if (i % 2 == 1)
      {
        layout = Layout::NCHW;
        shape = Shape{shapes[i].dim(0), shapes[i].dim(3), shapes[i].dim(1), shapes[i].dim(2)};
      }
      inputs[i] =
        std::make_unique<MockUpTensor>(shape, TypeInfo(DataType::FLOAT32), layout, input_pads[i]);
      inputs[i]->setBuffer(reinterpret_cast<uint8_t *>(expected_buffer));

      if (layout == Layout::NHWC)
      {
        layout = Layout::NCHW;
        shape = Shape{shapes[i].dim(0), shapes[i].dim(3), shapes[i].dim(1), shapes[i].dim(2)};
      }
      else
      {
        layout = Layout::NHWC;
        shape = shapes[i];
      }
      TypeInfo type_info{DataType::QUANT_UINT8_ASYMM, scale, zero_point};
      outputs[i] = std::make_unique<MockUpTensor>(shape, type_info, layout, output_pads[i]);
      output_buffers[i] = std::make_unique<uint8_t[]>(outputs[i]->total_size());
      outputs[i]->setBuffer(output_buffers[i].get());
    }

    auto mockup_layer = std::make_unique<MockUpLayer>(
      std::vector<ITensor *>{inputs[0].get(), inputs[1].get(), inputs[2].get(), inputs[3].get()},
      std::vector<ITensor *>{outputs[0].get(), outputs[1].get(), outputs[2].get(),
                             outputs[3].get()});
    mockup_layer->run();

    for (size_t i = 0; i < 4; ++i)
    {
      for (int32_t j = 0; j < shapes[i].dim(0); ++j)
      {
        for (int32_t k = 0; k < shapes[i].dim(1); ++k)
        {
          for (int32_t l = 0; l < shapes[i].dim(2); ++l)
          {
            for (int32_t m = 0; m < shapes[i].dim(3); ++m)
            {
              Coordinates input_coords;
              Coordinates output_coords;
              if (inputs[i]->layout() == Layout::NHWC)
              {
                input_coords = Coordinates{j, k, l, m};
              }
              else
              {
                input_coords = Coordinates{j, m, k, l};
              }
              if (outputs[i]->layout() == Layout::NHWC)
              {
                output_coords = Coordinates{j, k, l, m};
              }
              else
              {
                output_coords = Coordinates{j, m, k, l};
              }
              uint8_t qasymm8 = *reinterpret_cast<uint8_t *>(outputs[i]->buffer() +
                                                             outputs[i]->calcOffset(output_coords));
              float result = (qasymm8 - zero_point) * scale;
              float expected = *reinterpret_cast<float *>(inputs[i]->buffer() +
                                                          inputs[i]->calcOffset(input_coords));
              EXPECT_EQ(result, expected);
            }
          }
        }
      }
    }
  }

  // qasymm8 -> float
  {
    const size_t input_pads[4] = {0, 0, 1, 2};
    const size_t output_pads[4] = {0, 3, 2, 1};
    const std::vector<Shape> shapes{{1, 1, 4, 1}, {2, 1, 2, 3}, {1, 2, 1, 2}, {1, 1, 2, 3}};
    float expected_buffer[] = {10,  0,  -10,  -20, 30,   -40, 50,   -60, 70,
                               -80, 90, -100, 110, -120, 130, -140, 150, -160};
    float scale = 10;
    int32_t zero_point = 128;
    uint8_t input_buffer[18];

    int32_t min_val = std::numeric_limits<int16_t>::min();
    int32_t max_val = std::numeric_limits<int16_t>::max();
    for (int32_t i = 0; i < sizeof(expected_buffer) / sizeof(float); ++i)
    {
      int32_t unclamped = static_cast<int32_t>(std::round(expected_buffer[i] / scale)) + zero_point;
      input_buffer[i] = std::min(std::max(unclamped, min_val), max_val);
    }

    std::vector<std::unique_ptr<MockUpTensor>> inputs(4);
    std::vector<std::unique_ptr<MockUpTensor>> outputs(4);
    std::vector<std::unique_ptr<uint8_t[]>> output_buffers(4);
    for (size_t i = 0; i < 4; ++i)
    {
      Layout layout = Layout::NHWC;
      Shape shape = shapes[i];
      if (i % 2 == 1)
      {
        layout = Layout::NCHW;
        shape = Shape{shapes[i].dim(0), shapes[i].dim(3), shapes[i].dim(1), shapes[i].dim(2)};
      }
      TypeInfo type_info{DataType::QUANT_UINT8_ASYMM, scale, zero_point};
      inputs[i] = std::make_unique<MockUpTensor>(shape, type_info, layout, input_pads[i]);
      inputs[i]->setBuffer(reinterpret_cast<uint8_t *>(expected_buffer));

      if (layout == Layout::NHWC)
      {
        layout = Layout::NCHW;
        shape = Shape{shapes[i].dim(0), shapes[i].dim(3), shapes[i].dim(1), shapes[i].dim(2)};
      }
      else
      {
        layout = Layout::NHWC;
        shape = shapes[i];
      }
      outputs[i] =
        std::make_unique<MockUpTensor>(shape, TypeInfo(DataType::FLOAT32), layout, output_pads[i]);
      output_buffers[i] = std::make_unique<uint8_t[]>(outputs[i]->total_size());
      outputs[i]->setBuffer(output_buffers[i].get());
    }

    auto mockup_layer = std::make_unique<MockUpLayer>(
      std::vector<ITensor *>{inputs[0].get(), inputs[1].get(), inputs[2].get(), inputs[3].get()},
      std::vector<ITensor *>{outputs[0].get(), outputs[1].get(), outputs[2].get(),
                             outputs[3].get()});
    mockup_layer->run();

    for (size_t i = 0; i < 4; ++i)
    {
      for (int32_t j = 0; j < shapes[i].dim(0); ++j)
      {
        for (int32_t k = 0; k < shapes[i].dim(1); ++k)
        {
          for (int32_t l = 0; l < shapes[i].dim(2); ++l)
          {
            for (int32_t m = 0; m < shapes[i].dim(3); ++m)
            {
              Coordinates input_coords;
              Coordinates output_coords;
              if (inputs[i]->layout() == Layout::NHWC)
              {
                input_coords = Coordinates{j, k, l, m};
              }
              else
              {
                input_coords = Coordinates{j, m, k, l};
              }
              if (outputs[i]->layout() == Layout::NHWC)
              {
                output_coords = Coordinates{j, k, l, m};
              }
              else
              {
                output_coords = Coordinates{j, m, k, l};
              }
              float result = *reinterpret_cast<float *>(outputs[i]->buffer() +
                                                        outputs[i]->calcOffset(output_coords));
              uint8_t qasymm8 = *reinterpret_cast<uint8_t *>(inputs[i]->buffer() +
                                                             inputs[i]->calcOffset(input_coords));
              float expected = (qasymm8 - zero_point) * scale;
              EXPECT_EQ(result, expected);
            }
          }
        }
      }
    }
  }
}

} // namespace
