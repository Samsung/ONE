/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Reader.h"

#include "../MockTensor.test.h"

#include <gtest/gtest.h>

using namespace onert::exec::feature;

template <typename T> class Reader_nhwc : public testing::Test
{
public:
  void setData(std::initializer_list<T> list) { _data = std::make_shared<std::vector<T>>(list); }

  void setShape(int32_t batch, int32_t depth, int32_t height, int32_t width)
  {
    _shape = onert::ir::FeatureShape(batch, depth, height, width);
  }

  void setStride(int32_t batch, int32_t depth, int32_t height, int32_t width)
  {
    auto elem_size = sizeof(T);
    _stride = onert::ir::FeatureShape(batch * elem_size, depth * elem_size, height * elem_size,
                                      width * elem_size);
  }

  void createReader()
  {
    _reader =
      std::make_shared<nhwc::Reader<T>>(_shape, _stride, _data->data(), _data->size() * sizeof(T));
  }

  void createUsingMockTensor()
  {
    onert::ir::Shape shape = {_shape.N, _shape.H, _shape.W, _shape.C};
    _tensor = std::make_shared<MockTensor<T>>(shape, _data->data(), onert::ir::Layout::NHWC);
    _reader = std::make_shared<nhwc::Reader<T>>(_tensor.get());
  }

  std::shared_ptr<nhwc::Reader<T>> _reader = nullptr;

private:
  std::shared_ptr<std::vector<T>> _data = nullptr;
  onert::ir::FeatureShape _shape;
  onert::ir::FeatureShape _stride;
  std::shared_ptr<MockTensor<T>> _tensor = nullptr;
};

using ReaderTypes = ::testing::Types<float, int32_t, uint8_t, int8_t, int16_t>;
TYPED_TEST_SUITE(Reader_nhwc, ReaderTypes);
TYPED_TEST_SUITE(MockTensorReader_nhwc, ReaderTypes);

TYPED_TEST(Reader_nhwc, basic_reader)
{
  this->setData({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  this->setShape(1, 2, 3, 2);
  this->setStride(12, 1, 6, 2);
  this->createReader();

  // Data: NCHW
  // Shape: NHWC
  ASSERT_EQ(this->_reader->at(0, 1, 1, 0), 8);
  ASSERT_EQ(this->_reader->at(1, 1, 0), 8);

  // Data: NHWC
  // Shape: NHWC
  this->createUsingMockTensor();

  ASSERT_EQ(this->_reader->at(0, 1, 1, 0), 6);
  ASSERT_EQ(this->_reader->at(1, 1, 0), 6);
}
