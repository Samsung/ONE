/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "mio_circle/Reader.h"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

class mio_circle09_reader_test : public ::testing::Test
{
protected:
  void initialization_emty(void)
  {
    _model = circle::CreateModelDirect(_fbb, 0, &_opcodes_vec);
    circle::FinishModelBuffer(_fbb, _model);
  }

  const circle::Model *circleModel(void)
  {
    auto ptr = _fbb.GetBufferPointer();
    return circle::GetModel(ptr);
  }

private:
  flatbuffers::FlatBufferBuilder _fbb;
  flatbuffers::Offset<circle::Model> _model;
  std::vector<flatbuffers::Offset<circle::OperatorCode>> _opcodes_vec;
};

TEST_F(mio_circle09_reader_test, null_Model_NEG)
{
  EXPECT_THROW(mio::circle::Reader reader(nullptr), std::runtime_error);
}

TEST_F(mio_circle09_reader_test, empty_Model)
{
  initialization_emty();

  const circle::Model *model = circleModel();
  EXPECT_NE(nullptr, model);

  mio::circle::Reader reader(model);

  SUCCEED();
}

// TODO add more tests
