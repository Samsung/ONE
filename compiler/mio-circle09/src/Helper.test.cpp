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

#include "mio_circle/Helper.h"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <vector>

class mio_circle09_helper_test : public ::testing::Test
{
protected:
  void initialization_finish(void)
  {
    _fbb.Finish(circle::CreateModelDirect(_fbb, 0, &_opcodes_vec));
  }

protected:
  void add_operator_code(int8_t deprecated_builtin_code, const char *custom_code,
                         circle::BuiltinOperator builtin_code)
  {
    _opcodes_vec.push_back(circle::CreateOperatorCodeDirect(
      _fbb, deprecated_builtin_code, custom_code, 1 /* version */, builtin_code));
  }

  const circle::OperatorCode *get_operator_code(uint8_t idx)
  {
    return circle::GetModel(_fbb.GetBufferPointer())->operator_codes()->Get(idx);
  }

private:
  flatbuffers::FlatBufferBuilder _fbb;
  std::vector<flatbuffers::Offset<circle::OperatorCode>> _opcodes_vec;
};

TEST_F(mio_circle09_helper_test, v09)
{
  // BuiltinOperator_ADD = 0
  // BuiltinOperator_CONV_2D = 3
  add_operator_code(3, "", circle::BuiltinOperator_ADD);
  initialization_finish();

  ASSERT_TRUE(mio::circle::is_valid(get_operator_code(0)));
  ASSERT_EQ(mio::circle::builtin_code_neutral(get_operator_code(0)),
            circle::BuiltinOperator_CONV_2D);
  ASSERT_FALSE(mio::circle::is_custom(get_operator_code(0)));
}

TEST_F(mio_circle09_helper_test, v09_custom_old)
{
  // BuiltinOperator_ADD = 0
  // BuiltinOperator_CUSTOM = 32
  add_operator_code(32, "custom", circle::BuiltinOperator_ADD);
  initialization_finish();

  ASSERT_TRUE(mio::circle::is_valid(get_operator_code(0)));
  ASSERT_EQ(mio::circle::builtin_code_neutral(get_operator_code(0)),
            circle::BuiltinOperator_CUSTOM);
  ASSERT_TRUE(mio::circle::is_custom(get_operator_code(0)));
}

TEST_F(mio_circle09_helper_test, v09_NEG)
{
  // BuiltinOperator_ADD = 0
  // BuiltinOperator_CUMSUM = 128
  // deprecated_builtin_code cannot be negative value
  add_operator_code(128, "", circle::BuiltinOperator_ADD);
  initialization_finish();

  ASSERT_FALSE(mio::circle::is_valid(get_operator_code(0)));
}

TEST_F(mio_circle09_helper_test, v09_under127)
{
  // BuiltinOperator_CONV_2D = 3
  add_operator_code(3, "", circle::BuiltinOperator_CONV_2D);
  initialization_finish();

  ASSERT_TRUE(mio::circle::is_valid(get_operator_code(0)));
  ASSERT_EQ(mio::circle::builtin_code_neutral(get_operator_code(0)),
            circle::BuiltinOperator_CONV_2D);
  ASSERT_FALSE(mio::circle::is_custom(get_operator_code(0)));
}

TEST_F(mio_circle09_helper_test, v09_under127_NEG)
{
  // BuiltinOperator_CONV_2D = 3
  // BuiltinOperator_CUMSUM = 128
  // deprecated_builtin_code cannot be negative value
  add_operator_code(128, "", circle::BuiltinOperator_CONV_2D);
  initialization_finish();

  ASSERT_FALSE(mio::circle::is_valid(get_operator_code(0)));
}

TEST_F(mio_circle09_helper_test, v09_custom)
{
  // BuiltinOperator_CUSTOM = 32
  add_operator_code(32, "custom", circle::BuiltinOperator_CUSTOM);
  initialization_finish();

  ASSERT_TRUE(mio::circle::is_valid(get_operator_code(0)));
  ASSERT_EQ(mio::circle::builtin_code_neutral(get_operator_code(0)),
            circle::BuiltinOperator_CUSTOM);
  ASSERT_TRUE(mio::circle::is_custom(get_operator_code(0)));
}

TEST_F(mio_circle09_helper_test, v09_custom_NEG)
{
  // BuiltinOperator_CUMSUM = 128
  // deprecated_builtin_code cannot be negative value
  add_operator_code(128, "custom", circle::BuiltinOperator_CUSTOM);
  initialization_finish();

  ASSERT_FALSE(mio::circle::is_valid(get_operator_code(0)));
}

TEST_F(mio_circle09_helper_test, v09_over127)
{
  // BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES = 127
  // BuiltinOperator_CUMSUM = 128
  add_operator_code(127, "", circle::BuiltinOperator_CUMSUM);
  initialization_finish();

  ASSERT_TRUE(mio::circle::is_valid(get_operator_code(0)));
  ASSERT_EQ(mio::circle::builtin_code_neutral(get_operator_code(0)),
            circle::BuiltinOperator_CUMSUM);
  ASSERT_FALSE(mio::circle::is_custom(get_operator_code(0)));
}

TEST_F(mio_circle09_helper_test, v09_over127_NEG)
{
  // BuiltinOperator_CUMSUM = 128
  // deprecated_builtin_code cannot be negative value
  add_operator_code(128, "", circle::BuiltinOperator_CUMSUM);
  initialization_finish();

  ASSERT_FALSE(mio::circle::is_valid(get_operator_code(0)));
}
