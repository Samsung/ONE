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

#include "mio_tflite2121/Helper.h"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <vector>

class mio_tflite2121_helper_test : public ::testing::Test
{
protected:
  void initialization_finish(void)
  {
    _fbb.Finish(tflite::CreateModelDirect(_fbb, 0, &_opcodes_vec));
  }

protected:
  void add_operator_code(int8_t deprecated_builtin_code, const char *custom_code,
                         tflite::BuiltinOperator builtin_code)
  {
    _opcodes_vec.push_back(tflite::CreateOperatorCodeDirect(
      _fbb, deprecated_builtin_code, custom_code, 1 /* version */, builtin_code));
  }

  const tflite::OperatorCode *get_operator_code(uint8_t idx)
  {
    return tflite::GetModel(_fbb.GetBufferPointer())->operator_codes()->Get(idx);
  }

private:
  flatbuffers::FlatBufferBuilder _fbb;
  std::vector<flatbuffers::Offset<tflite::OperatorCode>> _opcodes_vec;
};

/**
 * Extended 'builtin_code' is not in TFLite schema v3.
 *
 * Thus it is filled with 0(BuiltinOperator_ADD) in schame v3. Please refer to
 * https://github.com/tensorflow/tensorflow/blob/1ab788fa8d08430be239ab970980b891ad7af494/tensorflow/lite/schema/schema_utils.cc#L28-L31
 */
TEST_F(mio_tflite2121_helper_test, v3)
{
  // BuiltinOperator_ADD = 0
  // BuiltinOperator_CONV_2D = 3
  add_operator_code(3, "", tflite::BuiltinOperator_ADD);
  initialization_finish();

  ASSERT_TRUE(mio::tflite::is_valid(get_operator_code(0)));
  ASSERT_EQ(mio::tflite::builtin_code_neutral(get_operator_code(0)),
            tflite::BuiltinOperator_CONV_2D);
  ASSERT_FALSE(mio::tflite::is_custom(get_operator_code(0)));
}

TEST_F(mio_tflite2121_helper_test, v3_custom)
{
  // BuiltinOperator_ADD = 0
  // BuiltinOperator_CUSTOM = 32
  add_operator_code(32, "custom", tflite::BuiltinOperator_ADD);
  initialization_finish();

  ASSERT_TRUE(mio::tflite::is_valid(get_operator_code(0)));
  ASSERT_EQ(mio::tflite::builtin_code_neutral(get_operator_code(0)),
            tflite::BuiltinOperator_CUSTOM);
  ASSERT_TRUE(mio::tflite::is_custom(get_operator_code(0)));
}

TEST_F(mio_tflite2121_helper_test, v3_NEG)
{
  // BuiltinOperator_ADD = 0
  // BuiltinOperator_CUMSUM = 128
  // deprecated_builtin_code cannot be negative value
  add_operator_code(128, "", tflite::BuiltinOperator_ADD);
  initialization_finish();

  ASSERT_FALSE(mio::tflite::is_valid(get_operator_code(0)));
}

TEST_F(mio_tflite2121_helper_test, v3a_under127)
{
  // BuiltinOperator_CONV_2D = 3
  add_operator_code(3, "", tflite::BuiltinOperator_CONV_2D);
  initialization_finish();

  ASSERT_TRUE(mio::tflite::is_valid(get_operator_code(0)));
  ASSERT_EQ(mio::tflite::builtin_code_neutral(get_operator_code(0)),
            tflite::BuiltinOperator_CONV_2D);
  ASSERT_FALSE(mio::tflite::is_custom(get_operator_code(0)));
}

TEST_F(mio_tflite2121_helper_test, v3a_under127_NEG)
{
  // BuiltinOperator_CONV_2D = 3
  // BuiltinOperator_CUMSUM = 128
  // deprecated_builtin_code cannot be negative value
  add_operator_code(128, "", tflite::BuiltinOperator_CONV_2D);
  initialization_finish();

  ASSERT_FALSE(mio::tflite::is_valid(get_operator_code(0)));
}

TEST_F(mio_tflite2121_helper_test, v3a_custom)
{
  // BuiltinOperator_CUSTOM = 32
  add_operator_code(32, "custom", tflite::BuiltinOperator_CUSTOM);
  initialization_finish();

  ASSERT_TRUE(mio::tflite::is_valid(get_operator_code(0)));
  ASSERT_EQ(mio::tflite::builtin_code_neutral(get_operator_code(0)),
            tflite::BuiltinOperator_CUSTOM);
  ASSERT_TRUE(mio::tflite::is_custom(get_operator_code(0)));
}

TEST_F(mio_tflite2121_helper_test, v3a_custom_NEG)
{
  // BuiltinOperator_CUMSUM = 128
  // deprecated_builtin_code cannot be negative value
  add_operator_code(128, "custom", tflite::BuiltinOperator_CUSTOM);
  initialization_finish();

  ASSERT_FALSE(mio::tflite::is_valid(get_operator_code(0)));
}

TEST_F(mio_tflite2121_helper_test, v3a_over127)
{
  // BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES = 127
  // BuiltinOperator_CUMSUM = 128
  add_operator_code(127, "", tflite::BuiltinOperator_CUMSUM);
  initialization_finish();

  ASSERT_TRUE(mio::tflite::is_valid(get_operator_code(0)));
  ASSERT_EQ(mio::tflite::builtin_code_neutral(get_operator_code(0)),
            tflite::BuiltinOperator_CUMSUM);
  ASSERT_FALSE(mio::tflite::is_custom(get_operator_code(0)));
}

TEST_F(mio_tflite2121_helper_test, v3a_over127_NEG)
{
  // BuiltinOperator_CUMSUM = 128
  // deprecated_builtin_code cannot be negative value
  add_operator_code(128, "", tflite::BuiltinOperator_CUMSUM);
  initialization_finish();

  ASSERT_FALSE(mio::tflite::is_valid(get_operator_code(0)));
}
