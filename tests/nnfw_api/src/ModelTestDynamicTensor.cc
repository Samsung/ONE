/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <gtest/gtest.h>
#include <nnfw_internal.h>

#include "common.h"
#include "fixtures.h"
#include "CircleGen.h"
#include "GenModelTest.h"
#include "NNPackages.h"

// This macro can be used instead of using NNFW_ENSURE_SUCCESS especially with negative test.
// E.g., setInputOutput() is written with this macro and the following check is available to check
// if there's any error while setting input or output:
//
//  EXPECT_ANY_THROW(setInputOutput(...));
//
#define THROW_WHEN_NNFW_ERROR(result)                                  \
  do                                                                   \
  {                                                                    \
    if (result != NNFW_STATUS_NO_ERROR)                                \
      throw std::runtime_error("returning error on calling nnfw api"); \
  } while (false)

template <class CPP_TYPE> struct nnfw_type;

template <> struct nnfw_type<float>
{
  static const NNFW_TYPE dtype = NNFW_TYPE_TENSOR_FLOAT32;
};

template <> struct nnfw_type<int32_t>
{
  static const NNFW_TYPE dtype = NNFW_TYPE_TENSOR_INT32;
};

// TODO Add more struct nnfw_type for other types when needed

template <class T_INPUT, class T_OUT>
void setInputOutput(nnfw_session *session, const std::vector<T_INPUT> &input,
                    std::vector<T_OUT> &actual_output)
{
  NNFW_STATUS result;
  result = nnfw_set_input(session, 0, nnfw_type<T_INPUT>::dtype, input.data(),
                          sizeof(T_INPUT) * input.size());
  THROW_WHEN_NNFW_ERROR(result);

  result = nnfw_set_output(session, 0, nnfw_type<T_OUT>::dtype, actual_output.data(),
                           sizeof(T_OUT) * actual_output.size());
  THROW_WHEN_NNFW_ERROR(result);
}

template <class T_INPUT0, class T_INPUT1, class T_OUT>
void setInputOutput(nnfw_session *session, const std::vector<T_INPUT0> &input0,
                    const std::vector<T_INPUT1> &input1, std::vector<T_OUT> &actual_output)
{
  NNFW_STATUS result;
  result = nnfw_set_input(session, 0, nnfw_type<T_INPUT0>::dtype, input0.data(),
                          sizeof(T_INPUT0) * input0.size());
  THROW_WHEN_NNFW_ERROR(result);

  result = nnfw_set_input(session, 1, nnfw_type<T_INPUT1>::dtype, input1.data(),
                          sizeof(T_INPUT1) * input1.size());
  THROW_WHEN_NNFW_ERROR(result);

  result = nnfw_set_output(session, 0, nnfw_type<T_OUT>::dtype, actual_output.data(),
                           sizeof(T_OUT) * actual_output.size());
  THROW_WHEN_NNFW_ERROR(result);
}

template <class T_OUTPUT>
void verifyOutput(nnfw_session *session, const nnfw_tensorinfo expected_ti,
                  const std::vector<T_OUTPUT> &expected, const std::vector<T_OUTPUT> &actual)
{
  uint32_t output_num = -1;
  nnfw_tensorinfo t_out;
  NNFW_ENSURE_SUCCESS(nnfw_output_size(session, &output_num));
  NNFW_ENSURE_SUCCESS(nnfw_output_tensorinfo(session, 0, &t_out));

  ASSERT_EQ(output_num, 1);

  // nnfw_tensorinfo of output
  tensorInfoEqual(t_out, expected_ti);

  // value of output
  ASSERT_EQ(expected.size(), actual.size());
  for (int i = 0; i < expected.size(); i++)
  {
    bool is_output_float = std::is_same<T_OUTPUT, float>::value;
    if (is_output_float)
      ASSERT_FLOAT_EQ(expected[i], actual[i]);
    else
      ASSERT_EQ(expected[i], actual[i]);
  }
}

/**
 * @brief Testing the following model:
 *
 * Testing the following model:
 *       #1 = const(value = [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5], shape=[2, 3])
 *       #2 = placeholder (shape = [2])      <-------- this is an input
 *       #3 = reshape(#1, #2)
 *
 * @note Run this test with "cpu" backend
 */
auto build_dynamic_Reshape()
{
  CircleGen cgen;

  auto f32 = circle::TensorType::TensorType_FLOAT32;
  auto i32 = circle::TensorType::TensorType_INT32;

  std::vector<float> new_shape_data{-1.5, -1.0, -0.5, 0.5, 1.0, 1.5};
  uint32_t input_buf = cgen.addBuffer(new_shape_data); // shape = [2, 3]
  int input = cgen.addTensor({{2, 3}, f32, input_buf});
  int new_shape = cgen.addTensor({{2}, i32});
  int out = cgen.addTensor({{}, f32}); // scalar, meaning output shape is unspecified

  CircleGen::Shape empty_new_shape;
  cgen.addOperatorReshape({{input, new_shape}, {out}}, &empty_new_shape);
  cgen.setInputsAndOutputs({new_shape}, {out});
  auto cbuf = cgen.finish();
  return cbuf;
}

TEST_F(GenModelTest, dynamic_reshape_from_2x3_to_3x2)
{
  const std::vector<int> new_shape{3, 2};
  const std::vector<float> expected{-1.5, -1.0, -0.5, 0.5, 1.0, 1.5};

  _context = std::make_unique<GenModelTestContext>(build_dynamic_Reshape());
  {
    _context->addTestCase(TestCaseData{}.addInput(new_shape).addOutput(expected));
    _context->setBackends({"cpu"}); // Currently, dynamic tensor runs on "cpu" only
    _context->output_sizes(0, sizeof(float) * expected.size());
  }
  // GenModelTest::teardown() will do the rest
  SUCCEED();
}

/**
 * @brief Negative test.
 *        Reshape's first input has 6 values but trying to reshaping to [3, 3]
 */
TEST_F(GenModelTest, neg_reshape_from_2x3_to_wrong_3x3)
{
  const std::vector<int> wrong_shape{3, 3}; // wrong shape input
  const std::vector<float> expected{0};     // whatever

  _context = std::make_unique<GenModelTestContext>(build_dynamic_Reshape());
  {

    _context->addTestCase(TestCaseData{}.addInput(wrong_shape).addOutput(expected).expectFailRun());
    _context->setBackends({"cpu"}); // Currently, dynamic tensor runs on "cpu" only
    _context->output_sizes(0, sizeof(float) * expected.size());
  }
  // GenModelTest::teardown() will do the rest
  SUCCEED();
}

TEST_F(GenModelTest, reshape_multiple_executions)
{
  std::vector<int> new_shape;
  std::vector<float> expected = {-1.5, -1.0, -0.5, 0.5, 1.0, 1.5};

  _context = std::make_unique<GenModelTestContext>(build_dynamic_Reshape());
  {
    _context->addTestCase(TestCaseData{}.addInput<int>({3, 2}).addOutput(expected));
    _context->addTestCase(TestCaseData{}.addInput<int>({1, 6}).addOutput(expected));
    _context->addTestCase(TestCaseData{}.addInput<int>({6, 1}).addOutput(expected));

    _context->setBackends({"cpu"}); // Currently, dynamic tensor runs on "cpu" only
    _context->output_sizes(0, sizeof(float) * expected.size());
  }
  // GenModelTest::teardown() will do the rest
  SUCCEED();
}

TEST_F(GenModelTest, neg_reshape_multiple_executions)
{
  std::vector<int> new_shape;
  std::vector<float> expected = {-1.5, -1.0, -0.5, 0.5, 1.0, 1.5};

  auto add_tcd = [&](const decltype(new_shape) &&new_shape, bool expect_fail_on_run) {
    TestCaseData tcd;
    tcd.addInput(new_shape).addOutput(expected);
    if (expect_fail_on_run)
      tcd.expectFailRun();
    _context->addTestCase(tcd);
  };

  _context = std::make_unique<GenModelTestContext>(build_dynamic_Reshape());
  {
    bool EXPECT_FAIL_ON_RUN = true;
    bool EXPECT_SUCCESS_ON_RUN = !EXPECT_FAIL_ON_RUN;

    add_tcd({3, 2}, EXPECT_SUCCESS_ON_RUN);
    add_tcd({1, 100}, EXPECT_FAIL_ON_RUN); // 1th tcd. wrong shape
    add_tcd({6, 1}, EXPECT_SUCCESS_ON_RUN);

    _context->setBackends({"cpu"}); // Currently, dynamic tensor runs on "cpu" only
    _context->output_sizes(0, sizeof(float) * expected.size());
  }
  // GenModelTest::teardown() will do the rest
  SUCCEED();
}

//
// Unknown Dimension Test
//    Trying to set unknown dim to other value before calling nnfw_prepare()
//

/**
 * @brief Testing the following model:
 *
 *        #0 = placeholder([None, None])   # initially, shape is [1, 1]
 *        #1 = placeholder([2, 3])
 *        #2 = concat (#0, #1, axis=0)
 *
 *        Calling sequence:
 *        - nnfw_set_input_tensorinfo(#0, [1, 3])    # now, [1, 3]
 *        - nnfw_prepare()                 # this should work
 *        - nnfw_set_input()
 *        - nnfw_run()
 *
 * @note Run this test with "cpu" backend
 */
auto build_model_buf_Concatenation_unknwon_dims()
{
  // Model is not important
  CircleGen cgen;
  auto f32 = circle::TensorType::TensorType_FLOAT32;
  int in1 = cgen.addTensor({{1, 1}, f32}); // consider this [None, None]
  int in2 = cgen.addTensor({{2, 3}, f32});
  int out = cgen.addTensor({{}, f32}); // scalar, meaning output shape is unspecified
  cgen.addOperatorConcatenation({{in1, in2}, {out}}, 0, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in1, in2}, {out});
  auto cbuf = cgen.finish();
  return cbuf;
}

TEST(TestDynamicTensor, concat_unknown_dim_input0_to_2x3)
{
  nnfw_session *session = nullptr;
  NNFW_ENSURE_SUCCESS(nnfw_create_session(&session));
  const auto model_buf = build_model_buf_Concatenation_unknwon_dims();
  NNFW_ENSURE_SUCCESS(nnfw_load_circle_from_buffer(session, model_buf.buffer(), model_buf.size()));

  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(session, "cpu"));

  const std::vector<float> input0 = {1, 2, 3};          // of shape [1, 3]
  const std::vector<float> input1 = {4, 5, 6, 7, 8, 9}; // of shape [2, 3]

  const std::vector<float> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> actual_output(expected.size());

  // input reshaping to [1, 3]
  nnfw_tensorinfo ti = {NNFW_TYPE_TENSOR_FLOAT32, 2, {1, 3}};
  NNFW_ENSURE_SUCCESS(nnfw_set_input_tensorinfo(session, 0, &ti));
  NNFW_ENSURE_SUCCESS(nnfw_prepare(session));

  setInputOutput(session, input0, input1, actual_output);

  // Do inference
  NNFW_STATUS res = nnfw_run(session);
  NNFW_ENSURE_SUCCESS(res);

  verifyOutput(session, {NNFW_TYPE_TENSOR_FLOAT32, 2, {3, 3}}, expected, actual_output);
}

/**
 * @brief Negative Test: Testing the following model:
 *
 *        #0 = placeholder([None, None])         # initially, [1, 1]
 *        #1 = placeholder([2, 3])
 *        #2 = concat (#0, #1, axis=0)
 *
 *        Calling sequence:
 *        - nnfw_set_input tensorinfo(#0, [3, 1]) # now [3, 1]
 *        - nnfw_prepare()                        # should fail (shape mismatch)
 *        - nnfw_set_input()
 *        - nnfw_run()
 *
 * @note Run this test with "cpu" backend and "linear" executor
 */
TEST(TestDynamicTensor, neg_concat_input0_to_wrong_shape)
{
  nnfw_session *session = nullptr;
  NNFW_ENSURE_SUCCESS(nnfw_create_session(&session));
  const auto model_buf = build_model_buf_Concatenation_unknwon_dims();
  NNFW_ENSURE_SUCCESS(nnfw_load_circle_from_buffer(session, model_buf.buffer(), model_buf.size()));

  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(session, "cpu"));

  const std::vector<float> input0 = {1, 2, 3};          // of shape [3, 1], wrong shape
  const std::vector<float> input1 = {4, 5, 6, 7, 8, 9}; // of shape [2, 3]

  std::vector<float> actual_output(100); // whatever size

  // input reshaping to [3, 1]
  nnfw_tensorinfo ti = {NNFW_TYPE_TENSOR_FLOAT32, 2, {3, 1}};
  NNFW_ENSURE_SUCCESS(nnfw_set_input_tensorinfo(session, 0, &ti));

  ASSERT_EQ(nnfw_prepare(session), NNFW_STATUS_ERROR);
}

//
// test about calling nnfw_set_input_tensorinfo() after compilation
//

/**
 * @brief Testing the following model, which has a binary operation:
 *
 *        #0 = placeholder([])
 *        #1 = placeholder([1, 2, 3])
 *        #2 = add (#0, #1)
 *        #3 = add (#2, #2)
 *
 *        Calling sequence:
 *        - nnfw_prepare()
 *        - nnfw_set_input_tensorinfo(#0, [2, 2, 3]) // This will make #3 tensor's shape [2, 2, 3]
 *        - nnfw_set_input()
 *        - nnfw_run()
 *
 * @note Run this test with "cpu" backend
 */
auto build_model_buf_Add_unspecified_rank()
{
  // Model is not important
  CircleGen cgen;
  auto f32 = circle::TensorType::TensorType_FLOAT32;
  int in1 = cgen.addTensor({{}, f32}); // scalar, meaning shape is unspecified
  int in2 = cgen.addTensor({{1, 2, 3}, f32});
  int op_out = cgen.addTensor({{}, f32}); // unspecified
  int out = cgen.addTensor({{}, f32});    // unspecified
  cgen.addOperatorAdd({{in1, in2}, {op_out}}, circle::ActivationFunctionType_NONE);
  cgen.addOperatorAdd({{op_out, op_out}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in1, in2}, {out});
  auto cbuf = cgen.finish();
  return cbuf;
}

TEST(TestDynamicTensor, set_input_tensorinfo_after_compilation_add)
{
  nnfw_session *session = nullptr;
  NNFW_ENSURE_SUCCESS(nnfw_create_session(&session));
  const auto model_buf = build_model_buf_Add_unspecified_rank();
  NNFW_ENSURE_SUCCESS(nnfw_load_circle_from_buffer(session, model_buf.buffer(), model_buf.size()));

  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(session, "cpu"));

  // input reshaping to [2, 2, 3]
  nnfw_tensorinfo input0_ti = {NNFW_TYPE_TENSOR_FLOAT32, 3, {2, 2, 3}};

  std::vector<float> input0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> input1 = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  std::vector<float> actual_output(12);
  std::vector<float> expected_output = {1.1 * 2, 2.1 * 2, 3.1 * 2, 4.1 * 2,  5.1 * 2,  6.1 * 2,
                                        7.1 * 2, 8.1 * 2, 9.1 * 2, 10.1 * 2, 11.1 * 2, 12.1 * 2};

  NNFW_ENSURE_SUCCESS(nnfw_prepare(session));

  NNFW_ENSURE_SUCCESS(nnfw_set_input_tensorinfo(session, 0, &input0_ti));

  setInputOutput(session, input0, input1, actual_output);

  // Do inference
  NNFW_STATUS res = nnfw_run(session);
  NNFW_ENSURE_SUCCESS(res);

  verifyOutput(session, {NNFW_TYPE_TENSOR_FLOAT32, 3, {2, 3, 3}}, expected_output, actual_output);
}

/**
 * @brief Testing the following model, which has a unary operation:
 *
 *        #0 = placeholder(shape = [4, 4])
 *        #1 = neg (#0)
 *
 *        Calling sequence:
 *        - nnfw_prepare()
 *        - nnfw_set_input_tensorinfo(#0, [20, 50])
 *        - nnfw_set_input()
 *        - nnfw_run()
 *
 * @note Run this test with "cpu" backend
 */

auto build_model_buf_NEG()
{
  // Model is not important
  CircleGen cgen;
  int in = cgen.addTensor({{4, 4}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{4, 4}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorNeg({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});
  auto cbuf = cgen.finish();
  return cbuf;
}

TEST(TestDynamicTensor, set_input_tensorinfo_after_compilation_neg)
{
  nnfw_session *session = nullptr;
  NNFW_ENSURE_SUCCESS(nnfw_create_session(&session));
  const auto model_buf = build_model_buf_NEG();
  nnfw_load_circle_from_buffer(session, model_buf.buffer(), model_buf.size());

  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(session, "cpu"));

  nnfw_tensorinfo input0_ti_original = {NNFW_TYPE_TENSOR_FLOAT32, 2, {4, 4}};

  // input reshaping to [20, 50]
  nnfw_tensorinfo input0_ti;
  {
    input0_ti.dtype = NNFW_TYPE_TENSOR_FLOAT32;
    input0_ti.rank = 2;
    input0_ti.dims[0] = 20;
    input0_ti.dims[1] = 50;
  }

  std::vector<float> input0(20 * 50);
  std::vector<float> actual_output(20 * 50);
  std::vector<float> expected_output(20 * 50);

  for (int i = 0; i < input0.size(); i++)
  {
    input0[i] = i * 1.1;
    expected_output[i] = -1 * input0[i];
  }

  NNFW_ENSURE_SUCCESS(nnfw_prepare(session));

  // input shape check
  {
    nnfw_tensorinfo ti = {};
    NNFW_ENSURE_SUCCESS(nnfw_input_tensorinfo(session, 0, &ti));
    ASSERT_TRUE(tensorInfoEqual(input0_ti_original, ti));
  }

  NNFW_ENSURE_SUCCESS(nnfw_set_input_tensorinfo(session, 0, &input0_ti));

  // input shape check
  {
    nnfw_tensorinfo ti = {};
    NNFW_ENSURE_SUCCESS(nnfw_input_tensorinfo(session, 0, &ti));
    ASSERT_TRUE(tensorInfoEqual(input0_ti, ti));
  }

  setInputOutput(session, input0, actual_output);

  // Do inference
  NNFW_STATUS res = nnfw_run(session);
  NNFW_ENSURE_SUCCESS(res);

  // output value check
  verifyOutput(session, {NNFW_TYPE_TENSOR_FLOAT32, 2, {20, 50}}, expected_output, actual_output);
}

using TestWhileDynamicModelLoaded = ValidationTestModelLoaded<NNPackages::WHILE_DYNAMIC>;

// clang-format off
const static std::vector<float> while_dynamic_input0{ 0.4325029254, 0.7332934141, 0.2969786823, 0.1540192217, 0.4608841240, 0.1523699313, 0.4334940016, 0.1022945493, 0.6928671598, 0.5891978741, 0.8283287883, 0.7041553259, 0.5243381262, 0.5623597503, 0.3395180404, 0.3212788701, 0.5248492956, 0.2551939189, 0.1338981092, 0.6406514645, 0.7089318633, 0.8164196610, 0.7689018846, 0.3551857173, 0.7668499351, 0.4942102134, 0.7345644236, 0.4689270556, 0.3495515287, 0.0768318549, 0.0868133679, 0.7823525667, 0.0791761801, 0.4397472143, 0.8150953054, 0.5074489713, 0.0895665437, 0.9451501966, 0.1064314246, 0.8803006411, 0.9903403521, 0.1259460151, 0.1889930069, 0.7466737032, 0.0553287826, 0.9712036252, 0.6352610588, 0.6301708817, 0.3079694211, 0.5367568731, 0.4070350230, 0.6815373302, 0.6948529482, 0.6158187985, 0.1485853940, 0.9162485600, 0.3622985184, 0.2672208250, 0.3396688998, 0.4135381579, 0.6450354457, 0.2386536747, 0.7072004080, 0.5289406180, 0.0643024296, 0.1969666779, 0.8667400479, 0.3396836221, 0.5878564715, 0.4551178813, 0.4318033755, 0.4376230836, 0.8211942315, 0.0230764486, 0.9005268812, 0.2147378176, 0.6036583781, 0.7161545157, 0.8246262074, 0.2989832759, 0.5491395593, 0.9779474735, 0.2006554008, 0.8227099776, 0.6018718481, 0.0132929254, 0.2212856710, 0.2032340616, 0.3059777319, 0.9094917178, 0.5409486890, 0.5595687032, 0.2436837852, 0.5649250150, 0.6730466485, 0.4421939552, 0.1432305574, 0.7053307891, 0.6284835935, 0.9216189384, 0.8686438799, 0.8385053873, 0.6248987913, 0.7697140574, 0.9808958173, 0.7571622133, 0.2297872156, 0.4201298952, 0.1305913031, 0.4572514296, 0.3072260618, 0.4668756723, 0.1919649392, 0.2050305754, 0.6062370539, 0.0006580966, 0.6217135191, 0.5123317838, 0.7305839658, 0.0610331446, 0.3614645600, 0.6455501914, 0.2919872701, 0.6446499228, 0.6293424964, 0.6947519779, 0.2680567801, 0.9756787419, 0.6422977448, 0.6911727786, 0.0343145914, 0.4764069021, 0.0876256451, 0.2926266789, 0.0487026349, 0.3558900952, 0.7788275480, 0.8566400409, 0.4791142642, 0.0595066175, 0.9609330297, 0.4075229764, 0.8758037090, 0.3485401869, 0.7945867181, 0.3457054794, 0.3327955306, 0.2870546579, 0.5697714090, 0.6144676208, 0.3251711428, 0.2342026234, 0.4153896868, 0.2149699926, 0.1064170301, 0.7240911722, 0.8196219206, 0.0208647959, 0.3081029952, 0.5742419958, 0.3027088642, 0.5005563498, 0.1707910597, 0.3358575106, 0.2290909439, 0.7788143754, 0.7611069679, 0.3525909781, 0.2308424413, 0.2585839927, 0.5973339677, 0.3728699684, 0.4975571036, 0.0781342834, 0.7119221091, 0.3926881850, 0.5501778126, 0.7364945412, 0.4965503812, 0.8785862923, 0.6024044752, 0.2638861239, 0.9093352556, 0.9069826007, 0.0359279662, 0.4043401778, 0.3457658887, 0.1013033912, 0.1810855120, 0.4946146905, 0.0194541160, 0.5453770161, 0.7965603471, 0.5493819714, 0.2422309667, 0.8376919031, 0.8350337148, 0.1898939908, 0.4576793313, 0.9535705447, 0.1353026628, 0.9474196434, 0.4256035388, 0.0255583692, 0.9593925476, 0.9245427847, 0.9780472517, 0.4356954992, 0.5673046708, 0.7346579432, 0.8614835143, 0.8782553673, 0.3395713866, 0.0013978065, 0.7640301585, 0.2504623234, 0.3626150787, 0.6888222694, 0.9404846430, 0.3519821763, 0.6855628490, 0.2415955663, 0.2107568830, 0.7718742490, 0.3419062793, 0.1280658394, 0.5126360059, 0.1722176671, 0.6543742418, 0.4206473231, 0.2138152719, 0.4514643848, 0.4293326437, 0.0042719250, 0.3195750117, 0.3874749541, 0.6262724400, 0.1620737463, 0.7417458892, 0.8521968126, 0.6405420303, 0.0713626966, 0.0474211276, 0.9068223834, 0.8541609645, 0.4279667437, 0.9738950133, 0.7167884707, 0.6812457442, 0.7938374281, 0.2077793330, 0.5163270831, 0.8487322927, 0.6320008039, 0.5116547942, 0.0056989277, 0.5253843665, 0.1517033428, 0.9921303988, 0.8305052519, 0.0771176443, 0.4621275961, 0.0299932379, 0.8129007220, 0.0946875364, 0.4544205368, 0.0143135618, 0.6373457313, 0.8202091455, 0.3447127640, 0.8560513258, 0.8079835773, 0.9697201252, 0.1521986276, 0.2269581258, 0.2245485932, 0.3396310210, 0.2649262249, 0.7799206972, 0.4020069242, 0.4444113672, 0.8123176098, 0.6460852027, 0.2041657269, 0.7889582515, 0.6526331902, 0.6626461744, 0.6049868464, 0.6901782155, 0.3364612758, 0.3053490818, 0.1905532777, 0.5362346172, 0.3618801832, 0.3485457003, 0.4509411156, 0.5986957550, 0.7858221531, 0.8822937012, 0.8280826807, 0.5261783004, 0.7312974334, 0.6962512732, 0.5243815780, 0.2492258698, 0.1734466404, 0.2547666430, 0.9950503111, 0.1781345457, 0.5630541444, 0.4552696049, 0.8874762058, 0.5965846777, 0.3575465977, 0.1213323772, 0.2790489793, 0.3157011569, 0.6218565702, 0.0304181967, 0.4112739265, 0.7361903787, 0.6753587723, 0.3667163849, 0.6275368929, 0.4185036719, 0.4791659117, 0.1246187463, 0.6651734114, 0.1778147966, 0.8796271682, 0.3000938296, 0.5996896029, 0.5020698309, 0.1601593345, 0.4467433393, 0.0287379269, 0.9011575580, 0.2722401917, 0.1642841995, 0.9468663335, 0.0238759480, 0.7811399102, 0.2070412934, 0.3746992052, 0.8473496437, 0.3498605192, 0.2693480551, 0.1523104310, 0.9660695791, 0.8762652278, 0.1654927284, 0.8743498921, 0.3143339157, 0.3896550536, 0.7256560922, 0.2408472896, 0.0930071324, 0.3269865215, 0.8070673347, 0.1218842566, 0.9943904281, 0.6901395917, 0.9491872787, 0.3617239892, 0.5459694862, 0.9408421516, 0.5354272127, 0.0377946161, 0.3319100142, 0.9823720455, 0.2373940945, 0.2439561784, 0.0767217800, 0.1102360934, 0.6404867172, 0.7430088520, 0.0165513344, 0.9841650128, 0.0532640740, 0.1635770351, 0.3721100390, 0.0598411299, 0.6548883319, 0.3812481761, 0.8741319180, 0.6431996226, 0.0550124273, 0.2009697258, 0.6922588348, 0.0673767105, 0.3385711610, 0.6945076585, 0.7870846987, 0.3323138356, 0.1601967812, 0.9595350623, 0.6049567461, 0.2068863660, 0.2562771440, 0.1041606516, 0.3444063365, 0.1464221030, 0.8932089210, 0.2040112168, 0.3407483399, 0.3251829743, 0.4777953327, 0.0534981787, 0.3613175154, 0.6707065105, 0.1188806742, 0.8228670359, 0.9907929897, 0.1556126177, 0.5561179519, 0.0124231419, 0.2054836601, 0.5855912566, 0.8455434442, 0.2268345803, 0.1841085702, 0.1096092239, 0.8316007257, 0.5046240687, 0.2195746899, 0.9222528338, 0.3633532226, 0.9383196831, 0.8803531528, 0.5124011636, 0.3909464478, 0.2731699646, 0.1102369502, 0.7489478588, 0.0600390583, 0.9290241599, 0.1041191891, 0.9347958565, 0.5584807396, 0.7331624031, 0.2267376930, 0.2868649662, 0.0016489516, 0.2301262319, 0.5107504129, 0.6500277519, 0.6766125560, 0.2019786686, 0.5890167952, 0.7182423472, 0.6890133023, 0.4442900419, 0.5760958791, 0.1364797056, 0.8246579766, 0.2527448535, 0.5444371700, 0.1561367512, 0.7551656961, 0.7171260715, 0.4264259040, 0.3883202970, 0.9166873693, 0.6557167768, 0.0264711548, 0.0761224255, 0.4693228602, 0.5476956964, 0.6261154413, 0.7666952610, 0.9579501152, 0.2581985295, 0.2322760671, 0.8342292905, 0.8143266439, 0.5771137476, 0.5815665126, 0.9772894382, 0.2359700650, 0.6501487494, 0.7841209769, 0.2793208659, 0.1745450795, 0.9626912475, 0.2373798192, 0.1235965416, 0.4632637799, 0.3763884604, 0.9971673489, 0.3533810079, 0.3203127384, 0.6102763414, 0.3859500289, 0.5929466486, 0.6658803821, 0.4130606949, 0.0352911949, 0.9713683128, 0.7546037436, 0.9780107737, 0.3970599473, 0.0187621433, 0.4941402078, 0.7670620680, 0.5360869765, 0.9634684920, 0.5996263027, 0.1895584762, 0.1214910895, 0.7381310463, 0.4301493466, 0.7403219938, 0.4817020297, 0.1843791455, 0.6473838091, 0.4138627350, 0.6825908422, 0.4481185675, 0.2030784935, 0.8468620777, 0.8059213758, 0.7525423169, 0.1854387224, 0.9046887755, 0.6654230952, 0.2029620409, 0.7164457440, 0.4172891080, 0.7797588110, 0.4135729969, 0.0026064927, 0.8375009894, 0.8355652690, 0.9187932014, 0.6724888086, 0.0276171323, 0.9106697440, 0.4562708735, 0.3417910039, 0.1569930464, 0.2029796541, 0.5049355626, 0.8143045306, 0.2432538420, 0.1068324223, 0.6258177757, 0.9749278426, 0.5378444791, 0.1657523215, 0.1930697113, 0.4833569825, 0.8000370264, 0.4315882921, 0.7571453452, 0.6069541574, 0.2073590159, 0.8702615499, 0.1951662153, 0.9303797483, 0.9241660833, 0.2795540988, 0.4241578877, 0.2383123934, 0.8627647758, 0.1700671613, 0.9635605216, 0.2514486313, 0.7766968012, 0.7126773596, 0.7009662986, 0.1317531914, 0.1318600327, 0.5509422421, 0.2159194350, 0.7851343751, 0.7231494188, 0.3523120880, 0.4999881089, 0.8202708960, 0.6340972185, 0.9181259274, 0.0057039275, 0.7197939754, 0.3580873907, 0.1026016176, 0.9657412767, 0.1973488480, 0.8099604845, 0.3302915096, 0.7635477781, 0.7097011805, 0.6271768212, 0.6583901644, 0.2334843278, 0.9448583126, 0.7434690595, 0.4068029821, 0.8815746307, 0.6311643124, 0.3891237080, 0.1507531852, 0.5215465426, 0.3248603344, 0.5837653279, 0.6689655185, 0.1362081915, 0.5130022764, 0.8519401550, 0.4397114217, 0.4129846096, 0.8706676960, 0.4183416367, 0.1135022715, 0.3501874208, 0.1142706573, 0.4111732543, 0.3972048163, 0.0740565360, 0.8445752263, 0.5659885406, 0.1107598469, 0.1261267066, 0.3106530905, 0.9623307586, 0.0014953646, 0.0421718284, 0.9182401299, 0.6180395484, 0.7947646379, 0.4402076006, 0.7980208993, 0.6131495237, 0.8885827065, 0.9406354427, 0.4568731785, 0.8838264346, 0.7086120248, 0.2050074339, 0.8598041534, 0.6360205412, 0.6444933414, 0.1086360887, 0.2146544755, 0.4044065177, 0.8566969037, 0.0974318087, 0.9650754929, 0.7885782719, 0.5817304850, 0.0668027699, 0.2600722611, 0.9546993971, 0.2609280050, 0.2063084394, 0.2960519791, 0.8144530654, 0.5386683941, 0.2757037580, 0.3237824142, 0.3469774723, 0.5878881812, 0.8034821153, 0.7495883107, 0.8035441637, 0.6059562564, 0.2713213861, 0.4108335674, 0.5539482832, 0.5046381950, 0.8435614705, 0.3766961098, 0.7583506107, 0.6175935268, 0.3487794399, 0.0058784639, 0.2900554240, 0.9057408571, 0.1079123169, 0.3200630546, 0.7326458693, 0.0237412248, 0.2757625282, 0.8461791873, 0.6101186872, 0.3705151379, 0.6318973899, 0.4013423026, 0.0222425349, 0.0391604938, 0.6966052055, 0.3186582327, 0.3277960122, 0.3301376998, 0.0874366611, 0.3782529831, 0.1412206143, 0.2574128807, 0.3423563242, 0.7656893730, 0.2097123116, 0.8109381199, 0.4845644534, 0.1744513661, 0.3877931535, 0.5369505286, 0.0147142150, 0.2457712293, 0.4901090264, 0.6373463869, 0.2244705260, 0.6722853184, 0.2888159454, 0.5694347620, 0.3042352200, 0.3482132256, 0.5619021654, 0.6760555506, 0.2648956776, 0.9160912037, 0.8973199129, 0.8901007175, 0.8260267973, 0.2438062280, 0.8338996172, 0.7751584649, 0.1436893344, 0.3578631580, 0.8111414909, 0.9454294443, 0.6478928924, 0.0714371502, 0.0711339787, 0.6473786235, 0.0266824700, 0.2442116290, 0.5528301001, 0.2558279037, 0.3684701622, 0.6729193330, 0.8132147193, 0.5830360651, 0.8655517101, 0.0593610443, 0.9748560190, 0.0221947283, 0.6729801893, 0.5001031756, 0.5116565824, 0.2824120522, 0.4552524984, 0.1693765223, 0.1908069402, 0.7663541436, 0.5339511037, 0.0649234429, 0.6125215292, 0.6771115661, 0.6019635797, 0.6840563416, 0.9653987288, 0.1369341463, 0.8428027630, 0.5227881670, 0.5990189910, 0.0936695337, 0.3645765185, 0.9354769588, 0.6745044589, 0.2816980183, 0.3783183694, 0.7331027389, 0.4139548242, 0.1671119779, 0.6703656316, 0.8604171872, 0.6643752456, 0.7547178268, 0.1386961490, 0.4443438351, 0.3267543018, 0.3348949254, 0.9952459931, 0.4534417391, 0.2089741081 };
const static std::vector<float> while_dynamic_output0{ 0.0388205424, 0.0426156297, 0.0980401114, 0.0568757951, 0.1230962500, 0.0412184112, 0.0595490113, 0.4391007423, 0.0377574340, 0.0629260018 };
// clang-format on

TEST_F(TestWhileDynamicModelLoaded, run_verify)
{
  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(_session, "cpu"));
  NNFW_ENSURE_SUCCESS(nnfw_prepare(_session));

  std::vector<float> actual_output0(10);

  nnfw_tensorinfo ti = {NNFW_TYPE_TENSOR_FLOAT32, 3, {1, 28, 28}};
  NNFW_ENSURE_SUCCESS(nnfw_set_input_tensorinfo(_session, 0, &ti));

  setInputOutput(_session, while_dynamic_input0, actual_output0);

  NNFW_ENSURE_SUCCESS(nnfw_run(_session));

  // output check
  verifyOutput(_session, {NNFW_TYPE_TENSOR_FLOAT32, 2, {1, 10}}, while_dynamic_output0,
               actual_output0);
}

TEST_F(TestWhileDynamicModelLoaded, neg_run_verify)
{
  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(_session, "cpu"));
  NNFW_ENSURE_SUCCESS(nnfw_prepare(_session));

  nnfw_tensorinfo ti = {NNFW_TYPE_TENSOR_FLOAT32, 3, {1, 28, 28}};
  NNFW_ENSURE_SUCCESS(nnfw_set_input_tensorinfo(_session, 0, &ti));

  // Insufficient size of output (10 or more is sufficient)
  std::vector<float> actual_output0(9);

  setInputOutput(_session, while_dynamic_input0, actual_output0);

  ASSERT_EQ(nnfw_run(_session), NNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE);
}

using TestIfDynamicModelLoaded = ValidationTestModelLoaded<NNPackages::IF_DYNAMIC>;

// clang-format off
const static std::vector<float> if_dynamic_input0{ 0.7106545568, 0.2156167328, 0.0837147385, 0.0381200500, 0.8007305861, 0.2976274490, 0.8563324213, 0.7781477571, 0.5745304823, 0.8303883672, 0.0862579569, 0.0544887781, 0.1988027841, 0.2230974138, 0.4716774523, 0.4606758952, 0.4920695722, 0.1058474109, 0.0391142406, 0.9550740719, 0.9775217772, 0.1644495875, 0.6734005809, 0.2771040201, 0.4015675485, 0.9389892220, 0.5739571452, 0.6168602109, 0.4262073934, 0.1955287308, 0.6361171603, 0.3251913190, 0.9311535358, 0.9403554797, 0.2734249830, 0.8866292834, 0.5992837548, 0.2142961770, 0.7889495492, 0.0741494149, 0.1030917764, 0.4724597037, 0.3861218989, 0.8098146915, 0.2832616270, 0.6557519436, 0.5689851642, 0.8294774294, 0.4495503902, 0.5395354629, 0.7472639680, 0.4290334582, 0.6575341225, 0.3844197690, 0.5194811821, 0.9411858320, 0.8186575174, 0.6588338614, 0.5179415941, 0.7074140310, 0.1678132862, 0.7229011655, 0.3164389431, 0.6544682384, 0.7210181952, 0.0454275832, 0.6507202387, 0.4012205899, 0.2719061375, 0.2579342127, 0.1064170823, 0.5994709730, 0.1010676920, 0.3968397975, 0.5670611858, 0.1786351353, 0.9127767086, 0.9268618822, 0.6603804827, 0.3673154712, 0.3415949941, 0.5930755138, 0.3685272932, 0.6884198189, 0.1833280921, 0.3941298127, 0.0632725284, 0.1516269594, 0.2316887528, 0.8105147481, 0.1674028039, 0.2784884572, 0.5205677748, 0.4399658442, 0.6527903080, 0.6785870790, 0.2533956766, 0.0617546029, 0.5094803572, 0.5204600096, 0.0249194298, 0.0450648703, 0.1241398007, 0.3705165386, 0.9986394048, 0.6402000785, 0.4894598126, 0.8702902794, 0.4500190616, 0.8115220070, 0.8781826496, 0.6121248603, 0.9077111483, 0.4646541476, 0.7442384362, 0.5584337115, 0.0265889056, 0.9247944951, 0.5661407709, 0.9730864167, 0.6722183824, 0.9564477801, 0.6998952627, 0.6105464697, 0.8297851086, 0.7167860270, 0.6002981067, 0.4256598651, 0.1964918524, 0.9581518769, 0.3121621907, 0.8813912272, 0.3803862929, 0.8825226426, 0.9783715010, 0.1397246420, 0.6996101737, 0.1947445422, 0.9981691837, 0.9528205395, 0.1440794915, 0.2994889319, 0.9605104923, 0.7394120097, 0.8036665916, 0.1226263046, 0.5607838035, 0.5100311637, 0.9977583289, 0.1812620014, 0.8162402511, 0.6829946637, 0.8054547906, 0.5318715572, 0.2573204339, 0.6401459575, 0.9395645857, 0.0523465686, 0.1189657971, 0.4010948837, 0.5229173303, 0.3700955212, 0.8600971103, 0.2058345824, 0.0952973440, 0.6578513980, 0.8096982241, 0.3292799890, 0.3189097345, 0.2228140533, 0.7665079832, 0.3701375425, 0.7601019740, 0.8501300216, 0.5380855203, 0.7509619594, 0.8447382450, 0.6025870442, 0.6957519054, 0.6805172563, 0.5877657533, 0.3472520709, 0.0291769207, 0.0723123997, 0.4284786880, 0.5264689922, 0.4927068353, 0.7379829884, 0.9378200173, 0.8644418716, 0.8671935797, 0.9434295297, 0.5507473350, 0.0760083497, 0.1079615131, 0.1603826135, 0.2987570167, 0.4970068038, 0.0533443913, 0.7932291627, 0.4054899216, 0.8708239794, 0.8852948546, 0.7709504366, 0.2500700951, 0.7328734398, 0.1770015359, 0.4787373245, 0.6746702790, 0.6232759953, 0.8252257109, 0.5074343681, 0.4582579136, 0.7136889100, 0.1850759387, 0.0999758169, 0.9016878009, 0.0968299136, 0.9786298275, 0.7106815577, 0.5932894945, 0.5901473165, 0.8644450903, 0.8777941465, 0.3545308709, 0.5543619394, 0.4764245450, 0.4866352081, 0.7842248678, 0.8535351157, 0.8261910677, 0.4928103089, 0.4883008599, 0.9132300615, 0.0520589016, 0.0571883246, 0.8107213974, 0.2263001502, 0.4195134640, 0.1585850269, 0.6892622709, 0.9932649732, 0.9146085382, 0.3438154757, 0.3597939610, 0.8383805156, 0.1434784085, 0.1592836231, 0.3735914230, 0.5118701458, 0.6597173810, 0.5932899714, 0.7643446326, 0.7639417052, 0.7257087231, 0.8367394209, 0.7241969705, 0.2863937616, 0.7383541465, 0.3918549418, 0.8693540096, 0.8002281189, 0.0121407788, 0.3702836633, 0.3193098009, 0.2857846618, 0.3450623155, 0.8419249654, 0.4484305680, 0.0768098459, 0.1011011526, 0.9832069874, 0.2806532979, 0.6486470103, 0.0038275064, 0.5200383663, 0.5825559497, 0.8526763320, 0.2604954541, 0.4765493274, 0.8257845044, 0.9679267406, 0.3583108485, 0.5755933523, 0.6114814878, 0.5805739164, 0.1076851040, 0.0532303862, 0.3102329671, 0.2268214077, 0.3422079682, 0.3890814781, 0.2123251557, 0.6259000301, 0.9530308843, 0.2377676368, 0.4969599247, 0.3911451399, 0.6869695187, 0.4768487513, 0.0319234431, 0.5153809190, 0.7592291832, 0.5699093938, 0.6517769098, 0.1294958293, 0.5191193819, 0.9886645675, 0.2082915604, 0.9330775738, 0.1966033280, 0.7179551721, 0.4047450423, 0.3280299902, 0.7132403255, 0.7453812361, 0.1643252373, 0.0279585645, 0.0323586352, 0.0771650672, 0.8751529455, 0.3228718042, 0.0091584828, 0.2462333292, 0.2639203966, 0.1246995181, 0.7825807929, 0.0825880542, 0.5019466281, 0.5546332598, 0.2470002472, 0.3974646032, 0.3941309452, 0.2988025546, 0.5270965099, 0.0565799475, 0.7965186834, 0.8401004672, 0.8962592483, 0.2836867571, 0.9854408503, 0.1736569554, 0.3543607295, 0.1489263922, 0.0296417754, 0.8644942045, 0.5768237114, 0.5055403709, 0.7033663988, 0.7610059381, 0.7680964470, 0.9276048541, 0.4661210179, 0.1926902831, 0.8331482410, 0.3478438258, 0.4423305690, 0.1226840168, 0.2631755769, 0.7300418615, 0.8501742482, 0.7732837200, 0.1645421237, 0.9328539968, 0.3299001455, 0.1737864316, 0.6760513186, 0.6878529191, 0.8000500202, 0.7643007040, 0.8427000046, 0.7743517756, 0.4847290516, 0.5107879639, 0.1321444362, 0.2521093190, 0.6971111894, 0.9226302505, 0.7618960738, 0.0798677281, 0.9345219731, 0.3526974618, 0.5779649615, 0.6659775376, 0.0080328183, 0.6179481745, 0.3388322592, 0.8871348500, 0.3849443495, 0.5805974007, 0.4485530853, 0.0118454825, 0.1535516083, 0.9892683029, 0.6305456758, 0.8417525887, 0.9201779366, 0.5443179011, 0.3694557250, 0.9480580688, 0.0420885272, 0.3705308735, 0.1857404709, 0.2711791396, 0.3184533417, 0.2894020677, 0.8524381518, 0.1369639933, 0.5524237156, 0.2515565455, 0.2611325383, 0.7106022239, 0.7720850706, 0.5917789340, 0.1294544786, 0.1406515092, 0.4081685841, 0.7773256898, 0.0337970816, 0.2720888555, 0.6040735841, 0.4713420272, 0.2154571265, 0.7050493360, 0.5699684024, 0.8653516769, 0.2943878472, 0.0710595697, 0.7601916790, 0.8260607719, 0.5490139127, 0.2270360142, 0.6353984475, 0.0237506367, 0.1613635123, 0.2657604814, 0.9112974405, 0.3940451145, 0.9857107997, 0.6584201455, 0.2996906042, 0.6385321617, 0.3025711179, 0.5442391634, 0.5316760540, 0.9278558493, 0.2960957289, 0.2758596539, 0.8092618585, 0.7210826278, 0.5532572269, 0.0433825813, 0.4293606579, 0.9231137037, 0.7861453891, 0.0529759154, 0.2881730795, 0.4177611172, 0.0751738325, 0.2110737860, 0.0087767169, 0.9394732714, 0.7669738531, 0.1285874546, 0.0892729312, 0.7701640129, 0.3619799912, 0.1591310948, 0.5716432333, 0.3634774089, 0.5689123273, 0.1703432053, 0.7500917912, 0.8368289471, 0.6899937391, 0.8733949065, 0.3469920754, 0.9645365477, 0.9452517629, 0.0622390397, 0.0313139819, 0.9253467917, 0.5542111993, 0.4027656317, 0.5191525817, 0.3981988430, 0.7461462021, 0.6761778593, 0.2998072505, 0.8195981979, 0.6851982474, 0.0545753241, 0.1639913172, 0.8172791600, 0.7425212264, 0.1970316321, 0.1586989313, 0.3941454589, 0.8775137067, 0.3532845676, 0.1445332468, 0.4015854299, 0.7155395746, 0.4261780679, 0.7957311273, 0.8265135884, 0.5879834294, 0.7252638340, 0.3942884803, 0.7504889965, 0.5733796358, 0.7747340798, 0.9431585670, 0.5627400875, 0.3371616900, 0.6190663576, 0.5733695626, 0.2214016914, 0.8767938614, 0.2509712279, 0.6909803748, 0.3777657151, 0.6170743704, 0.7373610735, 0.0204360615, 0.7325904369, 0.4920690358, 0.5081653595, 0.9917234182, 0.2093250901, 0.8361138105, 0.7211740017, 0.2606147230, 0.3064637780, 0.1124278903, 0.6320124269, 0.2425052077, 0.4785803258, 0.4747911394, 0.8021139503, 0.3956191838, 0.7217889428, 0.7445480227, 0.1360257119, 0.3709513843, 0.5552678704, 0.2192365974, 0.9431814551, 0.8592399359, 0.7907270789, 0.5545215607, 0.6895139813, 0.1169689223, 0.2043674886, 0.0381150991, 0.7708708644, 0.4759636819, 0.9230924845, 0.6857032776, 0.4432366490, 0.3041133285, 0.7970084548, 0.5629503727, 0.2329168320, 0.2320910394, 0.8098289967, 0.8152811527, 0.9269255996, 0.2628753185, 0.7178934216, 0.1607068628, 0.6057552695, 0.5256694555, 0.5559988022, 0.8001552820, 0.5592993498, 0.5585735440, 0.7596833110, 0.4926379025, 0.8108907342, 0.5142205954, 0.8292154074, 0.9844856262, 0.9281103611, 0.8271671534, 0.8411998153, 0.4101325572, 0.9839829803, 0.1782312542, 0.5126013756, 0.4867194891, 0.9041156173, 0.8752650619, 0.9434064627, 0.5353408456, 0.3405859768, 0.9340458512, 0.1240679324, 0.5371315479, 0.3755141199, 0.2990591526, 0.0670647249, 0.0626592115, 0.7673836946, 0.2539713681, 0.4617587030, 0.9303754568, 0.4884444177, 0.9808034897, 0.7934950590, 0.9362392426, 0.8001930714, 0.8370914459, 0.4767935276, 0.8847136497, 0.8713309765, 0.8301703334, 0.9254899621, 0.5875709057, 0.4544037282, 0.2598260045, 0.7427998781, 0.7183818817, 0.9003841877, 0.0916625410, 0.2609814405, 0.6743535399, 0.7733583450, 0.7338136435, 0.7596724033, 0.7973198891, 0.0015392932, 0.2874146104, 0.1189730167, 0.4800435603, 0.7962353230, 0.4249678552, 0.7483268380, 0.0146148857, 0.6297842860, 0.3471757770, 0.9144366980, 0.8106345534, 0.1789025515, 0.7346886992, 0.1539165080, 0.4280290008, 0.2338476181, 0.3317435384, 0.9998268485, 0.3580373228, 0.9422348738, 0.1251947135, 0.5737128258, 0.6803853512, 0.0485891216, 0.8118965626, 0.7890921235, 0.7665926218, 0.8405004144, 0.3489693701, 0.1429360062, 0.1063490957, 0.5086215734, 0.1312662065, 0.0978318676, 0.4471830130, 0.0830681920, 0.0757851526, 0.1809245348, 0.9280508757, 0.4107315242, 0.5944178104, 0.5625417829, 0.2328256220, 0.9285324812, 0.9903659821, 0.9403946996, 0.5126894712, 0.0232842807, 0.3405880928, 0.6531285644, 0.8213183880, 0.7210904360, 0.4180826247, 0.7917050719, 0.7738851309, 0.1693093032, 0.4396123290, 0.7139748335, 0.8910710216, 0.5668603778, 0.4374921620, 0.8098046780, 0.4076835811, 0.1027061120, 0.5390046835, 0.0044658147, 0.8642644286, 0.8590582609, 0.2715446949, 0.8128718734, 0.7381446362, 0.3621498942, 0.5211849809, 0.6139976382, 0.8567240834, 0.1329502016, 0.2441152930, 0.4219030440, 0.1751736850, 0.6326612234, 0.3929811120, 0.0947103724, 0.1078760102, 0.8769059777, 0.1599343121, 0.6111860275, 0.0368208028, 0.0899466202, 0.9127882719, 0.1146656275, 0.4647151828, 0.3303563893, 0.5797663927, 0.8400436044, 0.2845958769, 0.2181742340, 0.9651557207, 0.1241061762, 0.0102593508, 0.6999664903, 0.8487475514, 0.6001151800, 0.9682601690, 0.6127328873, 0.1502806544, 0.2512893379, 0.3930048048, 0.3448313475, 0.5263126493, 0.7319667935, 0.9264212251, 0.4489789009, 0.0418849625, 0.5219999552, 0.3397078812, 0.4435234964, 0.4758536220, 0.1290920675, 0.1649249196, 0.1736114621, 0.5685442686, 0.3253444433, 0.0540574715, 0.2022368759, 0.0260062832, 0.9889448285, 0.2064949423, 0.3756456375, 0.8462600112, 0.8166462779, 0.1788506061, 0.6607533097, 0.1638182998, 0.7888727188, 0.3304887116, 0.3085075021, 0.6626392603, 0.2860932350, 0.1577534527, 0.0126363616, 0.1958409399, 0.2475458980, 0.1514713019, 0.5241229534, 0.9845717549, 0.8002693653, 0.3091083765, 0.3348104060, 0.1341333240, 0.3546191454, 0.3800157905, 0.0364337005 };
const static std::vector<float> if_dynamic_output0{ 0.0444660522, 0.0271649156, 0.0191113371, 0.0014375688, 0.0690929219, 0.0001767588, 0.0030322229, 0.0118752792, 0.0419745520, 0.7816683054 };
// clang-format on

TEST_F(TestIfDynamicModelLoaded, run_verify)
{
  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(_session, "cpu"));
  NNFW_ENSURE_SUCCESS(nnfw_prepare(_session));

  nnfw_tensorinfo ti_output0_expected = {NNFW_TYPE_TENSOR_FLOAT32, 2, {1, 10}};

  // Output tensor sizes are inferenced after `nnfw_prepare`
  {
    nnfw_tensorinfo ti;
    NNFW_ENSURE_SUCCESS(nnfw_output_tensorinfo(_session, 0, &ti));
    ASSERT_TRUE(tensorInfoEqual(ti, ti_output0_expected));
  }

  std::vector<float> actual_output0(10);
  setInputOutput(_session, if_dynamic_input0, actual_output0);

  NNFW_ENSURE_SUCCESS(nnfw_run(_session));

  // Check output tensor sizes again
  {
    nnfw_tensorinfo ti;
    NNFW_ENSURE_SUCCESS(nnfw_output_tensorinfo(_session, 0, &ti));
    ASSERT_TRUE(tensorInfoEqual(ti, ti_output0_expected));
  }

  // Output value check
  for (int i = 0; i < actual_output0.size(); ++i)
    ASSERT_NEAR(if_dynamic_output0[i], actual_output0[i], 0.00001);
}

class CombinationTest1 : public ::testing::Test
{
protected:
  void SetUp() override
  {
    CircleGen cgen;

    // Creating a graph which has dynamic tensors after compilation.
    // #3 and #4 are dynamic. This model was used to check if internal dynamic tensors could
    // make any side-effect.
    //
    // #0 = input 0 of shape [1]
    // #1 = input 1 of shape [2]
    // #2 = cast(#0, int to float)
    // #3 = reshape(const of shape [4] , #1)
    // #4 = add(#2, #3)

    constexpr circle::TensorType CIRCLE_DTYPE = circle::TensorType::TensorType_FLOAT32;

    int cast_in = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32});
    int cast_out = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});

    cgen.addOperatorCast({{cast_in}, {cast_out}}, circle::TensorType::TensorType_INT32,
                         circle::TensorType::TensorType_FLOAT32);

    std::vector<float> reshape_in_data{0, 1, 2, 3}; // defining constant tensor
    uint32_t reshape_in_buf = cgen.addBuffer(reshape_in_data);
    int reshape_in = cgen.addTensor({{4}, CIRCLE_DTYPE, reshape_in_buf});
    int reshape_shape_in = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32});
    int reshape_out = cgen.addTensor({{}, CIRCLE_DTYPE}); // dynamic tensor of shape {}

    cgen.addOperatorReshape({{reshape_in, reshape_shape_in}, {reshape_out}});

    int out = cgen.addTensor({{}, CIRCLE_DTYPE}); // dynamic tensor of shape {}
    cgen.addOperatorAdd({{cast_out, reshape_out}, {out}}, circle::ActivationFunctionType_NONE);
    cgen.setInputsAndOutputs({cast_in, reshape_shape_in}, {out});

    _circle_buffer = cgen.finish();
  }

  void TearDown() override
  { // DO NOTHING
  }

  void setSession(nnfw_session *session) { _session = session; }

  CircleBuffer &getCircleBuffer() { return _circle_buffer; }

  void run_WITHOUT_set_input_tensorinfo(const std::vector<int32_t> &cast_input,
                                        const std::vector<int32_t> &reshape_shape_input,
                                        const nnfw_tensorinfo &expected_ti,
                                        const std::vector<float> &expected,
                                        std::vector<float> &actual)
  {
    setInputOutput(_session, cast_input, reshape_shape_input, actual);
    NNFW_ENSURE_SUCCESS(nnfw_run(_session));
    verifyOutput(_session, expected_ti, expected, actual);
  }

  void run_WITH_set_input_tensorinfo(int32_t new_dim_0, const std::vector<int32_t> &cast_input,
                                     const std::vector<int32_t> &reshape_shape_input,
                                     const nnfw_tensorinfo &expected_ti,
                                     const std::vector<float> &expected, std::vector<float> &actual)
  {
    nnfw_tensorinfo t_in;
    t_in.dtype = NNFW_TYPE_TENSOR_INT32;
    t_in.rank = 1;
    t_in.dims[0] = new_dim_0;
    NNFW_ENSURE_SUCCESS(nnfw_set_input_tensorinfo(_session, 0, &t_in));

    setInputOutput(_session, cast_input, reshape_shape_input, actual);
    NNFW_ENSURE_SUCCESS(nnfw_run(_session));
    verifyOutput(_session, expected_ti, expected, actual);
  }

private:
  nnfw_session *_session;
  CircleBuffer _circle_buffer;
};

// test for https://github.com/Samsung/ONE/issues/4625
TEST_F(CombinationTest1, combination_of_set_input_tensorinfo_and_nnfw_run)
{
  constexpr NNFW_TYPE NNFW_DTYPE = NNFW_TYPE_TENSOR_FLOAT32;
  std::vector<int32_t> cast_in_buf;
  std::vector<int32_t> reshape_shape_in_buf;
  std::vector<float> actual(4), expected(4);

  nnfw_session *session = nullptr;
  auto &cbuf = getCircleBuffer();

  auto create_prepare_session = [&](const CircleBuffer &cbuf) {
    NNFW_ENSURE_SUCCESS(nnfw_create_session(&session));
    NNFW_ENSURE_SUCCESS(nnfw_load_circle_from_buffer(session, cbuf.buffer(), cbuf.size()));
    NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(session, "cpu"));
    NNFW_ENSURE_SUCCESS(nnfw_prepare(session));
  };

  // combinations of executions of static and dynamic tensors
  // 1) no change on the shape of #0 -> change #0 to shape [1] -> no change. use previous shape
  // 2) no change on the shape of #0 -> change #0 to shape [2] -> no change. use previous shape
  // 3) no change on the shape of #0 -> change #0 to shape [2] -> change #0 to shape [1]

  // 1) no change on the shape of #0 -> change #0 to shape [1] -> no input change
  //       static                             dynamic                      dynamic
  create_prepare_session(cbuf);
  {
    setSession(session);

    // no change on the shape of #0
    cast_in_buf = {10};
    reshape_shape_in_buf = {1, 4};
    expected = {10, 11, 12, 13};
    run_WITHOUT_set_input_tensorinfo(cast_in_buf, reshape_shape_in_buf,
                                     {NNFW_TYPE_TENSOR_FLOAT32, 2, {1, 4}}, expected, actual);

    // change to the default shape [1] of #0, this treats 0# dynamic
    int32_t new_dim_0 = 1;
    cast_in_buf = {10};
    reshape_shape_in_buf = {1, 4};
    expected = {10, 11, 12, 13};
    run_WITH_set_input_tensorinfo(new_dim_0, cast_in_buf, reshape_shape_in_buf,
                                  {NNFW_TYPE_TENSOR_FLOAT32, 2, {1, 4}}, expected, actual);

    // no change. Use previous shape
    run_WITHOUT_set_input_tensorinfo(cast_in_buf, reshape_shape_in_buf,
                                     {NNFW_TYPE_TENSOR_FLOAT32, 2, {1, 4}}, expected, actual);

    NNFW_ENSURE_SUCCESS(nnfw_close_session(session));
  }

  // 2) no change on the shape of #0 -> change #0 to shape [2] -> no change(use previous shape)
  //       static                             dynamic                      dynamic
  create_prepare_session(cbuf);
  {
    setSession(session);

    // no change on the shape of #0
    cast_in_buf = {10};
    reshape_shape_in_buf = {1, 4};
    expected = {10, 11, 12, 13};
    run_WITHOUT_set_input_tensorinfo(cast_in_buf, reshape_shape_in_buf,
                                     {NNFW_TYPE_TENSOR_FLOAT32, 2, {1, 4}}, expected, actual);

    // change shape of #0 to [2], this treats 0# dynamic
    int32_t new_dim_0 = 2;
    cast_in_buf = {10, 20};
    reshape_shape_in_buf = {2, 2};
    expected = {10, 21, 12, 23};
    run_WITH_set_input_tensorinfo(new_dim_0, cast_in_buf, reshape_shape_in_buf,
                                  {NNFW_TYPE_TENSOR_FLOAT32, 2, {2, 2}}, expected, actual);

    // no change. Use previous shape
    run_WITH_set_input_tensorinfo(new_dim_0, cast_in_buf, reshape_shape_in_buf,
                                  {NNFW_TYPE_TENSOR_FLOAT32, 2, {2, 2}}, expected, actual);

    NNFW_ENSURE_SUCCESS(nnfw_close_session(session));
  }

  // 3) no change on the shape of #0 -> change #0 to shape [2] -> change #0 to shape [1]
  //       static                             dynamic                      dynamic
  create_prepare_session(cbuf);
  {
    setSession(session);

    // no change on the shape of #0
    cast_in_buf = {10};
    reshape_shape_in_buf = {1, 4};
    expected = {10, 11, 12, 13};
    run_WITHOUT_set_input_tensorinfo(cast_in_buf, reshape_shape_in_buf,
                                     {NNFW_TYPE_TENSOR_FLOAT32, 2, {1, 4}}, expected, actual);

    // change shape of #0 to [2], this treats 0# dynamic
    int32_t new_dim_0 = 2;
    cast_in_buf = {10, 20};
    reshape_shape_in_buf = {2, 2};
    expected = {10, 21, 12, 23};
    run_WITH_set_input_tensorinfo(new_dim_0, cast_in_buf, reshape_shape_in_buf,
                                  {NNFW_TYPE_TENSOR_FLOAT32, 2, {2, 2}}, expected, actual);

    // change #0 to shape [1]
    new_dim_0 = 1;
    cast_in_buf = {100};
    reshape_shape_in_buf = {1, 4};
    expected = {100, 101, 102, 103};
    run_WITH_set_input_tensorinfo(new_dim_0, cast_in_buf, reshape_shape_in_buf,
                                  {NNFW_TYPE_TENSOR_FLOAT32, 2, {1, 4}}, expected, actual);

    NNFW_ENSURE_SUCCESS(nnfw_close_session(session));
  }
}

TEST_F(CombinationTest1, neg_combination_of_set_input_tensorinfo_and_nnfw_run)
{
  nnfw_session *session = nullptr;
  auto &cbuf = getCircleBuffer();
  NNFW_ENSURE_SUCCESS(nnfw_create_session(&session));
  NNFW_ENSURE_SUCCESS(nnfw_load_circle_from_buffer(session, cbuf.buffer(), cbuf.size()));
  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(session, "cpu"));
  NNFW_ENSURE_SUCCESS(nnfw_prepare(session));

  setSession(session);

  std::vector<int32_t> cast_in_buf;
  std::vector<int32_t> reshape_shape_in_buf;
  std::vector<float> actual(4), expected(4);

  // no change on the shape of #0
  cast_in_buf = {10};
  reshape_shape_in_buf = {1, 4};
  expected = {10, 11, 12, 13};
  setInputOutput(session, cast_in_buf, reshape_shape_in_buf, actual);
  NNFW_ENSURE_SUCCESS(nnfw_run(session));

  // change the shape of #0 to [4]
  cast_in_buf = {10, 20, 30, 40};
  reshape_shape_in_buf = {1, 4};
  expected = {10, 21, 32, 43};
  run_WITH_set_input_tensorinfo(4, cast_in_buf, reshape_shape_in_buf,
                                {NNFW_TYPE_TENSOR_FLOAT32, 2, {1, 4}}, expected, actual);
  setInputOutput(session, cast_in_buf, reshape_shape_in_buf, actual);
  NNFW_ENSURE_SUCCESS(nnfw_run(session));

  // run without changing #0 but caller thought that it is now shape [1]
  cast_in_buf = {10};
  reshape_shape_in_buf = {1, 4};
  expected = {10, 11, 12, 13};
  // This should throw an error
  EXPECT_ANY_THROW(setInputOutput(session, cast_in_buf, reshape_shape_in_buf, actual));

  NNFW_ENSURE_SUCCESS(nnfw_close_session(session));
}

// Class to test set_input_tensorinfo() against "two" inputs
class CombinationTest2 : public ::testing::Test
{
protected:
  void SetUp() override
  {
    CircleGen cgen;

    // creating a graph with two inputs
    //
    // #0 = input 0 of shape [1]
    // #1 = input 1 of shape [1]
    // #2 = add(#0, #1)

    constexpr circle::TensorType CIRCLE_DTYPE = circle::TensorType::TensorType_FLOAT32;

    int in0 = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32});
    int in1 = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32});
    int out = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32});

    cgen.addOperatorAdd({{in0, in1}, {out}}, circle::ActivationFunctionType_NONE);
    cgen.setInputsAndOutputs({in0, in1}, {out});

    _circle_buffer = cgen.finish();
  }

  void TearDown() override
  { // DO NOTHING
  }

  void setSession(nnfw_session *session) { _session = session; }

  CircleBuffer &getCircleBuffer() { return _circle_buffer; }

  void run_WITHOUT_set_input_tensorinfo(const std::vector<int32_t> &in0,
                                        const std::vector<int32_t> &in1,
                                        const nnfw_tensorinfo &expected_ti,
                                        const std::vector<int32_t> &expected,
                                        std::vector<int32_t> &actual)
  {
    setInputOutput(_session, in0, in1, actual);
    NNFW_ENSURE_SUCCESS(nnfw_run(_session));
    verifyOutput(_session, expected_ti, expected, actual);
  }

  // Pass -1 for t0_new_dim_0 (or t1_new_dim_0)
  // if shape of tensor 0 (or tensor 1) does not changed from the shape in a model
  void run_WITH_set_input_tensorinfo(int32_t t0_new_dim_0, int32_t t1_new_dim_0,
                                     const std::vector<int32_t> &in0,
                                     const std::vector<int32_t> &in1,
                                     const nnfw_tensorinfo &expected_ti,
                                     const std::vector<int32_t> &expected,
                                     std::vector<int32_t> &actual)
  {
    if (t0_new_dim_0 >= 0)
    {
      nnfw_tensorinfo t_in;
      t_in.dtype = NNFW_TYPE_TENSOR_INT32;
      t_in.rank = 1;
      t_in.dims[0] = t0_new_dim_0;
      NNFW_ENSURE_SUCCESS(nnfw_set_input_tensorinfo(_session, 0, &t_in));
    }

    if (t1_new_dim_0 >= 0)
    {
      nnfw_tensorinfo t_in;
      t_in.dtype = NNFW_TYPE_TENSOR_INT32;
      t_in.rank = 1;
      t_in.dims[0] = t1_new_dim_0;
      NNFW_ENSURE_SUCCESS(nnfw_set_input_tensorinfo(_session, 1, &t_in));
    }

    setInputOutput(_session, in0, in1, actual);
    NNFW_ENSURE_SUCCESS(nnfw_run(_session));
    verifyOutput(_session, expected_ti, expected, actual);
  }

private:
  nnfw_session *_session;
  CircleBuffer _circle_buffer;
};

// test for https://github.com/Samsung/ONE/issues/4625
TEST_F(CombinationTest2, combination_set_input_tensorinfo_for_two_inputs)
{
  nnfw_session *session = nullptr;

  // combinations of executions of static and dynamic tensors for "two" inputs (#0, #1)
  // 0. both input shapes are [1] (input shapes of the model are [1], [1])
  // 1. change shape of #0 to [2]
  // 2. change shape of #0 to [1], change shape of #1 to [2]
  // 3. change shape of #0 to [2], (shape of #1 is still [2])
  // 4. don't call set_input_tensorinfo (both are still [2] and [2])
  // 5. change shape of #0 to [1], change shape of #1 to [1]
  std::vector<int32_t> in0, in1;
  std::vector<int32_t> actual, expected;
  nnfw_tensorinfo expected_ti;

  auto &cbuf = getCircleBuffer();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&session));
  NNFW_ENSURE_SUCCESS(nnfw_load_circle_from_buffer(session, cbuf.buffer(), cbuf.size()));
  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(session, "cpu"));
  NNFW_ENSURE_SUCCESS(nnfw_prepare(session));
  setSession(session);

  constexpr int32_t NO_CHANGE = -1;

  // 0. both input shapes are [1]
  in0 = {10};
  in1 = {100};
  expected = {110};
  expected_ti = {NNFW_TYPE_TENSOR_INT32, 1, {1}};
  actual.resize(1);
  run_WITHOUT_set_input_tensorinfo(in0, in1, expected_ti, expected, actual);

  // 1. change shape of #0 to [2]
  int32_t new_dim_0 = 2;
  int32_t new_dim_1 = NO_CHANGE;
  in0 = {10, 20};
  in1 = {100};
  expected = {110, 120};
  expected_ti = {NNFW_TYPE_TENSOR_INT32, 1, {2}};
  actual.resize(2);
  run_WITH_set_input_tensorinfo(new_dim_0, new_dim_1, in0, in1, expected_ti, expected, actual);

  // 2. change shape of #0 to [1], change shape of #1 to [2]
  new_dim_0 = 1;
  new_dim_1 = 2;
  in0 = {1000};
  in1 = {10, 20};
  expected = {1010, 1020};
  expected_ti = {NNFW_TYPE_TENSOR_INT32, 1, {2}};
  actual.resize(2);
  run_WITH_set_input_tensorinfo(new_dim_0, new_dim_1, in0, in1, expected_ti, expected, actual);

  // // 3. change shape of #0 to [2], (shape of #1 is still [2])
  new_dim_0 = 2;
  new_dim_1 = NO_CHANGE;
  in0 = {10, 20};
  in1 = {100, 200};
  expected = {110, 220};
  expected_ti = {NNFW_TYPE_TENSOR_INT32, 1, {2}};
  actual.resize(2);
  run_WITH_set_input_tensorinfo(new_dim_0, new_dim_1, in0, in1, expected_ti, expected, actual);

  // // 4. don't call set_input_tensorinfo (both are still [2] and [2])
  in0 = {11, 22};
  in1 = {1000, 2000};
  expected = {1011, 2022};
  expected_ti = {NNFW_TYPE_TENSOR_INT32, 1, {2}};
  actual.resize(2);
  run_WITHOUT_set_input_tensorinfo(in0, in1, expected_ti, expected, actual);

  // // 5. change shape of #0 to [1], change shape of #1 to [1]
  new_dim_0 = 1;
  new_dim_1 = 1;
  in0 = {50};
  in1 = {500};
  expected = {550};
  expected_ti = {NNFW_TYPE_TENSOR_INT32, 1, {1}};
  actual.resize(1);
  run_WITH_set_input_tensorinfo(new_dim_0, new_dim_1, in0, in1, expected_ti, expected, actual);

  NNFW_ENSURE_SUCCESS(nnfw_close_session(session));
}

TEST_F(CombinationTest2, neg_combination_set_input_tensorinfo_for_two_inputs)
{
  nnfw_session *session = nullptr;

  // change shape of #1 to [2]
  // then, do not call nnfw_set_input_tensorinfo for #1
  std::vector<int32_t> in0, in1;
  std::vector<int32_t> expected_shape;
  std::vector<int32_t> actual, expected;
  nnfw_tensorinfo expected_ti;

  auto &cbuf = getCircleBuffer();

  NNFW_ENSURE_SUCCESS(nnfw_create_session(&session));
  NNFW_ENSURE_SUCCESS(nnfw_load_circle_from_buffer(session, cbuf.buffer(), cbuf.size()));
  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(session, "cpu"));
  NNFW_ENSURE_SUCCESS(nnfw_prepare(session));
  setSession(session);

  constexpr int32_t NO_CHANGE = -1;

  // change shape of #1 to [2]
  int32_t new_dim_0 = NO_CHANGE;
  int32_t new_dim_1 = 2;
  in0 = {10};
  in1 = {100, 200};
  expected = {110, 210};
  expected_ti = {NNFW_TYPE_TENSOR_INT32, 1, {2}};
  actual.resize(2);
  run_WITH_set_input_tensorinfo(new_dim_0, new_dim_1, in0, in1, expected_ti, expected, actual);

  // then, do not call nnfw_set_input_tensorinfo for #1, thinking that
  // #1 has now shape [1], which is wrong
  in0 = {10};
  in1 = {100};
  expected = {110};                               // wrong
  expected_ti = {NNFW_TYPE_TENSOR_INT32, 1, {1}}; // wrong
  actual.resize(1);                               // wrong
  EXPECT_ANY_THROW(run_WITHOUT_set_input_tensorinfo(in0, in1, expected_ti, expected, actual));

  NNFW_ENSURE_SUCCESS(nnfw_close_session(session));
}
