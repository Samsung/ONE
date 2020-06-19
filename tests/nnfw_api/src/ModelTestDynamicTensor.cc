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
#include <nnfw_debug.h>

#include "common.h"
#include "fixtures.h"
#include "NNPackages.h"

void set_input_output(nnfw_session *session, const std::vector<float> &input,
                      std::vector<float> *actual_output)
{
  ASSERT_EQ(nnfw_set_input(session, 0, NNFW_TYPE_TENSOR_FLOAT32, input.data(),
                           sizeof(float) * input.size()),
            NNFW_STATUS_NO_ERROR);

  ASSERT_EQ(nnfw_set_output(session, 0, NNFW_TYPE_TENSOR_FLOAT32, actual_output->data(),
                            sizeof(float) * actual_output->size()),
            NNFW_STATUS_NO_ERROR);
}

void set_input_output(nnfw_session *session, const std::vector<float> &input0,
                      const std::vector<float> &input1, std::vector<float> *actual_output)
{
  ASSERT_EQ(nnfw_set_input(session, 0, NNFW_TYPE_TENSOR_FLOAT32, input0.data(),
                           sizeof(float) * input0.size()),
            NNFW_STATUS_NO_ERROR);
  ASSERT_EQ(nnfw_set_input(session, 1, NNFW_TYPE_TENSOR_FLOAT32, input1.data(),
                           sizeof(float) * input1.size()),
            NNFW_STATUS_NO_ERROR);

  ASSERT_EQ(nnfw_set_output(session, 0, NNFW_TYPE_TENSOR_FLOAT32, actual_output->data(),
                            sizeof(float) * actual_output->size()),
            NNFW_STATUS_NO_ERROR);
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
class TestDynamicTensorReshapeModelLoaded
    : public ValidationTestModelLoaded<NNPackages::DYNAMIC_TENSOR_RESHAPE>
{
protected:
  void set_input_output(const std::vector<int> &new_shape, int actual_output_size,
                        std::vector<float> *actual_output)
  {
    NNFW_STATUS res = nnfw_set_input(_session, 0, NNFW_TYPE_TENSOR_INT32, new_shape.data(),
                                     sizeof(int) * new_shape.size());
    ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

    res = nnfw_set_output(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, actual_output->data(),
                          sizeof(float) * actual_output_size);
    ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);
  }

  void prepare_and_set_input_output(const std::vector<int> &new_shape, int actual_output_size,
                                    std::vector<float> *actual_output)
  {
    ASSERT_EQ(nnfw_set_available_backends(_session, "cpu"), NNFW_STATUS_NO_ERROR);

    NNFW_STATUS res = NNFW_STATUS_ERROR;

    res = nnfw_prepare(_session);
    ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

    set_input_output(new_shape, actual_output_size, actual_output);
    // real test case should start from calling nnfw_run()
  }

  // call this after calling nnfw_prepare()
  void set_input_output_and_run(const std::vector<int> &new_shape,
                                const std::vector<float> &expected_output, bool no_run_error = true)
  {
    int output_element_num = expected_output.size();
    std::vector<float> actual_output(output_element_num);

    set_input_output(new_shape, output_element_num, &actual_output);

    // Do inference
    NNFW_STATUS res = nnfw_run(_session);

    if (no_run_error)
    {
      ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

      // output shape check
      nnfw_tensorinfo info;
      ASSERT_EQ(nnfw_output_tensorinfo(_session, 0, &info), NNFW_STATUS_NO_ERROR);
      ASSERT_EQ(info.rank, new_shape.size());
      for (uint32_t d = 0; d < info.rank; ++d)
        ASSERT_EQ(info.dims[d], new_shape[d]);

      // output value check
      for (int i = 0; i < expected_output.size(); ++i)
        ASSERT_EQ(expected_output[i], actual_output[i]);
    }
    else
    {
      ASSERT_EQ(res, NNFW_STATUS_ERROR);
    }
  };

  void TearDown() override
  {
    ValidationTestModelLoaded<NNPackages::DYNAMIC_TENSOR_RESHAPE>::TearDown();
  }
};

TEST_F(TestDynamicTensorReshapeModelLoaded, reshape_to_3x2)
{
  const std::vector<int> new_shape = {3, 2};
  const std::vector<float> expected = {-1.5, -1.0, -0.5, 0.5, 1.0, 1.5};
  std::vector<float> actual_output(expected.size());

  prepare_and_set_input_output(new_shape, expected.size(), &actual_output);

  // Do inference
  NNFW_STATUS res = nnfw_run(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // output value check
  for (int i = 0; i < expected.size(); ++i)
    ASSERT_EQ(expected[i], actual_output[i]);
}

/**
 * @brief Negative test.
 *        Reshape's first input has 6 values but trying to reshaping to [3, 3]
 */
TEST_F(TestDynamicTensorReshapeModelLoaded, neg_reshape_to_wrong_3x3)
{
  const std::vector<int> wrong_shape = {3, 3}; // wrong shape input
  const int actual_element_num = 9;            // whatever number
  std::vector<float> actual_output(9);         // whatever size

  prepare_and_set_input_output(wrong_shape, actual_element_num, &actual_output);

  // Do inference
  NNFW_STATUS res = nnfw_run(_session);
  ASSERT_EQ(res, NNFW_STATUS_ERROR); // run should fail
}

TEST_F(TestDynamicTensorReshapeModelLoaded, reshape_multiple_executions)
{
  ASSERT_EQ(nnfw_set_available_backends(_session, "cpu"), NNFW_STATUS_NO_ERROR);

  NNFW_STATUS res = nnfw_prepare(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  std::vector<int> new_shape;
  std::vector<float> expected = {-1.5, -1.0, -0.5, 0.5, 1.0, 1.5};

  // let's call multiple times
  new_shape = {3, 2};
  set_input_output_and_run(new_shape, expected);

  new_shape = {1, 6};
  set_input_output_and_run(new_shape, expected);

  new_shape = {6, 1};
  set_input_output_and_run(new_shape, expected);
}

TEST_F(TestDynamicTensorReshapeModelLoaded, neg_reshape_multiple_executions)
{
  ASSERT_EQ(nnfw_set_available_backends(_session, "cpu"), NNFW_STATUS_NO_ERROR);

  NNFW_STATUS res = nnfw_prepare(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  std::vector<int> new_shape;
  std::vector<float> expected = {-1.5, -1.0, -0.5, 0.5, 1.0, 1.5};

  // let's call multiple times including the second nnfw_run() to fail
  new_shape = {3, 2};
  set_input_output_and_run(new_shape, expected);

  new_shape = {1, 100};                                 // wrong shape
  set_input_output_and_run(new_shape, expected, false); // Run will fail

  // next run should succeed
  new_shape = {6, 1};
  set_input_output_and_run(new_shape, expected);
}

//
// Unknown Dimension Test
//    Trying to set unknown dim to other value before calling nnfw_prepare()
//

class TestInputUnknownDimInputConcatModelLoaded
    : public ValidationTestModelLoaded<NNPackages::UNKNOWN_DIM_INPUT_CONCAT>
{
protected:
  void prepare_apply_set_input_output(const std::vector<float> &input0,
                                      const std::vector<float> &input1,
                                      std::vector<float> *actual_output, nnfw_tensorinfo input0_ti)
  {
    ASSERT_EQ(nnfw_prepare(_session), NNFW_STATUS_NO_ERROR);
    ASSERT_EQ(nnfw_set_input_tensorinfo(_session, 0, &input0_ti), NNFW_STATUS_NO_ERROR);

    ASSERT_EQ(nnfw_set_input(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, input0.data(),
                             sizeof(float) * input0.size()),
              NNFW_STATUS_NO_ERROR);
    ASSERT_EQ(nnfw_set_input(_session, 1, NNFW_TYPE_TENSOR_FLOAT32, input1.data(),
                             sizeof(float) * input1.size()),
              NNFW_STATUS_NO_ERROR);

    ASSERT_EQ(nnfw_set_output(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, actual_output->data(),
                              sizeof(float) * actual_output->size()),
              NNFW_STATUS_NO_ERROR);
  }
};

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
TEST_F(TestInputUnknownDimInputConcatModelLoaded, concat_input0_to_2x3)
{
  ASSERT_EQ(nnfw_set_available_backends(_session, "cpu"), NNFW_STATUS_NO_ERROR);

  const std::vector<float> input0 = {1, 2, 3};          // of shape [1, 3]
  const std::vector<float> input1 = {4, 5, 6, 7, 8, 9}; // of shape [2, 3]

  const std::vector<float> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<float> actual_output(expected.size());

  // input reshaping to [1, 3]
  nnfw_tensorinfo ti = {NNFW_TYPE_TENSOR_FLOAT32, 2, {1, 3}};
  ASSERT_EQ(nnfw_set_input_tensorinfo(_session, 0, &ti), NNFW_STATUS_NO_ERROR);
  ASSERT_EQ(nnfw_prepare(_session), NNFW_STATUS_NO_ERROR);

  set_input_output(_session, input0, input1, &actual_output);

  // Do inference
  NNFW_STATUS res = nnfw_run(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // output value check
  for (int i = 0; i < expected.size(); ++i)
    ASSERT_EQ(expected[i], actual_output[i]);
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
TEST_F(TestInputUnknownDimInputConcatModelLoaded, neg_concat_input0_to_wrong_shape)
{
  ASSERT_EQ(nnfw_set_available_backends(_session, "cpu"), NNFW_STATUS_NO_ERROR);

  const std::vector<float> input0 = {1, 2, 3};          // of shape [3, 1], wrong shape
  const std::vector<float> input1 = {4, 5, 6, 7, 8, 9}; // of shape [2, 3]

  std::vector<float> actual_output(100); // whatever size

  // input reshaping to [3, 1]
  nnfw_tensorinfo ti = {ti.dtype = NNFW_TYPE_TENSOR_FLOAT32, 2, {3, 1}};
  ASSERT_EQ(nnfw_set_input_tensorinfo(_session, 0, &ti), NNFW_STATUS_NO_ERROR);

  ASSERT_EQ(nnfw_prepare(_session), NNFW_STATUS_ERROR);
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
using TestDynamicTensorApplyTensorInfoBinaryOp =
    ValidationTestModelLoaded<NNPackages::ADD_UNSPECIFIED_RANK_INPUTS>;

TEST_F(TestDynamicTensorApplyTensorInfoBinaryOp, set_input_tensorinfo_after_compilation_add)
{
  ASSERT_EQ(nnfw_set_available_backends(_session, "cpu"), NNFW_STATUS_NO_ERROR);

  // input reshaping to [2, 2, 3]
  nnfw_tensorinfo input0_ti = {NNFW_TYPE_TENSOR_FLOAT32, 3, {2, 2, 3}};

  std::vector<float> input0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> input1 = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  std::vector<float> actual_output(12);
  std::vector<float> expected_output = {1.1 * 2, 2.1 * 2, 3.1 * 2, 4.1 * 2,  5.1 * 2,  6.1 * 2,
                                        7.1 * 2, 8.1 * 2, 9.1 * 2, 10.1 * 2, 11.1 * 2, 12.1 * 2};

  ASSERT_EQ(nnfw_prepare(_session), NNFW_STATUS_NO_ERROR);

  ASSERT_EQ(nnfw_set_input_tensorinfo(_session, 0, &input0_ti), NNFW_STATUS_NO_ERROR);

  set_input_output(_session, input0, input1, &actual_output);

  // Do inference
  NNFW_STATUS res = nnfw_run(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // output value check
  for (int i = 0; i < expected_output.size(); ++i)
    ASSERT_EQ(expected_output[i], actual_output[i]);
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
using TestDynamicTensorApplyTensorInfoUnaryOp = ValidationTestModelLoaded<NNPackages::NEG>;

TEST_F(TestDynamicTensorApplyTensorInfoUnaryOp, set_input_tensorinfo_after_compilation_neg)
{
  ASSERT_EQ(nnfw_set_available_backends(_session, "cpu"), NNFW_STATUS_NO_ERROR);

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

  ASSERT_EQ(nnfw_prepare(_session), NNFW_STATUS_NO_ERROR);

  // input shape check
  {
    nnfw_tensorinfo ti = {};
    ASSERT_EQ(nnfw_input_tensorinfo(_session, 0, &ti), NNFW_STATUS_NO_ERROR);
    ASSERT_TRUE(tensorInfoEqual(input0_ti_original, ti));
  }

  ASSERT_EQ(nnfw_set_input_tensorinfo(_session, 0, &input0_ti), NNFW_STATUS_NO_ERROR);

  // input shape check
  {
    nnfw_tensorinfo ti = {};
    ASSERT_EQ(nnfw_input_tensorinfo(_session, 0, &ti), NNFW_STATUS_NO_ERROR);
    ASSERT_TRUE(tensorInfoEqual(input0_ti, ti));
  }

  set_input_output(_session, input0, &actual_output);

  // Do inference
  NNFW_STATUS res = nnfw_run(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // output value check
  for (int i = 0; i < expected_output.size(); ++i)
    ASSERT_EQ(expected_output[i], actual_output[i]);
}

using TestWhileDynamicModelLoaded = ValidationTestModelLoaded<NNPackages::WHILE_DYNAMIC>;

// clang-format off
const static std::vector<float> while_dynamic_input0{ 0.4325029254, 0.7332934141, 0.2969786823, 0.1540192217, 0.4608841240, 0.1523699313, 0.4334940016, 0.1022945493, 0.6928671598, 0.5891978741, 0.8283287883, 0.7041553259, 0.5243381262, 0.5623597503, 0.3395180404, 0.3212788701, 0.5248492956, 0.2551939189, 0.1338981092, 0.6406514645, 0.7089318633, 0.8164196610, 0.7689018846, 0.3551857173, 0.7668499351, 0.4942102134, 0.7345644236, 0.4689270556, 0.3495515287, 0.0768318549, 0.0868133679, 0.7823525667, 0.0791761801, 0.4397472143, 0.8150953054, 0.5074489713, 0.0895665437, 0.9451501966, 0.1064314246, 0.8803006411, 0.9903403521, 0.1259460151, 0.1889930069, 0.7466737032, 0.0553287826, 0.9712036252, 0.6352610588, 0.6301708817, 0.3079694211, 0.5367568731, 0.4070350230, 0.6815373302, 0.6948529482, 0.6158187985, 0.1485853940, 0.9162485600, 0.3622985184, 0.2672208250, 0.3396688998, 0.4135381579, 0.6450354457, 0.2386536747, 0.7072004080, 0.5289406180, 0.0643024296, 0.1969666779, 0.8667400479, 0.3396836221, 0.5878564715, 0.4551178813, 0.4318033755, 0.4376230836, 0.8211942315, 0.0230764486, 0.9005268812, 0.2147378176, 0.6036583781, 0.7161545157, 0.8246262074, 0.2989832759, 0.5491395593, 0.9779474735, 0.2006554008, 0.8227099776, 0.6018718481, 0.0132929254, 0.2212856710, 0.2032340616, 0.3059777319, 0.9094917178, 0.5409486890, 0.5595687032, 0.2436837852, 0.5649250150, 0.6730466485, 0.4421939552, 0.1432305574, 0.7053307891, 0.6284835935, 0.9216189384, 0.8686438799, 0.8385053873, 0.6248987913, 0.7697140574, 0.9808958173, 0.7571622133, 0.2297872156, 0.4201298952, 0.1305913031, 0.4572514296, 0.3072260618, 0.4668756723, 0.1919649392, 0.2050305754, 0.6062370539, 0.0006580966, 0.6217135191, 0.5123317838, 0.7305839658, 0.0610331446, 0.3614645600, 0.6455501914, 0.2919872701, 0.6446499228, 0.6293424964, 0.6947519779, 0.2680567801, 0.9756787419, 0.6422977448, 0.6911727786, 0.0343145914, 0.4764069021, 0.0876256451, 0.2926266789, 0.0487026349, 0.3558900952, 0.7788275480, 0.8566400409, 0.4791142642, 0.0595066175, 0.9609330297, 0.4075229764, 0.8758037090, 0.3485401869, 0.7945867181, 0.3457054794, 0.3327955306, 0.2870546579, 0.5697714090, 0.6144676208, 0.3251711428, 0.2342026234, 0.4153896868, 0.2149699926, 0.1064170301, 0.7240911722, 0.8196219206, 0.0208647959, 0.3081029952, 0.5742419958, 0.3027088642, 0.5005563498, 0.1707910597, 0.3358575106, 0.2290909439, 0.7788143754, 0.7611069679, 0.3525909781, 0.2308424413, 0.2585839927, 0.5973339677, 0.3728699684, 0.4975571036, 0.0781342834, 0.7119221091, 0.3926881850, 0.5501778126, 0.7364945412, 0.4965503812, 0.8785862923, 0.6024044752, 0.2638861239, 0.9093352556, 0.9069826007, 0.0359279662, 0.4043401778, 0.3457658887, 0.1013033912, 0.1810855120, 0.4946146905, 0.0194541160, 0.5453770161, 0.7965603471, 0.5493819714, 0.2422309667, 0.8376919031, 0.8350337148, 0.1898939908, 0.4576793313, 0.9535705447, 0.1353026628, 0.9474196434, 0.4256035388, 0.0255583692, 0.9593925476, 0.9245427847, 0.9780472517, 0.4356954992, 0.5673046708, 0.7346579432, 0.8614835143, 0.8782553673, 0.3395713866, 0.0013978065, 0.7640301585, 0.2504623234, 0.3626150787, 0.6888222694, 0.9404846430, 0.3519821763, 0.6855628490, 0.2415955663, 0.2107568830, 0.7718742490, 0.3419062793, 0.1280658394, 0.5126360059, 0.1722176671, 0.6543742418, 0.4206473231, 0.2138152719, 0.4514643848, 0.4293326437, 0.0042719250, 0.3195750117, 0.3874749541, 0.6262724400, 0.1620737463, 0.7417458892, 0.8521968126, 0.6405420303, 0.0713626966, 0.0474211276, 0.9068223834, 0.8541609645, 0.4279667437, 0.9738950133, 0.7167884707, 0.6812457442, 0.7938374281, 0.2077793330, 0.5163270831, 0.8487322927, 0.6320008039, 0.5116547942, 0.0056989277, 0.5253843665, 0.1517033428, 0.9921303988, 0.8305052519, 0.0771176443, 0.4621275961, 0.0299932379, 0.8129007220, 0.0946875364, 0.4544205368, 0.0143135618, 0.6373457313, 0.8202091455, 0.3447127640, 0.8560513258, 0.8079835773, 0.9697201252, 0.1521986276, 0.2269581258, 0.2245485932, 0.3396310210, 0.2649262249, 0.7799206972, 0.4020069242, 0.4444113672, 0.8123176098, 0.6460852027, 0.2041657269, 0.7889582515, 0.6526331902, 0.6626461744, 0.6049868464, 0.6901782155, 0.3364612758, 0.3053490818, 0.1905532777, 0.5362346172, 0.3618801832, 0.3485457003, 0.4509411156, 0.5986957550, 0.7858221531, 0.8822937012, 0.8280826807, 0.5261783004, 0.7312974334, 0.6962512732, 0.5243815780, 0.2492258698, 0.1734466404, 0.2547666430, 0.9950503111, 0.1781345457, 0.5630541444, 0.4552696049, 0.8874762058, 0.5965846777, 0.3575465977, 0.1213323772, 0.2790489793, 0.3157011569, 0.6218565702, 0.0304181967, 0.4112739265, 0.7361903787, 0.6753587723, 0.3667163849, 0.6275368929, 0.4185036719, 0.4791659117, 0.1246187463, 0.6651734114, 0.1778147966, 0.8796271682, 0.3000938296, 0.5996896029, 0.5020698309, 0.1601593345, 0.4467433393, 0.0287379269, 0.9011575580, 0.2722401917, 0.1642841995, 0.9468663335, 0.0238759480, 0.7811399102, 0.2070412934, 0.3746992052, 0.8473496437, 0.3498605192, 0.2693480551, 0.1523104310, 0.9660695791, 0.8762652278, 0.1654927284, 0.8743498921, 0.3143339157, 0.3896550536, 0.7256560922, 0.2408472896, 0.0930071324, 0.3269865215, 0.8070673347, 0.1218842566, 0.9943904281, 0.6901395917, 0.9491872787, 0.3617239892, 0.5459694862, 0.9408421516, 0.5354272127, 0.0377946161, 0.3319100142, 0.9823720455, 0.2373940945, 0.2439561784, 0.0767217800, 0.1102360934, 0.6404867172, 0.7430088520, 0.0165513344, 0.9841650128, 0.0532640740, 0.1635770351, 0.3721100390, 0.0598411299, 0.6548883319, 0.3812481761, 0.8741319180, 0.6431996226, 0.0550124273, 0.2009697258, 0.6922588348, 0.0673767105, 0.3385711610, 0.6945076585, 0.7870846987, 0.3323138356, 0.1601967812, 0.9595350623, 0.6049567461, 0.2068863660, 0.2562771440, 0.1041606516, 0.3444063365, 0.1464221030, 0.8932089210, 0.2040112168, 0.3407483399, 0.3251829743, 0.4777953327, 0.0534981787, 0.3613175154, 0.6707065105, 0.1188806742, 0.8228670359, 0.9907929897, 0.1556126177, 0.5561179519, 0.0124231419, 0.2054836601, 0.5855912566, 0.8455434442, 0.2268345803, 0.1841085702, 0.1096092239, 0.8316007257, 0.5046240687, 0.2195746899, 0.9222528338, 0.3633532226, 0.9383196831, 0.8803531528, 0.5124011636, 0.3909464478, 0.2731699646, 0.1102369502, 0.7489478588, 0.0600390583, 0.9290241599, 0.1041191891, 0.9347958565, 0.5584807396, 0.7331624031, 0.2267376930, 0.2868649662, 0.0016489516, 0.2301262319, 0.5107504129, 0.6500277519, 0.6766125560, 0.2019786686, 0.5890167952, 0.7182423472, 0.6890133023, 0.4442900419, 0.5760958791, 0.1364797056, 0.8246579766, 0.2527448535, 0.5444371700, 0.1561367512, 0.7551656961, 0.7171260715, 0.4264259040, 0.3883202970, 0.9166873693, 0.6557167768, 0.0264711548, 0.0761224255, 0.4693228602, 0.5476956964, 0.6261154413, 0.7666952610, 0.9579501152, 0.2581985295, 0.2322760671, 0.8342292905, 0.8143266439, 0.5771137476, 0.5815665126, 0.9772894382, 0.2359700650, 0.6501487494, 0.7841209769, 0.2793208659, 0.1745450795, 0.9626912475, 0.2373798192, 0.1235965416, 0.4632637799, 0.3763884604, 0.9971673489, 0.3533810079, 0.3203127384, 0.6102763414, 0.3859500289, 0.5929466486, 0.6658803821, 0.4130606949, 0.0352911949, 0.9713683128, 0.7546037436, 0.9780107737, 0.3970599473, 0.0187621433, 0.4941402078, 0.7670620680, 0.5360869765, 0.9634684920, 0.5996263027, 0.1895584762, 0.1214910895, 0.7381310463, 0.4301493466, 0.7403219938, 0.4817020297, 0.1843791455, 0.6473838091, 0.4138627350, 0.6825908422, 0.4481185675, 0.2030784935, 0.8468620777, 0.8059213758, 0.7525423169, 0.1854387224, 0.9046887755, 0.6654230952, 0.2029620409, 0.7164457440, 0.4172891080, 0.7797588110, 0.4135729969, 0.0026064927, 0.8375009894, 0.8355652690, 0.9187932014, 0.6724888086, 0.0276171323, 0.9106697440, 0.4562708735, 0.3417910039, 0.1569930464, 0.2029796541, 0.5049355626, 0.8143045306, 0.2432538420, 0.1068324223, 0.6258177757, 0.9749278426, 0.5378444791, 0.1657523215, 0.1930697113, 0.4833569825, 0.8000370264, 0.4315882921, 0.7571453452, 0.6069541574, 0.2073590159, 0.8702615499, 0.1951662153, 0.9303797483, 0.9241660833, 0.2795540988, 0.4241578877, 0.2383123934, 0.8627647758, 0.1700671613, 0.9635605216, 0.2514486313, 0.7766968012, 0.7126773596, 0.7009662986, 0.1317531914, 0.1318600327, 0.5509422421, 0.2159194350, 0.7851343751, 0.7231494188, 0.3523120880, 0.4999881089, 0.8202708960, 0.6340972185, 0.9181259274, 0.0057039275, 0.7197939754, 0.3580873907, 0.1026016176, 0.9657412767, 0.1973488480, 0.8099604845, 0.3302915096, 0.7635477781, 0.7097011805, 0.6271768212, 0.6583901644, 0.2334843278, 0.9448583126, 0.7434690595, 0.4068029821, 0.8815746307, 0.6311643124, 0.3891237080, 0.1507531852, 0.5215465426, 0.3248603344, 0.5837653279, 0.6689655185, 0.1362081915, 0.5130022764, 0.8519401550, 0.4397114217, 0.4129846096, 0.8706676960, 0.4183416367, 0.1135022715, 0.3501874208, 0.1142706573, 0.4111732543, 0.3972048163, 0.0740565360, 0.8445752263, 0.5659885406, 0.1107598469, 0.1261267066, 0.3106530905, 0.9623307586, 0.0014953646, 0.0421718284, 0.9182401299, 0.6180395484, 0.7947646379, 0.4402076006, 0.7980208993, 0.6131495237, 0.8885827065, 0.9406354427, 0.4568731785, 0.8838264346, 0.7086120248, 0.2050074339, 0.8598041534, 0.6360205412, 0.6444933414, 0.1086360887, 0.2146544755, 0.4044065177, 0.8566969037, 0.0974318087, 0.9650754929, 0.7885782719, 0.5817304850, 0.0668027699, 0.2600722611, 0.9546993971, 0.2609280050, 0.2063084394, 0.2960519791, 0.8144530654, 0.5386683941, 0.2757037580, 0.3237824142, 0.3469774723, 0.5878881812, 0.8034821153, 0.7495883107, 0.8035441637, 0.6059562564, 0.2713213861, 0.4108335674, 0.5539482832, 0.5046381950, 0.8435614705, 0.3766961098, 0.7583506107, 0.6175935268, 0.3487794399, 0.0058784639, 0.2900554240, 0.9057408571, 0.1079123169, 0.3200630546, 0.7326458693, 0.0237412248, 0.2757625282, 0.8461791873, 0.6101186872, 0.3705151379, 0.6318973899, 0.4013423026, 0.0222425349, 0.0391604938, 0.6966052055, 0.3186582327, 0.3277960122, 0.3301376998, 0.0874366611, 0.3782529831, 0.1412206143, 0.2574128807, 0.3423563242, 0.7656893730, 0.2097123116, 0.8109381199, 0.4845644534, 0.1744513661, 0.3877931535, 0.5369505286, 0.0147142150, 0.2457712293, 0.4901090264, 0.6373463869, 0.2244705260, 0.6722853184, 0.2888159454, 0.5694347620, 0.3042352200, 0.3482132256, 0.5619021654, 0.6760555506, 0.2648956776, 0.9160912037, 0.8973199129, 0.8901007175, 0.8260267973, 0.2438062280, 0.8338996172, 0.7751584649, 0.1436893344, 0.3578631580, 0.8111414909, 0.9454294443, 0.6478928924, 0.0714371502, 0.0711339787, 0.6473786235, 0.0266824700, 0.2442116290, 0.5528301001, 0.2558279037, 0.3684701622, 0.6729193330, 0.8132147193, 0.5830360651, 0.8655517101, 0.0593610443, 0.9748560190, 0.0221947283, 0.6729801893, 0.5001031756, 0.5116565824, 0.2824120522, 0.4552524984, 0.1693765223, 0.1908069402, 0.7663541436, 0.5339511037, 0.0649234429, 0.6125215292, 0.6771115661, 0.6019635797, 0.6840563416, 0.9653987288, 0.1369341463, 0.8428027630, 0.5227881670, 0.5990189910, 0.0936695337, 0.3645765185, 0.9354769588, 0.6745044589, 0.2816980183, 0.3783183694, 0.7331027389, 0.4139548242, 0.1671119779, 0.6703656316, 0.8604171872, 0.6643752456, 0.7547178268, 0.1386961490, 0.4443438351, 0.3267543018, 0.3348949254, 0.9952459931, 0.4534417391, 0.2089741081 };
const static std::vector<float> while_dynamic_output0{ 0.0388205424, 0.0426156297, 0.0980401114, 0.0568757951, 0.1230962500, 0.0412184112, 0.0595490113, 0.4391007423, 0.0377574340, 0.0629260018 };
// clang-format on

TEST_F(TestWhileDynamicModelLoaded, run_verify)
{
  ASSERT_EQ(nnfw_set_available_backends(_session, "cpu"), NNFW_STATUS_NO_ERROR);
  ASSERT_EQ(nnfw_prepare(_session), NNFW_STATUS_NO_ERROR);

  std::vector<float> actual_output0(10);

  nnfw_tensorinfo ti = {NNFW_TYPE_TENSOR_FLOAT32, 3, {1, 28, 28}};
  ASSERT_EQ(nnfw_set_input_tensorinfo(_session, 0, &ti), NNFW_STATUS_NO_ERROR);

  set_input_output(_session, while_dynamic_input0, &actual_output0);

  ASSERT_EQ(nnfw_run(_session), NNFW_STATUS_NO_ERROR);

  nnfw_tensorinfo ti_output0_expected = {NNFW_TYPE_TENSOR_FLOAT32, 2, {1, 10}};
  ASSERT_EQ(nnfw_output_tensorinfo(_session, 0, &ti), NNFW_STATUS_NO_ERROR);
  ASSERT_TRUE(tensorInfoEqual(ti, ti_output0_expected));

  // output value check
  for (int i = 0; i < actual_output0.size(); ++i)
    ASSERT_FLOAT_EQ(while_dynamic_output0[i], actual_output0[i]);
}

TEST_F(TestWhileDynamicModelLoaded, neg_run_verify)
{
  ASSERT_EQ(nnfw_set_available_backends(_session, "cpu"), NNFW_STATUS_NO_ERROR);
  ASSERT_EQ(nnfw_prepare(_session), NNFW_STATUS_NO_ERROR);

  nnfw_tensorinfo ti = {NNFW_TYPE_TENSOR_FLOAT32, 3, {1, 28, 28}};
  ASSERT_EQ(nnfw_set_input_tensorinfo(_session, 0, &ti), NNFW_STATUS_NO_ERROR);

  // Insufficient size of output (10 or more is sufficient)
  std::vector<float> actual_output0(9);

  set_input_output(_session, while_dynamic_input0, &actual_output0);

  // TODO Change error code NNFW_STATUS_ERROR -> NNFW_INSUFFICIENT_OUTPUT_SIZE
  ASSERT_EQ(nnfw_run(_session), NNFW_STATUS_ERROR);
}
