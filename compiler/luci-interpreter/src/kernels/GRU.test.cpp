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

#include "kernels/GRU.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class GRUTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(GRUTest, floatTest)
{
  Shape input_shape{2, 1, 2};
  std::vector<float> input_data{0.98045033, 0.39546537, 0.5209594, 0.72873044};

  Shape ref_output_shape{1, 1, 2};
  std::vector<float> ref_output_data{0.22777566, -0.1976251};

  Shape hidden_hidden_shape{6, 2};
  std::vector<float> hidden_hidden_data{
    0.8073279857635498,   -0.5218740105628967, 0.1166749969124794,  0.33110499382019043,
    0.2770330011844635,   0.23767800629138947, 0.1293960064649582,  0.17175200581550598,
    -0.15584999322891235, 0.8137810230255127,  -0.2667199969291687, -0.23028500378131866};
  Shape hidden_input_shape{6, 2};
  std::vector<float> hidden_input_data{
    -0.1928129941225052, -0.4582270085811615, -0.17884500324726105, -0.27543601393699646,
    0.704787015914917,   0.1874309927225113,  -0.28071099519729614, -0.40605801343917847,
    -0.4156219959259033, 0.6752780079841614,  0.4272859990596771,   -0.24114100635051727};

  Shape state_shape{1, 2};
  std::vector<float> state_data{0.0, 0.0};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Tensor hidden_hidden_tensor = makeInputTensor<DataType::FLOAT32>(
    hidden_hidden_shape, hidden_hidden_data, _memory_manager.get());

  Tensor hidden_input_tensor = makeInputTensor<DataType::FLOAT32>(
    hidden_input_shape, hidden_input_data, _memory_manager.get());

  Tensor state_tensor =
    makeInputTensor<DataType::FLOAT32>(state_shape, state_data, _memory_manager.get());

  GRUParams params{};

  GRU kernel(&input_tensor, &hidden_hidden_tensor, nullptr, &hidden_input_tensor, nullptr,
             &state_tensor, &output_tensor, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
}

TEST_F(GRUTest, Unmatched_io_type_NEG)
{
  Shape input_shape{2, 1, 2};
  std::vector<float> input_data{0.98045033, 0.39546537, 0.5209594, 0.72873044};

  Shape ref_output_shape{1, 1, 2};
  std::vector<float> ref_output_data{0.22777566, -0.1976251};

  Shape hidden_hidden_shape{6, 2};
  std::vector<float> hidden_hidden_data{
    0.8073279857635498,   -0.5218740105628967, 0.1166749969124794,  0.33110499382019043,
    0.2770330011844635,   0.23767800629138947, 0.1293960064649582,  0.17175200581550598,
    -0.15584999322891235, 0.8137810230255127,  -0.2667199969291687, -0.23028500378131866};
  Shape hidden_input_shape{6, 2};
  std::vector<float> hidden_input_data{
    -0.1928129941225052, -0.4582270085811615, -0.17884500324726105, -0.27543601393699646,
    0.704787015914917,   0.1874309927225113,  -0.28071099519729614, -0.40605801343917847,
    -0.4156219959259033, 0.6752780079841614,  0.4272859990596771,   -0.24114100635051727};

  Shape state_shape{1, 2};
  std::vector<float> state_data{0.0, 0.0};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U32);

  Tensor hidden_hidden_tensor = makeInputTensor<DataType::FLOAT32>(
    hidden_hidden_shape, hidden_hidden_data, _memory_manager.get());

  Tensor hidden_input_tensor = makeInputTensor<DataType::FLOAT32>(
    hidden_input_shape, hidden_input_data, _memory_manager.get());

  Tensor state_tensor =
    makeInputTensor<DataType::FLOAT32>(state_shape, state_data, _memory_manager.get());

  GRUParams params{};

  GRU kernel(&input_tensor, &hidden_hidden_tensor, nullptr, &hidden_input_tensor, nullptr,
             &state_tensor, &output_tensor, params);

  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(GRUTest, Unmatched_weight_size_NEG)
{
  Shape input_shape{2, 1, 2};
  std::vector<float> input_data{0.98045033, 0.39546537, 0.5209594, 0.72873044};

  Shape ref_output_shape{1, 1, 2};
  std::vector<float> ref_output_data{0.22777566, -0.1976251};

  Shape hidden_hidden_shape{1, 2};
  std::vector<float> hidden_hidden_data{-0.2667199969291687, -0.23028500378131866};
  Shape hidden_input_shape{6, 2};
  std::vector<float> hidden_input_data{
    -0.1928129941225052, -0.4582270085811615, -0.17884500324726105, -0.27543601393699646,
    0.704787015914917,   0.1874309927225113,  -0.28071099519729614, -0.40605801343917847,
    -0.4156219959259033, 0.6752780079841614,  0.4272859990596771,   -0.24114100635051727};

  Shape state_shape{1, 2};
  std::vector<float> state_data{0.0, 0.0};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Tensor hidden_hidden_tensor = makeInputTensor<DataType::FLOAT32>(
    hidden_hidden_shape, hidden_hidden_data, _memory_manager.get());

  Tensor hidden_input_tensor = makeInputTensor<DataType::FLOAT32>(
    hidden_input_shape, hidden_input_data, _memory_manager.get());

  Tensor state_tensor =
    makeInputTensor<DataType::FLOAT32>(state_shape, state_data, _memory_manager.get());

  GRUParams params{};

  GRU kernel(&input_tensor, &hidden_hidden_tensor, nullptr, &hidden_input_tensor, nullptr,
             &state_tensor, &output_tensor, params);

  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
