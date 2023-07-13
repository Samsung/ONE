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

#include <nnfw.h>

#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>

#include "../src/rawdataloader.h"
#include "../src/nnfw_util.h"

namespace
{
using namespace onert_train;

class DataFileGenerator
{
public:
  DataFileGenerator(uint32_t data_length)
    : _data_length{data_length}, _input_file{"input.bin"}, _expected_file{"expected.bin"}
  {
  }
  ~DataFileGenerator()
  {
    try
    {
      if (std::remove(_input_file.c_str()) != 0)
      {
        std::cerr << "Failed to remove " << _input_file << std::endl;
      }
      if (std::remove(_expected_file.c_str()) != 0)
      {
        std::cerr << "Failed to remove " << _expected_file << std::endl;
      }
    }
    catch (const std::exception &e)
    {
      std::cerr << "Exception: " << e.what() << std::endl;
    }
  }

  template <typename T>
  const std::string &generateInputData(const std::vector<std::vector<T>> &data)
  {
    generateData(_input_file, data);
    return _input_file;
  }

  template <typename T>
  const std::string &generateExpectedData(const std::vector<std::vector<T>> &data)
  {
    generateData(_expected_file, data);
    return _expected_file;
  }

private:
  template <typename T>
  void generateData(const std::string &name, const std::vector<std::vector<T>> &data)
  {
    try
    {
      std::ofstream file(name, std::ios::binary);
      for (uint32_t i = 0; i < data.size(); ++i)
      {
        for (uint32_t j = 0; j < _data_length; ++j)
        {
          for (uint32_t k = 0; k < data[i].size(); ++k)
          {
            file.write(reinterpret_cast<const char *>(&data[i][k]), sizeof(data[i][k]));
          }
        }
      }
    }
    catch (const std::exception &e)
    {
      std::cerr << "Exception: " << e.what() << std::endl;
    }
  }

private:
  uint32_t _data_length;
  std::string _input_file;
  std::string _expected_file;
};

class RawDataLoaderTest : public testing::Test
{
protected:
  void SetUp() override { nnfw_create_session(&_session); }

  void TearDown() override { nnfw_close_session(_session); }

  nnfw_session *_session = nullptr;
};

TEST_F(RawDataLoaderTest, loadDatas_1)
{
  const uint32_t data_length = 100;
  const uint32_t num_input = 1;
  const uint32_t num_expected = 1;
  const uint32_t batch_size = 16;

  // Set data tensor info
  nnfw_tensorinfo in_info = {
    .dtype = NNFW_TYPE_TENSOR_INT32,
    .rank = 4,
    .dims = {1, 2, 2, 2},
  };
  std::vector<nnfw_tensorinfo> in_infos{in_info};

  nnfw_tensorinfo expected_info = {
    .dtype = NNFW_TYPE_TENSOR_INT32,
    .rank = 4,
    .dims = {1, 1, 1, 1},
  };
  std::vector<nnfw_tensorinfo> expected_infos{expected_info};

  // Generate test data
  std::vector<std::vector<uint32_t>> in(num_input);
  for (uint32_t i = 0; i < num_input; ++i)
  {
    in[i].resize(num_elems(&in_infos[i]));
    std::generate(in[i].begin(), in[i].end(), [this] {
      static uint32_t i = 0;
      return i++;
    });
  }

  std::vector<std::vector<uint32_t>> expected(num_expected);
  for (uint32_t i = 0; i < num_expected; ++i)
  {
    expected[i].resize(num_elems(&expected_infos[i]));
    std::generate(expected[i].begin(), expected[i].end(), [in, i] {
      auto sum = std::accumulate(in[i].begin(), in[i].end(), 0);
      return sum;
    });
  }

  // Generate test data file
  DataFileGenerator file_gen(data_length);
  auto &input_file = file_gen.generateInputData<uint32_t>(in);
  auto &expected_file = file_gen.generateExpectedData<uint32_t>(expected);

  // Set expected datas
  std::vector<std::vector<uint32_t>> expected_in(num_input);
  std::vector<std::vector<uint32_t>> expected_ex(num_expected);
  for (uint32_t i = 0; i < num_input; ++i)
  {
    for (uint32_t j = 0; j < batch_size; ++j)
    {
      expected_in[i].insert(expected_in[i].end(), in[i].begin(), in[i].end());
    }
  }
  for (uint32_t i = 0; i < num_expected; ++i)
  {
    for (uint32_t j = 0; j < batch_size; ++j)
    {
      expected_ex[i].insert(expected_ex[i].end(), expected[i].begin(), expected[i].end());
    }
  }

  // Load test datas
  RawDataLoader loader;
  Generator generator =
    loader.loadData(input_file, expected_file, in_infos, expected_infos, data_length, batch_size);

  // Allocate inputs and expecteds data memory
  std::vector<Allocation> inputs(num_input * batch_size);
  for (uint32_t i = 0; i < num_input; ++i)
  {
    auto bufsz = bufsize_for(&in_infos[i]);
    for (uint32_t j = 0; j < batch_size; ++j)
    {
      inputs[i * batch_size + j].alloc(bufsz);
    }
  }
  std::vector<Allocation> expecteds(num_expected * batch_size);
  for (uint32_t i = 0; i < num_expected; ++i)
  {
    auto bufsz = bufsize_for(&expected_infos[i]);
    for (uint32_t j = 0; j < batch_size; ++j)
    {
      expecteds[i * batch_size + j].alloc(bufsz);
    }
  }

  uint32_t num_sample = data_length / batch_size;
  for (uint32_t i = 0; i < num_sample; ++i)
  {
    auto data = generator(i, inputs, expecteds);

    std::vector<std::vector<uint32_t>> gen_in(num_input);
    for (uint32_t h = 0; h < num_input; ++h)
    {
      auto num_elem = num_elems(&in_infos[h]);
      for (uint32_t j = 0; j < batch_size; ++j)
      {
        for (uint32_t k = 0; k < num_elem; ++k)
        {
          auto inbufs = reinterpret_cast<uint32_t *>(inputs[h * batch_size + j].data()) + k;
          gen_in[h].emplace_back(*inbufs);
        }
      }
    }
    std::vector<std::vector<uint32_t>> gen_ex(num_expected);
    for (uint32_t h = 0; h < num_expected; ++h)
    {
      auto num_elem = num_elems(&expected_infos[h]);
      for (uint32_t j = 0; j < batch_size; ++j)
      {
        for (uint32_t k = 0; k < num_elem; ++k)
        {
          auto exbufs = reinterpret_cast<uint32_t *>(expecteds[h * batch_size + j].data()) + k;
          gen_ex[h].emplace_back(*exbufs);
        }
      }
    }

    EXPECT_EQ(gen_in, expected_in);
    EXPECT_EQ(gen_ex, expected_ex);
  }
}

// Float32 test case
TEST_F(RawDataLoaderTest, loadData_2)
{
  const uint32_t data_length = 100;
  const uint32_t num_input = 1;
  const uint32_t num_expected = 1;
  const uint32_t batch_size = 16;

  // Set data tensor info
  nnfw_tensorinfo in_info = {
    .dtype = NNFW_TYPE_TENSOR_FLOAT32,
    .rank = 4,
    .dims = {1, 2, 2, 2},
  };
  std::vector<nnfw_tensorinfo> in_infos{in_info};

  nnfw_tensorinfo expected_info = {
    .dtype = NNFW_TYPE_TENSOR_FLOAT32,
    .rank = 4,
    .dims = {1, 1, 1, 1},
  };
  std::vector<nnfw_tensorinfo> expected_infos{expected_info};

  EXPECT_EQ(in_infos.size(), num_input);
  EXPECT_EQ(expected_infos.size(), num_expected);

  // Generate test data
  std::vector<std::vector<float>> in(num_input);
  for (uint32_t i = 0; i < num_input; ++i)
  {
    in[i].resize(num_elems(&in_infos[i]));
    std::generate(in[i].begin(), in[i].end(), [this] {
      static float i = 0.f;
      return i += 1.1f;
    });
  }

  std::vector<std::vector<float>> expected(num_expected);
  for (uint32_t i = 0; i < num_expected; ++i)
  {
    expected[i].resize(num_elems(&expected_infos[i]));
    std::generate(expected[i].begin(), expected[i].end(), [in, i] {
      auto sum = std::accumulate(in[i].begin(), in[i].end(), 0.f);
      return sum;
    });
  }

  // Generate test data file
  DataFileGenerator file_gen(data_length);
  auto &input_file = file_gen.generateInputData<float>(in);
  auto &expected_file = file_gen.generateExpectedData<float>(expected);

  // Set expected datas
  std::vector<std::vector<float>> expected_in(num_input);
  std::vector<std::vector<float>> expected_ex(num_expected);
  for (uint32_t i = 0; i < num_input; ++i)
  {
    for (uint32_t j = 0; j < batch_size; ++j)
    {
      expected_in[i].insert(expected_in[i].end(), in[i].begin(), in[i].end());
    }
  }
  for (uint32_t i = 0; i < num_expected; ++i)
  {
    for (uint32_t j = 0; j < batch_size; ++j)
    {
      expected_ex[i].insert(expected_ex[i].end(), expected[i].begin(), expected[i].end());
    }
  }

  // Load test datas
  RawDataLoader loader;
  Generator generator =
    loader.loadData(input_file, expected_file, in_infos, expected_infos, data_length, batch_size);

  // Allocate inputs and expecteds data memory
  std::vector<Allocation> inputs(num_input * batch_size);
  for (uint32_t i = 0; i < num_input; ++i)
  {
    auto bufsz = bufsize_for(&in_infos[i]);
    for (uint32_t j = 0; j < batch_size; ++j)
    {
      inputs[i * batch_size + j].alloc(bufsz);
    }
  }
  std::vector<Allocation> expecteds(num_expected * batch_size);
  for (uint32_t i = 0; i < num_expected; ++i)
  {
    auto bufsz = bufsize_for(&expected_infos[i]);
    for (uint32_t j = 0; j < batch_size; ++j)
    {
      expecteds[i * batch_size + j].alloc(bufsz);
    }
  }

  uint32_t num_sample = data_length / batch_size;
  for (uint32_t i = 0; i < num_sample; ++i)
  {
    auto data = generator(i, inputs, expecteds);

    std::vector<std::vector<float>> gen_in(num_input);
    for (uint32_t h = 0; h < num_input; ++h)
    {
      auto num_elem = num_elems(&in_infos[h]);
      for (uint32_t j = 0; j < batch_size; ++j)
      {
        for (uint32_t k = 0; k < num_elem; ++k)
        {
          auto inbufs = reinterpret_cast<float *>(inputs[h * batch_size + j].data()) + k;
          gen_in[h].emplace_back(*inbufs);
        }
      }
    }
    std::vector<std::vector<float>> gen_ex(num_expected);
    for (uint32_t h = 0; h < num_expected; ++h)
    {
      auto num_elem = num_elems(&expected_infos[h]);
      for (uint32_t j = 0; j < batch_size; ++j)
      {
        for (uint32_t k = 0; k < num_elem; ++k)
        {
          auto exbufs = reinterpret_cast<float *>(expecteds[h * batch_size + j].data()) + k;
          gen_ex[h].emplace_back(*exbufs);
        }
      }
    }

    EXPECT_EQ(gen_in, expected_in);
    EXPECT_EQ(gen_ex, expected_ex);
  }
}

} // namespace
