/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ModelData.h"

#include "luci/IR/Module.h"
#include "loco/IR/Graph.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <mio/circle/schema_generated.h>

#include <fstream>

using namespace circle_resizer;
using ::testing::HasSubstr;

namespace
{

bool compare_shapes(const std::vector<Shape> &current, const std::vector<Shape> &expected)
{
  if (current.size() != expected.size())
  {
    return false;
  }
  for (size_t i = 0; i < current.size(); ++i)
  {
    if (!(current[i] == expected[i]))
    {
      return false;
    }
  }
  return true;
}

std::string extract_subgraph_name(const std::vector<uint8_t> &buffer)
{
  auto model = circle::GetModel(buffer.data());
  if (model)
  {
    auto subgraphs = model->subgraphs();
    if (subgraphs->size() > 0)
    {
      auto subgraph = subgraphs->Get(0);
      if (subgraph->name()->c_str())
      {
        return subgraph->name()->c_str();
      }
    }
  }
  return "";
}

// change the first subgraph name using buffer as an input
bool change_subgraph_name(std::vector<uint8_t> &buffer, const std::string &name)
{
  auto model = circle::GetMutableModel(buffer.data());
  if (!model)
  {
    return false;
  }
  auto subgraphs = model->mutable_subgraphs();
  auto subgraph = subgraphs->GetMutableObject(0);
  if (subgraph->name()->size() != name.size())
  {
    return false;
  }
  for (size_t i = 0; i < name.size(); ++i)
  {
    subgraph->mutable_name()->Mutate(i, name[i]);
  }
  return true;
}

// change the first subgraph name using loco::Graph as an input
void change_subgraph_name(loco::Graph *graph, const std::string &name) { graph->name(name); }

} // namespace

class ModelDataTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    char *path = std::getenv("ARTIFACTS_PATH");
    if (path == nullptr)
    {
      throw std::runtime_error("environmental variable ARTIFACTS_PATH required for circle-resizer "
                               "tests was not not provided");
    }
    _test_models_dir = path;
  }

protected:
  std::string _test_models_dir;
};

TEST_F(ModelDataTest, proper_input_output_shapes)
{
  ModelData model_data(_test_models_dir + "/Add_000.circle");
  EXPECT_TRUE(compare_shapes(model_data.input_shapes(),
                             std::vector<Shape>{Shape{1, 4, 4, 3}, Shape{1, 4, 4, 3}}));
  EXPECT_TRUE(compare_shapes(model_data.output_shapes(), std::vector<Shape>{Shape{1, 4, 4, 3}}));
}

TEST_F(ModelDataTest, proper_output_stream)
{
  ModelData model_data(_test_models_dir + "/Add_000.circle");
  std::stringstream out_stream;
  model_data.save(out_stream);
  out_stream.seekg(0, std::ios::end);
  EXPECT_TRUE(out_stream.tellg() > 0);
}

TEST_F(ModelDataTest, invalidate_module)
{
  ModelData model_data(_test_models_dir + "/Add_000.circle");
  const auto module_before_name_change = model_data.module();
  const std::string new_subgraph_name = "abcd";
  ASSERT_TRUE(change_subgraph_name(model_data.buffer(), new_subgraph_name));
  model_data.invalidate_module(); // after buffer representation change the module is outdated
  const auto module_after_name_change = model_data.module();
  EXPECT_EQ(module_after_name_change->graph()->name(),
            new_subgraph_name); // check if buffer update applied to the module
}

TEST_F(ModelDataTest, invalidate_buffer)
{
  ModelData model_data(_test_models_dir + "/Add_000.circle");
  const auto buffer_before_name_change = model_data.buffer();
  const std::string new_subgraph_name = "abcd";
  change_subgraph_name(model_data.module()->graph(), new_subgraph_name);
  model_data.invalidate_buffer(); // after module representation change the buffer is outdated
  const auto buffer_after_name_change = model_data.buffer();
  EXPECT_EQ(extract_subgraph_name(buffer_after_name_change),
            new_subgraph_name); // check if module update applied to the buffer
}

TEST_F(ModelDataTest, model_file_not_exist_NEG)
{
  auto file_name = "/not_existed.circle";
  try
  {
    ModelData model_data(file_name);
    FAIL() << "Expected std::runtime_error";
  }
  catch (const std::runtime_error &err)
  {
    EXPECT_THAT(err.what(), HasSubstr("Failed to open file"));
    EXPECT_THAT(err.what(), HasSubstr(file_name));
  }
  catch (...)
  {
    FAIL() << "Expected std::runtime_error, other exception thrown";
  }
}

TEST_F(ModelDataTest, invalid_model_NEG)
{
  try
  {
    ModelData(std::vector<uint8_t>{1, 2, 3, 4, 5});
    FAIL() << "Expected std::runtime_error";
  }
  catch (const std::runtime_error &err)
  {
    EXPECT_THAT(err.what(), HasSubstr("Verification of the model failed"));
  }
  catch (...)
  {
    FAIL() << "Expected std::runtime_error, other exception thrown";
  }
}

TEST_F(ModelDataTest, incorrect_output_stream_NEG)
{
  auto model_data = std::make_shared<ModelData>(_test_models_dir + "/Add_000.circle");
  std::ofstream out_stream;
  try
  {
    model_data->save(out_stream);
  }
  catch (const std::runtime_error &err)
  {
    EXPECT_THAT(err.what(), HasSubstr("Failed to write to output stream"));
  }
  catch (...)
  {
    FAIL() << "Expected std::runtime_error, other exception thrown";
  }
}
