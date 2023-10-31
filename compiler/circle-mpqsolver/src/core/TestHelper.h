/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MPQSOLVER_TEST_HELPER_H__
#define __MPQSOLVER_TEST_HELPER_H__

#include <luci/IR/CircleNodes.h>
#include <luci/IR/Module.h>
#include <luci/test/TestIOGraph.h>

namespace mpqsolver
{
namespace test
{
namespace models
{

/**
 * @brief base class of simple graphs used for testing
 */
class SimpleGraph
{
public:
  SimpleGraph() : _g(loco::make_graph()) {}

public:
  void init();

  virtual ~SimpleGraph() = default;
  void transfer_to(luci::Module *module);

protected:
  virtual loco::Node *insertGraphBody(loco::Node *input) = 0;
  virtual void initInput(loco::Node *){};

public:
  std::unique_ptr<loco::Graph> _g;
  luci::CircleInput *_input = nullptr;
  luci::CircleOutput *_output = nullptr;
  uint32_t _channel_size = 16;
  uint32_t _width = 4;
  uint32_t _height = 4;
};

/**
 * @brief simple model with just an Add of input and constant
 */
class AddGraph final : public SimpleGraph
{
protected:
  void initInput(loco::Node *input) override;
  void initMinMax(luci::CircleNode *node);

  loco::Node *insertGraphBody(loco::Node *input) override;

public:
  float _a_min = -1.f;
  float _a_max = 1.f;
  luci::CircleAdd *_add = nullptr;
  luci::CircleConst *_beta = nullptr;
};

class SoftmaxGraphlet
{
public:
  SoftmaxGraphlet() = default;
  virtual ~SoftmaxGraphlet() = default;

  void init(loco::Graph *g);

protected:
  void initMinMax(luci::CircleNode *node, float min, float max);

public:
  luci::CircleAbs *_ifm = nullptr;
  luci::CircleReduceMax *_max = nullptr;
  luci::CircleSub *_sub = nullptr;
  luci::CircleExp *_exp = nullptr;
  luci::CircleSum *_sum = nullptr;
  luci::CircleDiv *_div = nullptr;

protected:
  luci::CircleConst *_softmax_indices = nullptr;
};

class SoftmaxTestGraph : public luci::test::TestIOGraph, public SoftmaxGraphlet
{
public:
  SoftmaxTestGraph() = default;

  void init(void);
};

} // namespace models

namespace io_utils
{

/**
 * @brief create valid name of temporary file
 */
void makeTemporaryFile(char *name_template);

/**
 * @brief write data to file_path
 */
void writeDataToFile(const std::string &file_path, const std::string &data);

/**
 * @brief create valid name of temporary folder
 */
std::string makeTemporaryFolder(char *name_template);

/**
 * @brief checks whether file exists
 */
bool isFileExists(const std::string &file_path);

} // namespace io_utils
} // namespace test
} // namespace mpqsolver

#endif //__MPQSOLVER_TEST_HELPER_H__
