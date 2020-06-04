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

#ifndef __RECORD_MINMAX_MINMAXOBSERVER_H__
#define __RECORD_MINMAX_MINMAXOBSERVER_H__

#include <luci_interpreter/Interpreter.h>
#include <luci_interpreter/core/Tensor.h>

#include <vector>
#include <unordered_map>

namespace record_minmax
{

struct MinMaxVectors
{
  std::vector<float> min_vector;
  std::vector<float> max_vector;
};

class MinMaxMap
{
public:
  // Record min/max of node
  void recordMinMax(const luci::CircleNode *node, float min, float max)
  {
    const auto iter = _minmax_map.find(node);
    if (iter == _minmax_map.end())
    {
      auto vectors = std::make_unique<MinMaxVectors>();
      vectors->min_vector.push_back(min);
      vectors->max_vector.push_back(max);
      _minmax_map.emplace(node, std::move(vectors));
    }
    else
    {
      auto vectors = iter->second.get();
      vectors->min_vector.push_back(min);
      vectors->max_vector.push_back(max);
    }
  }

  const std::unordered_map<const luci::CircleNode *, std::unique_ptr<MinMaxVectors>> *getMap() const
  {
    return &_minmax_map;
  }

private:
  std::unordered_map<const luci::CircleNode *, std::unique_ptr<MinMaxVectors>> _minmax_map;
};

class MinMaxObserver : public luci_interpreter::ExecutionObserver
{
public:
  MinMaxObserver()
  {
    // Do nothing
  }

  ~MinMaxObserver() {}

  void postTensorWrite(const luci::CircleNode *node,
                       const luci_interpreter::Tensor *tensor) override;

  const MinMaxMap *minMaxData() { return &_minmax_data; }

private:
  MinMaxMap _minmax_data;
};

} // namespace record_minmax

#endif // __RECORD_MINMAX_MINMAXOBSERVER_H__
