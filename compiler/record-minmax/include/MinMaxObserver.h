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

#include "MinMaxVectors.h"

#include <vector>
#include <unordered_map>

namespace record_minmax
{

class MinMaxMap
{
public:
  // Record min/max of node
  void recordMinMax(const luci::CircleNode *node, float min, float max)
  {
    MinMaxVectors &vectors = _minmax_map[node];
    vectors.min_vector.push_back(min);
    vectors.max_vector.push_back(max);
  }

  void appendMinMaxVector(const luci::CircleNode *node, const MinMaxVectors &minmax_vector)
  {
    MinMaxVectors &vectors = _minmax_map[node];
    vectors.min_vector.insert(vectors.min_vector.end(), minmax_vector.min_vector.begin(),
                              minmax_vector.min_vector.end());
    vectors.max_vector.insert(vectors.max_vector.end(), minmax_vector.max_vector.begin(),
                              minmax_vector.max_vector.end());
  }

  const std::unordered_map<const luci::CircleNode *, MinMaxVectors> *getMap() const
  {
    return &_minmax_map;
  }

private:
  std::unordered_map<const luci::CircleNode *, MinMaxVectors> _minmax_map;
};

class MinMaxObserver : public luci_interpreter::ExecutionObserver
{
public:
  MinMaxObserver()
  {
    // Do nothing
  }

  void postTensorWrite(const luci::CircleNode *node,
                       const luci_interpreter::Tensor *tensor) override;

  const MinMaxMap *minMaxData() { return &_minmax_data; }

private:
  MinMaxMap _minmax_data;
};

} // namespace record_minmax

#endif // __RECORD_MINMAX_MINMAXOBSERVER_H__
