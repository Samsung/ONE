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

#ifndef __RECORD_MINMAX_RANDOM_ITERATOR_H__
#define __RECORD_MINMAX_RANDOM_ITERATOR_H__

#include "DataBuffer.h"
#include "DataSetIterator.h"

#include <luci/IR/Module.h>
#include <luci/IR/CircleNodes.h>

#include <random>
#include <vector>

namespace record_minmax
{

class RandomIterator final : public DataSetIterator
{
public:
  RandomIterator(luci::Module *module);

  bool hasNext() const override;

  std::vector<DataBuffer> next() override;

  bool check_type_shape() const override;

private:
  std::mt19937 _gen;
  std::vector<const luci::CircleInput *> _input_nodes;
  uint32_t _curr_idx = 0;
  uint32_t _num_data = 0;
};

} // namespace record_minmax

#endif // __RECORD_MINMAX_RANDOM_ITERATOR_H__
