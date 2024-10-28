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

#ifndef __RECORD_MINMAX_LIST_FILE_ITERATOR_H__
#define __RECORD_MINMAX_LIST_FILE_ITERATOR_H__

#include "DataBuffer.h"
#include "DataSetIterator.h"

#include <luci/IR/Module.h>
#include <luci/IR/CircleNodes.h>

#include <string>
#include <vector>

namespace record_minmax
{

class ListFileIterator final : public DataSetIterator
{
public:
  ListFileIterator(const std::string &input_path, luci::Module *module);

  bool hasNext() const override;

  std::vector<DataBuffer> next() override;

  bool check_type_shape() const override;

private:
  std::vector<std::string> _lines;
  uint32_t _curr_idx = 0;
  std::vector<const luci::CircleInput *> _input_nodes;
};

} // namespace record_minmax

#endif // __RECORD_MINMAX_LIST_FILE_ITERATOR_H__
