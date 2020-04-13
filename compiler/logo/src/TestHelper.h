/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __TEST_HELPER_H__
#define __TEST_HELPER_H__

#include <loco.h>

namespace logo
{
namespace test
{

template <typename T> T *find_first_node_by_type(loco::Graph *g)
{
  T *first_node = nullptr;

  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    first_node = dynamic_cast<T *>(node);
    if (first_node != nullptr)
      break;
  }

  return first_node;
}

} // namespace test
} // namespace logo

#endif // __TEST_HELPER_H__
