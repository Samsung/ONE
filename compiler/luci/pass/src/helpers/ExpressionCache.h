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

#ifndef __LUCI_PASS_HELPERS_EXPRESSION_CACHE_H__
#define __LUCI_PASS_HELPERS_EXPRESSION_CACHE_H__

#include <luci/IR/CircleNodes.h>

#include <vector>
#include <unordered_map>

namespace luci
{
namespace pass
{

// Expression is defined as a circle node (operator) and its inputs
struct Expression final
{
private:
  // Prevent default ctor
  Expression() = default;

public:
  std::vector<loco::Node *> inputs;
  luci::CircleNode *op = nullptr;

  // Hash function for Expression (used for std::unordered_map)
  struct Hash final
  {
    std::size_t call(const Expression &k) const noexcept;
    std::size_t operator()(const Expression &k) const noexcept { return call(k); }
  };

  // Build expression from a circle node
  static Expression build(luci::CircleNode *node);
};

// Return true if two expressions are the same
// This is a core logic for common subexpression elimination
bool operator==(const Expression &x, const Expression &y);

// Cache for Expression object
class ExpressionCache final
{
public:
  using Key = Expression;
  using Value = luci::CircleNode *;

private:
  std::unordered_map<Key, Value, Key::Hash> _content;

public:
  // Return value for the corresponding key
  // Return nullptr if there is no item with the key
  Value get(const Key &k) const
  {
    auto item = _content.find(k);
    if (item == _content.end())
      return nullptr;

    return item->second;
  }

  // Save circle node for the corresponding key
  void put(const Key &k, const Value v) { _content[k] = v; }
};

} // namespace pass
} // namespace luci

#endif // __LUCI_PASS_HELPERS_EXPRESSION_CACHE_H__
