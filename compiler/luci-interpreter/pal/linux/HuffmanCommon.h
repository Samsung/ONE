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

#ifndef LUCI_INTERPRETER_PAL_HUFFMAN_COMMON_H
#define LUCI_INTERPRETER_PAL_HUFFMAN_COMMON_H

#include <memory>

namespace luci_interpreter_pal
{
namespace huffman
{

// Node of prefix tree
template <typename T> struct Node
{
  std::shared_ptr<Node<T>> p_left;
  std::shared_ptr<Node<T>> p_right;
  T data;
  unsigned int freq;
};

// Compare functor for priority queue
template <typename T> struct CompareNodes
{
  bool operator()(std::shared_ptr<Node<T>> l, std::shared_ptr<Node<T>> r)
  {
    return l->freq > r->freq;
  }
};

template <typename T>
std::shared_ptr<Node<T>> createNode(T data, unsigned int freq, std::shared_ptr<Node<T>> p_left,
                                    std::shared_ptr<Node<T>> p_right)
{
  std::shared_ptr<Node<T>> node = std::make_unique<Node<T>>();
  node->data = data;
  node->freq = freq;
  node->p_left = p_left;
  node->p_right = p_right;
  return node;
}

} // namespace huffman
} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_HUFFMAN_COMMON_H
