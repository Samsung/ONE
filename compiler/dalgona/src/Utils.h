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

#ifndef __DALGONA_UTILS_H__
#define __DALGONA_UTILS_H__

#include <luci_interpreter/Interpreter.h>

#include <pybind11/embed.h>

namespace py = pybind11;

namespace dalgona
{

template <typename... Args> void pySafeCall(py::object func, Args... args)
{
  try
  {
    func(args...);
  }
  catch (py::error_already_set &e)
  {
    throw std::runtime_error(e.what());
  }
}

py::dict outputPyArray(const luci::CircleNode *node, luci_interpreter::Interpreter *interpreter);

// Return a vector of Tensors(py::dict) which correspond to node's inputs
std::vector<py::dict> inputsPyArray(const luci::CircleNode *node,
                                    luci_interpreter::Interpreter *interpreter);

// Return a vector of Tensors(py::dict) which correspond to the outputs of multi-out node (ex:
// SPLIT)
std::vector<py::dict> outputsPyArray(const luci::CircleNode *node,
                                     luci_interpreter::Interpreter *interpreter);

py::object none();

bool isMultiOutNode(const luci::CircleNode *node);

} // namespace dalgona

#endif // __DALGONA_UTILS_H__
