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

#ifndef __DALGONA_UTILS_H__
#define __DALGONA_UTILS_H__

#include <luci_interpreter/Interpreter.h>
#include <luci_interpreter/core/Tensor.h>
#include <luci/IR/CircleOpcode.h>
#include <luci/IR/CircleNodeDecl.h>
#include <luci/IR/LuciNodeMixins.h>

#include <pybind11/numpy.h>

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

std::vector<py::dict> inputsPyArray(const luci::CircleNode *node,
                                    luci_interpreter::Interpreter *interpreter);

py::dict outputPyArray(const luci::CircleNode *node, luci_interpreter::Interpreter *interpreter);

} // namespace dalgona

#endif // __DALGONA_UTILS_H__
