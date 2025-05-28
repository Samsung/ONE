/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_API_PYTHON_NNFW_TENSORINFO_BINDINGS_H__
#define __ONERT_API_PYTHON_NNFW_TENSORINFO_BINDINGS_H__

#include <pybind11/pybind11.h>

namespace onert
{
namespace api
{
namespace python
{

// Declare binding tensorinfo
void bind_tensorinfo(pybind11::module_ &m);

// Declare binding enums
void bind_nnfw_enums(pybind11::module_ &m);

} // namespace python
} // namespace api
} // namespace onert

#endif // __ONERT_API_PYTHON_NNFW_TENSORINFO_BINDINGS_H__
