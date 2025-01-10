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

#ifndef __ONERT_API_PYTHON_NNFW_TRAININFO_BINDINGS_H__
#define __ONERT_API_PYTHON_NNFW_TRAININFO_BINDINGS_H__

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Declare binding train enums
void bind_nnfw_train_enums(py::module_ &m);

// Declare binding loss info
void bind_nnfw_loss_info(py::module_ &m);

// Declare binding train info
void bind_nnfw_train_info(py::module_ &m);

#endif // __ONERT_API_PYTHON_NNFW_TRAININFO_BINDINGS_H__
