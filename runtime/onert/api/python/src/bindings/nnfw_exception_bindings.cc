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

#include "nnfw_exception_bindings.h"

#include "nnfw_exceptions.h"

#include <pybind11/pybind11.h>

namespace onert::api::python
{

namespace py = pybind11;

void bind_nnfw_exceptions(py::module_ &m)
{
  // register base first
  py::register_exception<NnfwError>(m, "OnertError", PyExc_RuntimeError);

  // derived exceptions, each inheriting from NnfwError in Python as well
  py::register_exception<NnfwUnexpectedNullError>(m, "OnertUnexpectedNullError",
                                                  m.attr("OnertError").cast<py::object>());
  py::register_exception<NnfwInvalidStateError>(m, "OnertInvalidStateError",
                                                m.attr("OnertError").cast<py::object>());
  py::register_exception<NnfwOutOfMemoryError>(m, "OnertOutOfMemoryError",
                                               m.attr("OnertError").cast<py::object>());
  py::register_exception<NnfwInsufficientOutputError>(m, "OnertInsufficientOutputError",
                                                      m.attr("OnertError").cast<py::object>());
  py::register_exception<NnfwDeprecatedApiError>(m, "OnertDeprecatedApiError",
                                                 m.attr("OnertError").cast<py::object>());
}

} // namespace onert::api::python
