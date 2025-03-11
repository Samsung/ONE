
/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Shape.h"

#include <pybind11.h>
#include <stl_bind.h>

namespace py = pybind11;
using namespace circle_resizer;

PYBIND11_MAKE_OPAQUE(Shape);

PYBIND11_MODULE(circle_resizer_python_api, m)
{
  m.doc() = "circle-resizer module";

  py::class_<Dim> dim(m, "Dim");
  dim.doc() = "circle_resizer::Dim";
  dim.def(py::init<int32_t>());
  dim.def("is_dynamic", &Dim::is_dynamic);
  dim.def("value", &Dim::value);
  dim.def(
      "__eq__",
      [](const Shape& rhs, const Shape& lhs) {
          return rhs == lhs;
      });

  auto shape = py::bind_vector<Shape>(m, "Shape");
  shape.doc() = "circle_resizer::Shape";
}
