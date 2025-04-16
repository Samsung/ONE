
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
#include "ModelData.h"
#include "ModelEditor.h"

#include <vector>
#include <string>
#include <memory>
#include <sstream>

#include <pybind11.h>
#include <stl.h>
#include <stl_bind.h>

namespace py = pybind11;
using namespace circle_resizer;

using Shapes = std::vector<Shape>;

PYBIND11_MAKE_OPAQUE(Dim);
PYBIND11_MAKE_OPAQUE(Shape);
PYBIND11_MAKE_OPAQUE(Shapes);

PYBIND11_MODULE(circle_resizer_python_api, m)
{
  m.doc() = "circle-resizer module";

  py::class_<Dim> dim(m, "Dim");
  dim.doc() = "circle_resizer::Dim";
  dim.def(py::init<int32_t>());
  dim.def("is_dynamic", &Dim::is_dynamic);
  dim.def("value", &Dim::value);
  dim.def("__eq__", [](const Dim &rhs, const Dim &lhs) { return rhs.value() == lhs.value(); });
  dim.def("__str__", [](const Dim &self) { return std::to_string(self.value()); });

  py::class_<Dim> shape(m, "Shape");
  shape.doc() = "circle_resizer::Shape";
  shape.def("__eq__", [](const Shape &rhs, const Shape &lhs) { return rhs == lhs; });
  shape.def("__str__", [](const Shape &shape) -> std::string {
    std::stringstream ss;
    ss << shape;
    return ss.str();
  });

  auto shapes = py::bind_vector<Shapes>(m, "Shapes");
  shapes.doc() = "circle_resizer::Shapes";
  shapes.def("__str__", [](const Shapes &shapes) -> std::string {
    if (shapes.empty())
    {
      return "";
    }
    std::stringstream ss;
    for (int i = 0; i < shapes.size() - 1; ++i)
    {
      ss << shapes[i] << ", ";
    }
    ss << shapes.back();
    return ss.str();
  });

  py::class_<ModelData, std::shared_ptr<ModelData>> model_data(m, "ModelData");
  model_data.doc() = "circle_resizer::ModelData";
  model_data.def(py::init<const std::vector<uint8_t> &>(), py::arg("buffer"));
  model_data.def(py::init<const std::string &>(), py::arg("model_path"));
  model_data.def("buffer", &ModelData::buffer);
  model_data.def("input_shapes", &ModelData::input_shapes);
  model_data.def("output_shapes", &ModelData::output_shapes);
  model_data.def("save", py::overload_cast<std::ostream &>(&ModelData::save), py::arg("stream"));
  model_data.def("save", py::overload_cast<const std::string &>(&ModelData::save),
                 py::arg("output_path"));

  py::class_<ModelEditor> model_editor(m, "ModelEditor");
  model_editor.doc() = "circle_resizer::ModelEditor";
  model_editor.def(py::init<std::shared_ptr<ModelData>>(), py::arg("model_data"));
  model_editor.def("resize_inputs", &ModelEditor::resize_inputs, py::arg("shapes"));
}
