
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
#include "CircleModel.h"
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

  py::class_<CircleModel, std::shared_ptr<CircleModel>> circle_model(m, "CircleModel");
  circle_model.doc() = "circle_resizer::CircleModel";
  circle_model.def(py::init<const std::vector<uint8_t> &>(), py::arg("buffer"));
  circle_model.def(py::init<const std::string &>(), py::arg("model_path"));
  circle_model.def("input_shapes", &CircleModel::input_shapes);
  circle_model.def("output_shapes", &CircleModel::output_shapes);
  circle_model.def("save", py::overload_cast<std::ostream &>(&CircleModel::save),
                   py::arg("stream"));
  circle_model.def("save", py::overload_cast<const std::string &>(&CircleModel::save),
                   py::arg("output_path"));

  py::class_<ModelEditor> model_editor(m, "ModelEditor");
  model_editor.doc() = "circle_resizer::ModelEditor";
  model_editor.def(py::init<std::shared_ptr<CircleModel>>(), py::arg("circle_model"));
  model_editor.def("resize_inputs", &ModelEditor::resize_inputs, py::arg("shapes"));
}
