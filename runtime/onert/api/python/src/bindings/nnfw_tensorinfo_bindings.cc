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

#include "nnfw_tensorinfo_bindings.h"

#include <cstdint>
#include <vector>

#include "nnfw_api_wrapper.h"

namespace onert::api::python
{

namespace py = pybind11;

template <typename T> py::tuple shape_to_tuple(const T &shape)
{
  py::tuple t(shape.size());
  for (size_t i = 0; i < shape.size(); i++)
    t[i] = shape[i];
  return t;
}

// Bind the `tensorinfo` class and related `dtype` class.
void bind_tensorinfo(py::module_ &m)
{
  py::class_<dtype>(m, "dtype", "Defines the type of the OneRT tensor.", py::module_local())
    .def("__repr__", [](const dtype &dt) { return std::string("onert.") + dt.name; })
    .def_readonly("dtype", &dtype::dtype, "A numpy data type.");

  py::class_<tensorinfo>(m, "tensorinfo",
                         "Immutable information about the type and shape of a tensor.",
                         py::module_local())
    .def(py::init<struct dtype, tensorinfo::SHAPE>(),
         "Initialize new tensorinfo with dtype and shape.", py::arg("dtype"), py::arg("shape"))
    .def_property_readonly("dtype", &tensorinfo::get_dtype, "The data type of the tensor.")
    .def_property_readonly("rank", &tensorinfo::get_rank,
                           "The rank of the tensor. The maximum supported rank is 6.")
    .def_property_readonly(
      "shape", [](const tensorinfo &ti) { return shape_to_tuple(ti.get_shape()); },
      "The shape of the tensor.")
    .def("__repr__", [](const tensorinfo &ti) {
      auto dtype = py::repr(py::cast(ti.get_dtype())).cast<std::string>();
      auto shape = py::repr(shape_to_tuple(ti.get_shape())).cast<std::string>();
      return "<tensorinfo dtype=" + dtype + " shape=" + shape + ">";
    });

  static const dtype dtypes[] = {
    get_dtype(NNFW_TYPE::NNFW_TYPE_TENSOR_FLOAT32),
    get_dtype(NNFW_TYPE::NNFW_TYPE_TENSOR_INT32),
    get_dtype(NNFW_TYPE::NNFW_TYPE_TENSOR_QUANT8_ASYMM),
    get_dtype(NNFW_TYPE::NNFW_TYPE_TENSOR_UINT8),
    get_dtype(NNFW_TYPE::NNFW_TYPE_TENSOR_BOOL),
    get_dtype(NNFW_TYPE::NNFW_TYPE_TENSOR_INT64),
    get_dtype(NNFW_TYPE::NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED),
    get_dtype(NNFW_TYPE::NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED),
  };

  // Export OneRT dtypes in a submodule
  auto m_dtypes = m.def_submodule("dtypes", "OneRT tensor data types");
  for (const auto &dt : dtypes)
    m_dtypes.attr(dt.name) = dt;
}

void bind_nnfw_enums(py::module_ &m)
{
  // Bind NNFW_TRAIN_LOSS
  py::enum_<NNFW_PREPARE_CONFIG>(m, "prepare_config", py::module_local())
    .value("PREPARE_CONFIG_PROFILE", NNFW_PREPARE_CONFIG_PROFILE)
    .value("ENABLE_INTERNAL_OUTPUT_ALLOC", NNFW_ENABLE_INTERNAL_OUTPUT_ALLOC);
}

} // namespace onert::api::python
