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

#include "nnfw_api_wrapper.h"

#include <pybind11/operators.h>

namespace onert::api::python
{

namespace py = pybind11;

// Bind the `tensorinfo` class
void bind_tensorinfo(py::module_ &m)
{

  static const datatype dtypes[] = {
    datatype(NNFW_TYPE::NNFW_TYPE_TENSOR_FLOAT32),
    datatype(NNFW_TYPE::NNFW_TYPE_TENSOR_INT32),
    datatype(NNFW_TYPE::NNFW_TYPE_TENSOR_QUANT8_ASYMM),
    datatype(NNFW_TYPE::NNFW_TYPE_TENSOR_UINT8),
    datatype(NNFW_TYPE::NNFW_TYPE_TENSOR_BOOL),
    datatype(NNFW_TYPE::NNFW_TYPE_TENSOR_INT64),
    datatype(NNFW_TYPE::NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED),
    datatype(NNFW_TYPE::NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED),
  };

  // Export dedicated OneRT type for tensor types. The presence of the "dtype"
  // property allows this type to be used directly with numpy, e.g.:
  // >>> np.array([3, 6, 3], dtype=onert.float32)
  py::class_<datatype>(m, "dtype", "Defines the type of the OneRT tensor.", py::module_local())
    .def(py::self == py::self)
    .def(py::self != py::self)
    .def("__repr__", [](const datatype &dt) { return std::string("onert.") + dt.name; })
    .def_readonly("name", &datatype::name, "The name of the data type.")
    .def_readonly("dtype", &datatype::py_dtype, "A corresponding numpy data type.")
    .def_property_readonly(
      "itemsize", [](const datatype &dt) { return dt.py_dtype.itemsize(); },
      "The element size of this data-type object.");

  // Export OneRT dtypes in a submodule, so we can batch import them
  auto m_dtypes = m.def_submodule("dtypes", "OneRT tensor data types");
  for (const auto &dt : dtypes)
    m_dtypes.attr(dt.name) = dt;

  py::class_<tensorinfo>(m, "tensorinfo", "tensorinfo describes the type and shape of tensors",
                         py::module_local())
    .def(py::init<>(), "The constructor of tensorinfo")
    .def_readwrite("dtype", &tensorinfo::dtype, "The data type")
    .def_readwrite("rank", &tensorinfo::rank, "The number of dimensions (rank)")
    .def_property(
      "dims", [](const tensorinfo &ti) { return get_dims(ti); },
      [](tensorinfo &ti, const py::list &dims_list) { set_dims(ti, dims_list); },
      "The dimension of tensor. Maximum rank is 6 (NNFW_MAX_RANK).");
}

void bind_nnfw_enums(py::module_ &m)
{
  // Bind NNFW_TRAIN_LOSS
  py::enum_<NNFW_PREPARE_CONFIG>(m, "prepare_config", py::module_local())
    .value("PREPARE_CONFIG_PROFILE", NNFW_PREPARE_CONFIG_PROFILE)
    .value("ENABLE_INTERNAL_OUTPUT_ALLOC", NNFW_ENABLE_INTERNAL_OUTPUT_ALLOC);
}

} // namespace onert::api::python
