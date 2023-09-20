/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "nnfw_api_wrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(libnnfw_api_pybind, m)
{
  m.doc() = "python nnfw plugin";

  py::class_<tensorinfo>(m, "tensorinfo")
    .def(py::init<>())
    .def_readwrite("dtype", &tensorinfo::dtype)
    .def_readwrite("rank", &tensorinfo::rank)
    .def_property(
      "dims", [](const tensorinfo &ti) { return get_dims(ti); },
      [](tensorinfo &ti, const py::list &dims_list) { set_dims(ti, dims_list); });

  py::class_<NNFW_SESSION>(m, "nnfw_session")
    .def(py::init<const char *, const char *>())
    .def(py::init<const char *, const char *, const char *>())
    .def("close_session", &NNFW_SESSION::close_session)
    .def("set_input_tensorinfo", &NNFW_SESSION::set_input_tensorinfo)
    .def("run", &NNFW_SESSION::run)
    .def("run_async", &NNFW_SESSION::run_async)
    .def("await", &NNFW_SESSION::await)
    .def("set_input", [](NNFW_SESSION &session, uint32_t index,
                         py::array_t<float> &buffer) { session.set_input<float>(index, buffer); })
    .def("set_input", [](NNFW_SESSION &session, uint32_t index,
                         py::array_t<int> &buffer) { session.set_input<int>(index, buffer); })
    .def("set_input",
         [](NNFW_SESSION &session, uint32_t index, py::array_t<uint8_t> &buffer) {
           session.set_input<uint8_t>(index, buffer);
         })
    .def("set_input", [](NNFW_SESSION &session, uint32_t index,
                         py::array_t<bool> &buffer) { session.set_input<bool>(index, buffer); })
    .def("set_input",
         [](NNFW_SESSION &session, uint32_t index, py::array_t<int64_t> &buffer) {
           session.set_input<int64_t>(index, buffer);
         })
    .def("set_input", [](NNFW_SESSION &session, uint32_t index,
                         py::array_t<int8_t> &buffer) { session.set_input<int8_t>(index, buffer); })
    .def("set_input",
         [](NNFW_SESSION &session, uint32_t index, py::array_t<int16_t> &buffer) {
           session.set_input<int16_t>(index, buffer);
         })
    .def("set_output", [](NNFW_SESSION &session, uint32_t index,
                          py::array_t<float> &buffer) { session.set_output<float>(index, buffer); })
    .def("set_output", [](NNFW_SESSION &session, uint32_t index,
                          py::array_t<int> &buffer) { session.set_output<int>(index, buffer); })
    .def("set_output",
         [](NNFW_SESSION &session, uint32_t index, py::array_t<uint8_t> &buffer) {
           session.set_output<uint8_t>(index, buffer);
         })
    .def("set_output", [](NNFW_SESSION &session, uint32_t index,
                          py::array_t<bool> &buffer) { session.set_output<bool>(index, buffer); })
    .def("set_output",
         [](NNFW_SESSION &session, uint32_t index, py::array_t<int64_t> &buffer) {
           session.set_output<int64_t>(index, buffer);
         })
    .def("set_output",
         [](NNFW_SESSION &session, uint32_t index, py::array_t<int8_t> &buffer) {
           session.set_output<int8_t>(index, buffer);
         })
    .def("set_output",
         [](NNFW_SESSION &session, uint32_t index, py::array_t<int16_t> &buffer) {
           session.set_output<int16_t>(index, buffer);
         })
    .def("input_size", &NNFW_SESSION::input_size)
    .def("output_size", &NNFW_SESSION::output_size)
    .def("set_input_layout", &NNFW_SESSION::set_input_layout, py::arg("index"),
         py::arg("layout") = "NONE")
    .def("set_output_layout", &NNFW_SESSION::set_output_layout, py::arg("index"),
         py::arg("layout") = "NONE")
    .def("input_tensorinfo", &NNFW_SESSION::input_tensorinfo)
    .def("output_tensorinfo", &NNFW_SESSION::output_tensorinfo)
    .def("query_info_u32", &NNFW_SESSION::query_info_u32);
}
