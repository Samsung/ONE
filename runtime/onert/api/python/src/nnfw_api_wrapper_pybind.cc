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
  m.doc() = "nnfw python plugin";

  py::class_<tensorinfo>(m, "tensorinfo", "tensorinfo describes the type and shape of tensors")
    .def(py::init<>(), "The constructor of tensorinfo")
    .def_readwrite("dtype", &tensorinfo::dtype, "The data type")
    .def_readwrite("rank", &tensorinfo::rank, "The number of dimensions (rank)")
    .def_property(
      "dims", [](const tensorinfo &ti) { return get_dims(ti); },
      [](tensorinfo &ti, const py::list &dims_list) { set_dims(ti, dims_list); },
      "The dimension of tensor. Maximum rank is 6 (NNFW_MAX_RANK).");

  py::class_<NNFW_SESSION>(m, "nnfw_session")
    .def(
      py::init<const char *, const char *>(), py::arg("package_file_path"), py::arg("backends"),
      "Create a new session instance, load model from nnpackage file or directory, "
      "set available backends and prepare session to be ready for inference\n"
      "Parameters:\n"
      "\tpackage_file_path (str): Path to the nnpackage file or unzipped directory to be loaded\n"
      "\tbackends (str): Available backends on which nnfw uses\n"
      "\t\tMultiple backends can be set and they must be separated by a semicolon "
      "(ex: \"acl_cl;cpu\")\n"
      "\t\tAmong the multiple backends, the 1st element is used as the default backend.")
    .def(
      py::init<const char *, const char *, const char *>(), py::arg("package_file_path"),
      py::arg("op"), py::arg("backends"),
      "Create a new session instance, load model from nnpackage file or directory, "
      "set the operation's backend and prepare session to be ready for inference\n"
      "Parameters:\n"
      "\tpackage_file_path (str): Path to the nnpackage file or unzipped directory to be loaded\n"
      "\top (str): operation to be set\n"
      "\tbackends (str): Bakcend on which operation run")
    .def("set_input_tensorinfo", &NNFW_SESSION::set_input_tensorinfo, py::arg("index"),
         py::arg("tensor_info"),
         "Set input model's tensor info for resizing.\n"
         "Parameters:\n"
         "\tindex (int): Index of input to be set (0-indexed)\n"
         "\ttensor_info (tensorinfo): Tensor info to be set")
    .def("run", &NNFW_SESSION::run, "Run inference")
    .def("run_async", &NNFW_SESSION::run_async, "Run inference asynchronously")
    .def("await", &NNFW_SESSION::await, "Wait for asynchronous run to finish")
    .def(
      "set_input",
      [](NNFW_SESSION &session, uint32_t index, py::array_t<float> &buffer) {
        session.set_input<float>(index, buffer);
      },
      py::arg("index"), py::arg("buffer"),
      "Set input buffer\n"
      "Parameters:\n"
      "\tindex (int): Index of input to be set (0-indexed)\n"
      "\tbuffer (numpy): Raw buffer for input")
    .def(
      "set_input",
      [](NNFW_SESSION &session, uint32_t index, py::array_t<int> &buffer) {
        session.set_input<int>(index, buffer);
      },
      py::arg("index"), py::arg("buffer"),
      "Set input buffer\n"
      "Parameters:\n"
      "\tindex (int): Index of input to be set (0-indexed)\n"
      "\tbuffer (numpy): Raw buffer for input")
    .def(
      "set_input",
      [](NNFW_SESSION &session, uint32_t index, py::array_t<uint8_t> &buffer) {
        session.set_input<uint8_t>(index, buffer);
      },
      py::arg("index"), py::arg("buffer"),
      "Set input buffer\n"
      "Parameters:\n"
      "\tindex (int): Index of input to be set (0-indexed)\n"
      "\tbuffer (numpy): Raw buffer for input")
    .def(
      "set_input",
      [](NNFW_SESSION &session, uint32_t index, py::array_t<bool> &buffer) {
        session.set_input<bool>(index, buffer);
      },
      py::arg("index"), py::arg("buffer"),
      "Set input buffer\n"
      "Parameters:\n"
      "\tindex (int): Index of input to be set (0-indexed)\n"
      "\tbuffer (numpy): Raw buffer for input")
    .def(
      "set_input",
      [](NNFW_SESSION &session, uint32_t index, py::array_t<int64_t> &buffer) {
        session.set_input<int64_t>(index, buffer);
      },
      py::arg("index"), py::arg("buffer"),
      "Set input buffer\n"
      "Parameters:\n"
      "\tindex (int): Index of input to be set (0-indexed)\n"
      "\tbuffer (numpy): Raw buffer for input")
    .def(
      "set_input",
      [](NNFW_SESSION &session, uint32_t index, py::array_t<int8_t> &buffer) {
        session.set_input<int8_t>(index, buffer);
      },
      py::arg("index"), py::arg("buffer"),
      "Set input buffer\n"
      "Parameters:\n"
      "\tindex (int): Index of input to be set (0-indexed)\n"
      "\tbuffer (numpy): Raw buffer for input")
    .def(
      "set_input",
      [](NNFW_SESSION &session, uint32_t index, py::array_t<int16_t> &buffer) {
        session.set_input<int16_t>(index, buffer);
      },
      py::arg("index"), py::arg("buffer"),
      "Set input buffer\n"
      "Parameters:\n"
      "\tindex (int): Index of input to be set (0-indexed)\n"
      "\tbuffer (numpy): Raw buffer for input")
    .def(
      "set_output",
      [](NNFW_SESSION &session, uint32_t index, py::array_t<float> &buffer) {
        session.set_output<float>(index, buffer);
      },
      py::arg("index"), py::arg("buffer"),
      "Set output buffer\n"
      "Parameters:\n"
      "\tindex (int): Index of output to be set (0-indexed)\n"
      "\tbuffer (numpy): Raw buffer for output")
    .def(
      "set_output",
      [](NNFW_SESSION &session, uint32_t index, py::array_t<int> &buffer) {
        session.set_output<int>(index, buffer);
      },
      py::arg("index"), py::arg("buffer"),
      "Set output buffer\n"
      "Parameters:\n"
      "\tindex (int): Index of output to be set (0-indexed)\n"
      "\tbuffer (numpy): Raw buffer for output")
    .def(
      "set_output",
      [](NNFW_SESSION &session, uint32_t index, py::array_t<uint8_t> &buffer) {
        session.set_output<uint8_t>(index, buffer);
      },
      py::arg("index"), py::arg("buffer"),
      "Set output buffer\n"
      "Parameters:\n"
      "\tindex (int): Index of output to be set (0-indexed)\n"
      "\tbuffer (numpy): Raw buffer for output")
    .def(
      "set_output",
      [](NNFW_SESSION &session, uint32_t index, py::array_t<bool> &buffer) {
        session.set_output<bool>(index, buffer);
      },
      py::arg("index"), py::arg("buffer"),
      "Set output buffer\n"
      "Parameters:\n"
      "\tindex (int): Index of output to be set (0-indexed)\n"
      "\tbuffer (numpy): Raw buffer for output")
    .def(
      "set_output",
      [](NNFW_SESSION &session, uint32_t index, py::array_t<int64_t> &buffer) {
        session.set_output<int64_t>(index, buffer);
      },
      py::arg("index"), py::arg("buffer"),
      "Set output buffer\n"
      "Parameters:\n"
      "\tindex (int): Index of output to be set (0-indexed)\n"
      "\tbuffer (numpy): Raw buffer for output")
    .def(
      "set_output",
      [](NNFW_SESSION &session, uint32_t index, py::array_t<int8_t> &buffer) {
        session.set_output<int8_t>(index, buffer);
      },
      py::arg("index"), py::arg("buffer"),
      "Set output buffer\n"
      "Parameters:\n"
      "\tindex (int): Index of output to be set (0-indexed)\n"
      "\tbuffer (numpy): Raw buffer for output")
    .def(
      "set_output",
      [](NNFW_SESSION &session, uint32_t index, py::array_t<int16_t> &buffer) {
        session.set_output<int16_t>(index, buffer);
      },
      py::arg("index"), py::arg("buffer"),
      "Set output buffer\n"
      "Parameters:\n"
      "\tindex (int): Index of output to be set (0-indexed)\n"
      "\tbuffer (numpy): Raw buffer for output")
    .def("input_size", &NNFW_SESSION::input_size,
         "Get the number of inputs defined in loaded model\n"
         "Returns:\n"
         "\tint: The number of inputs")
    .def("output_size", &NNFW_SESSION::output_size,
         "Get the number of outputs defined in loaded model\n"
         "Returns:\n"
         "\tint: The number of outputs")
    .def("set_input_layout", &NNFW_SESSION::set_input_layout, py::arg("index"),
         py::arg("layout") = "NONE",
         "Set the layout of an input\n"
         "Parameters:\n"
         "\tindex (int): Index of input to be set (0-indexed)\n"
         "\tlayout (str): Layout to set to target input")
    .def("set_output_layout", &NNFW_SESSION::set_output_layout, py::arg("index"),
         py::arg("layout") = "NONE",
         "Set the layout of an output\n"
         "Parameters:\n"
         "\tindex (int): Index of output to be set (0-indexed)\n"
         "\tlayout (str): Layout to set to target output")
    .def("input_tensorinfo", &NNFW_SESSION::input_tensorinfo, py::arg("index"),
         "Get i-th input tensor info\n"
         "Parameters:\n"
         "\tindex (int): Index of input\n"
         "Returns:\n"
         "\ttensorinfo: Tensor info (shape, type, etc)")
    .def("output_tensorinfo", &NNFW_SESSION::output_tensorinfo, py::arg("index"),
         "Get i-th output tensor info\n"
         "Parameters:\n"
         "\tindex (int): Index of output\n"
         "Returns:\n"
         "\ttensorinfo: Tensor info (shape, type, etc)");
}
