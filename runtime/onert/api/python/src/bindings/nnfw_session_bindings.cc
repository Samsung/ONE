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

#include "nnfw_session_bindings.h"

#include "nnfw_api_wrapper.h"

namespace onert::api::python
{

namespace py = pybind11;

// Bind the `NNFW_SESSION` class with common inference APIs
void bind_nnfw_session(py::module_ &m)
{
  py::class_<NNFW_SESSION>(m, "nnfw_session", py::module_local())
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
    .def("set_input_tensorinfo", &NNFW_SESSION::set_input_tensorinfo, py::arg("index"),
         py::arg("tensor_info"),
         "Set input model's tensor info for resizing.\n"
         "Parameters:\n"
         "\tindex (int): Index of input to be set (0-indexed)\n"
         "\ttensor_info (tensorinfo): Tensor info to be set")
    .def("prepare", &NNFW_SESSION::prepare, "Prepare for inference")
    .def("run", &NNFW_SESSION::run, "Run inference")
    .def("run_async", &NNFW_SESSION::run_async, "Run inference asynchronously")
    .def("wait", &NNFW_SESSION::wait, "Wait for asynchronous run to finish")
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
         "\ttensorinfo: Tensor info (shape, type, etc)")
    .def("get_output", &NNFW_SESSION::get_output, py::arg("index"),
         R"pbdoc(
         Retrieve the internally-allocated dynamic output as a copy.
         Parameters:
             index (int): Index of the output tensor (0-indexed)
         Returns:
             numpy.ndarray: a copy of the internal buffer
         )pbdoc")
    .def("set_prepare_config", &NNFW_SESSION::set_prepare_config, py::arg("config"),
         "Set configuration to prepare");
}

// Bind the `NNFW_SESSION` class with experimental APIs
void bind_experimental_nnfw_session(py::module_ &m)
{
  // Add experimental APIs for the `NNFW_SESSION` class
  m.attr("nnfw_session")
    .cast<py::class_<NNFW_SESSION>>()
    .def("train_get_traininfo", &NNFW_SESSION::train_get_traininfo,
         "Retrieve training information for the model.")
    .def("train_set_traininfo", &NNFW_SESSION::train_set_traininfo, py::arg("info"),
         "Set training information for the model.")
    .def("train_prepare", &NNFW_SESSION::train_prepare, "Prepare for training")
    .def("train", &NNFW_SESSION::train, py::arg("update_weights") = true,
         "Run a training step, optionally updating weights.")
    .def("train_get_loss", &NNFW_SESSION::train_get_loss, py::arg("index"),
         "Retrieve the training loss for a specific index.")
    .def("train_set_input", &NNFW_SESSION::train_set_input<float>, py::arg("index"),
         py::arg("buffer"), "Set training input tensor for the given index (float).")
    .def("train_set_input", &NNFW_SESSION::train_set_input<int>, py::arg("index"),
         py::arg("buffer"), "Set training input tensor for the given index (int).")
    .def("train_set_input", &NNFW_SESSION::train_set_input<uint8_t>, py::arg("index"),
         py::arg("buffer"), "Set training input tensor for the given index (uint8).")
    .def("train_set_expected", &NNFW_SESSION::train_set_expected<float>, py::arg("index"),
         py::arg("buffer"), "Set expected output tensor for the given index (float).")
    .def("train_set_expected", &NNFW_SESSION::train_set_expected<int>, py::arg("index"),
         py::arg("buffer"), "Set expected output tensor for the given index (int).")
    .def("train_set_expected", &NNFW_SESSION::train_set_expected<uint8_t>, py::arg("index"),
         py::arg("buffer"), "Set expected output tensor for the given index (uint8).")
    .def("train_set_output", &NNFW_SESSION::train_set_output<float>, py::arg("index"),
         py::arg("buffer"), "Set output tensor for the given index (float).")
    .def("train_set_output", &NNFW_SESSION::train_set_output<int>, py::arg("index"),
         py::arg("buffer"), "Set output tensor for the given index (int).")
    .def("train_set_output", &NNFW_SESSION::train_set_output<uint8_t>, py::arg("index"),
         py::arg("buffer"), "Set output tensor for the given index (uint8).")
    .def("train_export_circle", &NNFW_SESSION::train_export_circle, py::arg("path"),
         "Export the trained model to a circle file.")
    .def("train_import_checkpoint", &NNFW_SESSION::train_import_checkpoint, py::arg("path"),
         "Import a training checkpoint from a file.")
    .def("train_export_checkpoint", &NNFW_SESSION::train_export_checkpoint, py::arg("path"),
         "Export the training checkpoint to a file.");
}

} // namespace onert::api::python
