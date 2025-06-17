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
#include "nnfw_exceptions.h"

#include <iostream>

namespace onert::api::python
{

namespace py = pybind11;

void ensure_status(NNFW_STATUS status)
{
  switch (status)
  {
    case NNFW_STATUS::NNFW_STATUS_NO_ERROR:
      return;
    case NNFW_STATUS::NNFW_STATUS_ERROR:
      throw NnfwError("NNFW_STATUS_ERROR");
    case NNFW_STATUS::NNFW_STATUS_UNEXPECTED_NULL:
      throw NnfwUnexpectedNullError("NNFW_STATUS_UNEXPECTED_NULL");
    case NNFW_STATUS::NNFW_STATUS_INVALID_STATE:
      throw NnfwInvalidStateError("NNFW_STATUS_INVALID_STATE");
    case NNFW_STATUS::NNFW_STATUS_OUT_OF_MEMORY:
      throw NnfwOutOfMemoryError("NNFW_STATUS_OUT_OF_MEMORY");
    case NNFW_STATUS::NNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE:
      throw NnfwInsufficientOutputError("NNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE");
    case NNFW_STATUS::NNFW_STATUS_DEPRECATED_API:
      throw NnfwDeprecatedApiError("NNFW_STATUS_DEPRECATED_API");
    default:
      throw NnfwError("NNFW_UNKNOWN_ERROR");
  }
}

NNFW_LAYOUT getLayout(const char *layout)
{
  if (std::strcmp(layout, "NCHW") == 0)
    return NNFW_LAYOUT::NNFW_LAYOUT_CHANNELS_FIRST;
  else if (std::strcmp(layout, "NHWC") == 0)
    return NNFW_LAYOUT::NNFW_LAYOUT_CHANNELS_LAST;
  else if (std::strcmp(layout, "NONE") == 0)
    return NNFW_LAYOUT::NNFW_LAYOUT_NONE;
  else
    throw NnfwError(std::string("Unknown layout type: '") + layout + "'");
}

NNFW_TYPE getType(const char *type)
{
  if (std::strcmp(type, "float32") == 0)
    return NNFW_TYPE::NNFW_TYPE_TENSOR_FLOAT32;
  else if (std::strcmp(type, "int32") == 0)
    return NNFW_TYPE::NNFW_TYPE_TENSOR_INT32;
  else if (std::strcmp(type, "bool") == 0)
    return NNFW_TYPE::NNFW_TYPE_TENSOR_UINT8;
  else if (std::strcmp(type, "bool") == 0)
    return NNFW_TYPE::NNFW_TYPE_TENSOR_BOOL;
  else if (std::strcmp(type, "int64") == 0)
    return NNFW_TYPE::NNFW_TYPE_TENSOR_INT64;
  else if (std::strcmp(type, "int8") == 0)
    return NNFW_TYPE::NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED;
  else if (std::strcmp(type, "int16") == 0)
    return NNFW_TYPE::NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED;
  else
    throw NnfwError(std::string("Cannot convert string to NNFW_TYPE: '") + type + "'");
}

const char *getStringType(NNFW_TYPE type)
{
  switch (type)
  {
    case NNFW_TYPE::NNFW_TYPE_TENSOR_FLOAT32:
      return "float32";
    case NNFW_TYPE::NNFW_TYPE_TENSOR_INT32:
      return "int32";
    case NNFW_TYPE::NNFW_TYPE_TENSOR_QUANT8_ASYMM:
    case NNFW_TYPE::NNFW_TYPE_TENSOR_UINT8:
      return "uint8";
    case NNFW_TYPE::NNFW_TYPE_TENSOR_BOOL:
      return "bool";
    case NNFW_TYPE::NNFW_TYPE_TENSOR_INT64:
      return "int64";
    case NNFW_TYPE::NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED:
      return "int8";
    case NNFW_TYPE::NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED:
      return "int16";
    default:
      throw NnfwError(std::string("Cannot convert NNFW_TYPE enum to string (value=") +
                      std::to_string(static_cast<int>(type)) + ")");
  }
}

uint64_t num_elems(const nnfw_tensorinfo *tensor_info)
{
  uint64_t n = 1;
  for (int32_t i = 0; i < tensor_info->rank; ++i)
  {
    n *= tensor_info->dims[i];
  }
  return n;
}

py::list get_dims(const tensorinfo &tensor_info)
{
  py::list dims_list;
  for (int32_t i = 0; i < tensor_info.rank; ++i)
  {
    dims_list.append(tensor_info.dims[i]);
  }
  return dims_list;
}

void set_dims(tensorinfo &tensor_info, const py::list &array)
{
  tensor_info.rank = py::len(array);
  for (int32_t i = 0; i < tensor_info.rank; ++i)
  {
    tensor_info.dims[i] = py::cast<int32_t>(array[i]);
  }
}

NNFW_SESSION::NNFW_SESSION(const char *package_file_path, const char *backends)
{
  this->session = nullptr;
  ensure_status(nnfw_create_session(&(this->session)));
  ensure_status(nnfw_load_model_from_file(this->session, package_file_path));
  ensure_status(nnfw_set_available_backends(this->session, backends));
}
NNFW_SESSION::~NNFW_SESSION()
{
  if (session)
  {
    close_session();
  }
}

void NNFW_SESSION::close_session()
{
  ensure_status(nnfw_close_session(this->session));
  this->session = nullptr;
}
void NNFW_SESSION::set_input_tensorinfo(uint32_t index, const tensorinfo *tensor_info)
{
  nnfw_tensorinfo ti;
  ti.dtype = getType(tensor_info->dtype);
  ti.rank = tensor_info->rank;
  for (int i = 0; i < NNFW_MAX_RANK; i++)
  {
    ti.dims[i] = tensor_info->dims[i];
  }
  ensure_status(nnfw_set_input_tensorinfo(session, index, &ti));
}
void NNFW_SESSION::prepare() { ensure_status(nnfw_prepare(session)); }
void NNFW_SESSION::run() { ensure_status(nnfw_run(session)); }
void NNFW_SESSION::run_async() { ensure_status(nnfw_run_async(session)); }
void NNFW_SESSION::wait() { ensure_status(nnfw_await(session)); }
uint32_t NNFW_SESSION::input_size()
{
  uint32_t number;
  NNFW_STATUS status = nnfw_input_size(session, &number);
  ensure_status(status);
  return number;
}
uint32_t NNFW_SESSION::output_size()
{
  uint32_t number;
  NNFW_STATUS status = nnfw_output_size(session, &number);
  ensure_status(status);
  return number;
}
void NNFW_SESSION::set_input_layout(uint32_t index, const char *layout)
{
  NNFW_LAYOUT nnfw_layout = getLayout(layout);
  ensure_status(nnfw_set_input_layout(session, index, nnfw_layout));
}
tensorinfo NNFW_SESSION::input_tensorinfo(uint32_t index)
{
  nnfw_tensorinfo tensor_info = nnfw_tensorinfo();
  ensure_status(nnfw_input_tensorinfo(session, index, &tensor_info));
  tensorinfo ti;
  ti.dtype = getStringType(tensor_info.dtype);
  ti.rank = tensor_info.rank;
  for (int i = 0; i < NNFW_MAX_RANK; i++)
  {
    ti.dims[i] = tensor_info.dims[i];
  }
  return ti;
}
tensorinfo NNFW_SESSION::output_tensorinfo(uint32_t index)
{
  nnfw_tensorinfo tensor_info = nnfw_tensorinfo();
  ensure_status(nnfw_output_tensorinfo(session, index, &tensor_info));
  tensorinfo ti;
  ti.dtype = getStringType(tensor_info.dtype);
  ti.rank = tensor_info.rank;
  for (int i = 0; i < NNFW_MAX_RANK; i++)
  {
    ti.dims[i] = tensor_info.dims[i];
  }
  return ti;
}

//////////////////////////////////////////////
// Internal APIs
//////////////////////////////////////////////
py::array NNFW_SESSION::get_output(uint32_t index)
{
  // First call into the C API
  nnfw_tensorinfo out_info = {};
  const void *out_buffer = nullptr;
  ensure_status(nnfw_get_output(session, index, &out_info, &out_buffer));

  // Convert nnfw_tensorinfo to our python-visible struct
  size_t num_elements = 1;
  std::vector<ssize_t> shape;
  shape.reserve(out_info.rank);
  for (int i = 0; i < out_info.rank; ++i)
  {
    shape.push_back(static_cast<ssize_t>(out_info.dims[i]));
    num_elements *= static_cast<size_t>(out_info.dims[i]);
  }

  // Wrap the raw buffer in a numpy array;
  auto np = py::module_::import("numpy");
  py::dtype dt = np.attr("dtype")(py::str(getStringType(out_info.dtype))).cast<py::dtype>();
  size_t itemsize = dt.attr("itemsize").cast<size_t>();

  py::array arr(dt, shape);
  std::memcpy(arr.mutable_data(), out_buffer, num_elements * itemsize);
  arr.attr("flags").attr("writeable") = false;

  return arr;
}

//////////////////////////////////////////////
// Experimental APIs for inference
//////////////////////////////////////////////
void NNFW_SESSION::set_prepare_config(NNFW_PREPARE_CONFIG config)
{
  ensure_status(nnfw_set_prepare_config(session, config, "true"));
}

//////////////////////////////////////////////
// Experimental APIs for training
//////////////////////////////////////////////
nnfw_train_info NNFW_SESSION::train_get_traininfo()
{
  nnfw_train_info train_info = nnfw_train_info();
  ensure_status(nnfw_train_get_traininfo(session, &train_info));
  return train_info;
}

void NNFW_SESSION::train_set_traininfo(const nnfw_train_info *info)
{
  ensure_status(nnfw_train_set_traininfo(session, info));
}

void NNFW_SESSION::train_prepare() { ensure_status(nnfw_train_prepare(session)); }

void NNFW_SESSION::train(bool update_weights)
{
  ensure_status(nnfw_train(session, update_weights));
}

float NNFW_SESSION::train_get_loss(uint32_t index)
{
  float loss = 0.f;
  ensure_status(nnfw_train_get_loss(session, index, &loss));
  return loss;
}

void NNFW_SESSION::train_export_circle(const py::str &path)
{
  const char *c_str_path = path.cast<std::string>().c_str();
  ensure_status(nnfw_train_export_circle(session, c_str_path));
}

void NNFW_SESSION::train_import_checkpoint(const py::str &path)
{
  const char *c_str_path = path.cast<std::string>().c_str();
  ensure_status(nnfw_train_import_checkpoint(session, c_str_path));
}

void NNFW_SESSION::train_export_checkpoint(const py::str &path)
{
  const char *c_str_path = path.cast<std::string>().c_str();
  ensure_status(nnfw_train_export_checkpoint(session, c_str_path));
}

} // namespace onert::api::python
