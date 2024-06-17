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

#include <iostream>

void ensure_status(NNFW_STATUS status)
{
  switch (status)
  {
    case NNFW_STATUS::NNFW_STATUS_NO_ERROR:
      break;
    case NNFW_STATUS::NNFW_STATUS_ERROR:
      std::cout << "[ERROR]\tNNFW_STATUS_ERROR\n";
      exit(1);
    case NNFW_STATUS::NNFW_STATUS_UNEXPECTED_NULL:
      std::cout << "[ERROR]\tNNFW_STATUS_UNEXPECTED_NULL\n";
      exit(1);
    case NNFW_STATUS::NNFW_STATUS_INVALID_STATE:
      std::cout << "[ERROR]\tNNFW_STATUS_INVALID_STATE\n";
      exit(1);
    case NNFW_STATUS::NNFW_STATUS_OUT_OF_MEMORY:
      std::cout << "[ERROR]\tNNFW_STATUS_OUT_OF_MEMORY\n";
      exit(1);
    case NNFW_STATUS::NNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE:
      std::cout << "[ERROR]\tNNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE\n";
      exit(1);
  }
}

NNFW_LAYOUT getLayout(const char *layout)
{
  if (!strcmp(layout, "NCHW"))
  {
    return NNFW_LAYOUT::NNFW_LAYOUT_CHANNELS_FIRST;
  }
  else if (!strcmp(layout, "NHWC"))
  {
    return NNFW_LAYOUT::NNFW_LAYOUT_CHANNELS_LAST;
  }
  else if (!strcmp(layout, "NONE"))
  {
    return NNFW_LAYOUT::NNFW_LAYOUT_NONE;
  }
  else
  {
    std::cout << "[ERROR]\tLAYOUT_TYPE\n";
    exit(1);
  }
}

NNFW_TYPE getType(const char *type)
{
  if (!strcmp(type, "float32"))
  {
    return NNFW_TYPE::NNFW_TYPE_TENSOR_FLOAT32;
  }
  else if (!strcmp(type, "int32"))
  {
    return NNFW_TYPE::NNFW_TYPE_TENSOR_INT32;
  }
  else if (!strcmp(type, "uint8"))
  {
    return NNFW_TYPE::NNFW_TYPE_TENSOR_UINT8;
    // return NNFW_TYPE::NNFW_TYPE_TENSOR_QUANT8_ASYMM;
  }
  else if (!strcmp(type, "bool"))
  {
    return NNFW_TYPE::NNFW_TYPE_TENSOR_BOOL;
  }
  else if (!strcmp(type, "int64"))
  {
    return NNFW_TYPE::NNFW_TYPE_TENSOR_INT64;
  }
  else if (!strcmp(type, "int8"))
  {
    return NNFW_TYPE::NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED;
  }
  else if (!strcmp(type, "int16"))
  {
    return NNFW_TYPE::NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED;
  }
  else
  {
    std::cout << "[ERROR] String to NNFW_TYPE Failure\n";
    exit(1);
  }
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
      std::cout << "[ERROR] NNFW_TYPE to String Failure\n";
      exit(1);
  }
}

uint64_t num_elems(const nnfw_tensorinfo *tensor_info)
{
  uint64_t n = 1;
  for (uint32_t i = 0; i < tensor_info->rank; ++i)
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
  ensure_status(nnfw_prepare(this->session));
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
void NNFW_SESSION::run() { ensure_status(nnfw_run(session)); }
void NNFW_SESSION::run_async() { ensure_status(nnfw_run_async(session)); }
void NNFW_SESSION::await() { ensure_status(nnfw_await(session)); }
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
void NNFW_SESSION::set_output_layout(uint32_t index, const char *layout)
{
  NNFW_LAYOUT nnfw_layout = getLayout(layout);
  ensure_status(nnfw_set_output_layout(session, index, nnfw_layout));
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
