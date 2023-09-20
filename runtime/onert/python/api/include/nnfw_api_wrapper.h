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

#include "nnfw.h"

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct tensorinfo
{
  const char *dtype;
  int32_t rank;
  int32_t dims[NNFW_MAX_RANK];
};

/**
 * @brief     Handle errors with NNFW_STATUS in API functions.
 *
 * This only handles NNFW_STATUS errors.
 *
 * @param[in] status The status returned by API functions
 */
void ensure_status(NNFW_STATUS status);

NNFW_LAYOUT getLayout(const char *layout = "");

NNFW_TYPE getType(const char *type = "");

const char *getStringType(NNFW_TYPE type);

/**
 * @brief     Get the total number of elements in nnfw_tensorinfo->dims.
 *
 * This function is called to set the size of the input, output array.
 *
 * @param[in] tensor_info Tensor info (shape, type, etc)
 */
uint64_t num_elems(const nnfw_tensorinfo *tensor_info);

/**
 * @brief     Get nnfw_tensorinfo->dims.
 *
 * This function is called to get dimension array of tensorinfo.
 *
 * @param[in] tensor_info Tensor info (shape, type, etc)
 */
py::list get_dims(const tensorinfo &tensor_info);

/**
 * @brief     Set nnfw_tensorinfo->dims.
 *
 * This function is called to set dimension array of tensorinfo.
 *
 * @param[in] tensor_info Tensor info (shape, type, etc)
 * @param[in] array       array to set dimension
 */
void set_dims(tensorinfo &tensor_info, const py::list &array);

class NNFW_SESSION
{
private:
  nnfw_session *session;

public:
  NNFW_SESSION(const char *package_file_path, const char *backends);
  NNFW_SESSION(const char *package_file_path, const char *op, const char *backend);
  ~NNFW_SESSION();

  void close_session();
  void set_input_tensorinfo(uint32_t index, const tensorinfo *tensor_info);
  void run();
  void run_async();
  void await();
  /**
   * @brief   process input array according to data type of numpy array sent by Python
   *          (int, float, uint8_t, bool, int64_t, int8_t, int16_t)
   */
  template <typename T> void set_input(uint32_t index, py::array_t<T> &buffer)
  {
    nnfw_tensorinfo tensor_info;
    nnfw_input_tensorinfo(this->session, index, &tensor_info);
    NNFW_TYPE type = tensor_info.dtype;
    uint32_t input_elements = num_elems(&tensor_info);
    size_t length = sizeof(T) * input_elements;

    ensure_status(nnfw_set_input(session, index, type, buffer.request().ptr, length));
  }
  /**
   * @brief   process output array according to data type of numpy array sent by Python
   *          (int, float, uint8_t, bool, int64_t, int8_t, int16_t)
   */
  template <typename T> void set_output(uint32_t index, py::array_t<T> &buffer)
  {
    nnfw_tensorinfo tensor_info;
    nnfw_output_tensorinfo(this->session, index, &tensor_info);
    NNFW_TYPE type = tensor_info.dtype;
    uint32_t input_elements = num_elems(&tensor_info);
    size_t length = sizeof(T) * input_elements;

    ensure_status(nnfw_set_output(session, index, type, buffer.request().ptr, length));
  }
  uint32_t input_size();
  uint32_t output_size();
  void set_input_layout(uint32_t index,
                        const char *layout); // process the input layout by receiving a string from
                                             // Python instead of NNFW_LAYOUT
  void set_output_layout(uint32_t index,
                         const char *layout); // process the output layout by receiving a string
                                              // from Python instead of NNFW_LAYOUT
  tensorinfo input_tensorinfo(uint32_t index);
  tensorinfo output_tensorinfo(uint32_t index);
  uint32_t query_info_u32(NNFW_INFO_ID id);
};
