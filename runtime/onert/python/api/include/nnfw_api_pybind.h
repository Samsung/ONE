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

#include <stl.h>
#include <stl_bind.h>
#include <numpy.h>

namespace py = pybind11;

/**
 * @brief     Handle erros with NNFW_STATUS in API functions.
 *
 * This only handles NNFW_STATUS errors.
 *
 * @param[in] status The status returned by API functions
 */
void ensure_status(NNFW_STATUS status);

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
py::list get_dims(const nnfw_tensorinfo &tensor_info);

/**
 * @brief     Set nnfw_tensorinfo->dims.
 *
 * This function is called to set dimension array of tensorinfo.
 *
 * @param[in] tensor_info Tensor info (shape, type, etc)
 * @param[in] array       array to set dimension
 */
void set_dims(nnfw_tensorinfo &tensor_info, const py::list &array);

class NNFW_SESSION
{
private:
  nnfw_session *session;

public:
  NNFW_SESSION();
  ~NNFW_SESSION();

  void create_session();
  void close_session();
  void load_model_from_file(const char *package_file_path);
  void apply_tensorinfo(uint32_t index, nnfw_tensorinfo tensor_info);
  void set_input_tensorinfo(uint32_t index, const nnfw_tensorinfo *tensor_info);
  void prepare();
  void run();
  void run_async();
  void await();
  /**
   * @brief   process input array according to data type of numpy array sent by Python
   *          (int, float, uint8_t, bool, int64_t, int8_t, int16_t)
   */
  template <typename T>
  void set_input(uint32_t index, nnfw_tensorinfo *tensor_info, py::array_t<T> &buffer);
  /**
   * @brief   process output array according to data type of numpy array sent by Python
   *          (int, float, uint8_t, bool, int64_t, int8_t, int16_t)
   */
  template <typename T>
  void set_output(uint32_t index, nnfw_tensorinfo *tensor_info, py::array_t<T> &buffer);
  uint32_t input_size();
  uint32_t output_size();
  void set_input_layout(uint32_t index,
                        const char *layout); // process the input layout by receiving a string from
                                             // Python instead of NNFW_LAYOUT
  void set_output_layout(uint32_t index,
                         const char *layout); // process the output layout by receiving a string
                                              // from Python instead of NNFW_LAYOUT
  nnfw_tensorinfo input_tensorinfo(uint32_t index);
  nnfw_tensorinfo output_tensorinfo(uint32_t index);
  void set_available_backends(const char *backends);
  void set_op_backend(const char *op, const char *backend);
  uint32_t query_info_u32(NNFW_INFO_ID id);
};
