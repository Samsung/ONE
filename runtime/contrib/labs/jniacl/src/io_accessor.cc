/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

/*
 * Copyright (c) 2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "io_accessor.h"
#include <ostream>
#include <android/log.h>

bool InputAccessor::access_tensor(arm_compute::ITensor &tensor)
{
  // Subtract the mean value from each channel
  arm_compute::Window window;
  window.use_tensor_dimensions(tensor.info()->tensor_shape());

  execute_window_loop(window, [&](const arm_compute::Coordinates &id) {
    *reinterpret_cast<float *>(tensor.ptr_to_element(id)) = _test_input;
    _test_input += _inc ? 1.0 : 0.0;

    __android_log_print(ANDROID_LOG_DEBUG, "LOG_TAG", "Input %d, %d = %lf\r\n", id.y(), id.x(),
                        *reinterpret_cast<float *>(tensor.ptr_to_element(id)));
  });
  return true;
}

bool OutputAccessor::access_tensor(arm_compute::ITensor &tensor)
{
  // Subtract the mean value from each channel
  arm_compute::Window window;
  window.use_tensor_dimensions(tensor.info()->tensor_shape());

  execute_window_loop(window, [&](const arm_compute::Coordinates &id) {
    __android_log_print(ANDROID_LOG_DEBUG, "Output", "Input %d, %d = %lf\r\n", id.y(), id.x(),
                        *reinterpret_cast<float *>(tensor.ptr_to_element(id)));
  });
  return false; // end the network
}

bool WeightAccessor::access_tensor(arm_compute::ITensor &tensor)
{
  // Subtract the mean value from each channel
  arm_compute::Window window;
  window.use_tensor_dimensions(tensor.info()->tensor_shape());

  execute_window_loop(window, [&](const arm_compute::Coordinates &id) {
    *reinterpret_cast<float *>(tensor.ptr_to_element(id)) = _test_weight;
    _test_weight += _inc ? 1.0 : 0.0;
  });
  return true;
}

bool BiasAccessor::access_tensor(arm_compute::ITensor &tensor)
{
  // Subtract the mean value from each channel
  arm_compute::Window window;
  window.use_tensor_dimensions(tensor.info()->tensor_shape());

  execute_window_loop(window, [&](const arm_compute::Coordinates &id) {
    *reinterpret_cast<float *>(tensor.ptr_to_element(id)) = 0.0;
  });
  return true;
}
