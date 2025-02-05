/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

/*
 * Copyright (c) 2018-2020 ARM Limited.
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
#include "helpers.h"

#if defined(DATA_TYPE) && defined(AXIS) && defined(DEPTH) && defined(OUTPUT_DIM_Z)

/** Performs the OneHot operation along the chosen axis
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g.
 * -DDATA_TYPE=short
 * @note Axis should be given as a preprocessor argument using -DAXIS=axis. e.g. -DAXIS=1
 * @attention Output tensor depth should be given as a preprocessor argument using
 * -DOUTPUT_DIM_Z=size. e.g. -DOUTPUT_DIM_Z=16
 * @attention Input tensor depth should be given as a preprocessor argument using
 * -DINPUT_DIM_Z=size. e.g. -DINPUT_DIM_Z=16
 *
 *
 * @param[in]  indices_ptr                              Pointer to the source tensor. Supported data
 * types: S32
 * @param[in]  indices_stride_x                         Stride of the source tensor in X dimension
 * (in bytes)
 * @param[in]  indices_step_x                           indices_stride_x * number of elements along
 * X processed per work item (in bytes)
 * @param[in]  indices_stride_y                         Stride of the source tensor in Y dimension
 * (in bytes)
 * @param[in]  indices_step_y                           indices_stride_y * number of elements along
 * Y processed per work item (in bytes)
 * @param[in]  indices_stride_z                         Stride of the source tensor in Y dimension
 * (in bytes)
 * @param[in]  indices_step_z                           indices_stride_z * number of elements along
 * Z processed per work item (in bytes)
 * @param[in]  indices_offset_first_element_in_bytes    Offset of the first element in the source
 * tensor
 * @param[in]  on_value_ptr                             Pointer to the on_value vector. Supported
 * data types: U8/S8/U16/S16/F16/U32/S32/F32.
 * @param[in]  on_value_stride_x                        Stride of the on_value vector in X dimension
 * (in bytes)
 * @param[in]  on_value_step_x                          on_value_stride_x * number of elements along
 * X processed per work item (in bytes)
 * @param[in]  on_value_offset_first_element_in_bytes   Offset of the first element in the on_value
 * vector
 * @param[in]  off_value_ptr                            Pointer to the off_value vector. Supported
 * data types: Same as @p on_value.
 * @param[in]  off_value_stride_x                       Stride of the off_value vector in X
 * dimension (in bytes)
 * @param[in]  off_value_step_x                         off_value_stride_x * number of elements
 * along X processed per work item (in bytes)
 * @param[in]  off_value_offset_first_element_in_bytes  Offset of the first element in the off_value
 * vector
 * @param[out] output_ptr                               Pointer to the destination tensor. Supported
 * data types: same as @p on_value
 * @param[in]  output_stride_x                          Stride of the destination tensor in X
 * dimension (in bytes)
 * @param[in]  output_step_x                            output_stride_x * number of elements along X
 * processed per work item (in bytes)
 * @param[in]  output_stride_y                          Stride of the destination tensor in Y
 * dimension (in bytes)
 * @param[in]  output_step_y                            output_stride_y * number of elements along Y
 * processed per work item (in bytes)
 * @param[in]  output_stride_z                          Stride of the destination tensor in Z
 * dimension (in bytes)
 * @param[in]  output_step_z                            output_stride_z * number of elements along Z
 * processed per work item (in bytes)
 * @param[in]  output_stride_w                          Stride of the destination tensor in W
 * dimension (in bytes)
 * @param[in]  output_step_w                            output_stride_w * number of elements along W
 * processed per work item (in bytes)
 * @param[in]  output_offset_first_element_in_bytes     Offset of the first element in the
 * destination tensor
 */
__kernel void one_hot(TENSOR3D_DECLARATION(indices), VECTOR_DECLARATION(on_value),
                      VECTOR_DECLARATION(off_value), TENSOR4D_DECLARATION(output))
{
  const int px = get_global_id(0);
  const int py = get_global_id(1);
  const int pz = get_global_id(2) % OUTPUT_DIM_Z;
  const int pw = get_global_id(2) / OUTPUT_DIM_Z;

  const Tensor3D indices = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(indices);
  Tensor4D output = CONVERT_TO_TENSOR4D_STRUCT(output, OUTPUT_DIM_Z);

#if AXIS == 0
  const int index = *(__global const int *)tensor3D_offset(&indices, py, pz, pw);
  *(__global DATA_TYPE *)output.ptr = index == px ? *((__global const DATA_TYPE *)on_value_ptr)
                                                  : *((__global const DATA_TYPE *)off_value_ptr);
#elif AXIS == 1
  const uint index = *(__global const uint *)tensor3D_offset(&indices, px, pz, pw);
  *(__global DATA_TYPE *)output.ptr = index == py ? *((__global const DATA_TYPE *)on_value_ptr)
                                                  : *((__global const DATA_TYPE *)off_value_ptr);
#elif AXIS == 2
  const uint index = *(__global const uint *)tensor3D_offset(&indices, px, py, pw);
  *(__global DATA_TYPE *)output.ptr = index == pz ? *((__global const DATA_TYPE *)on_value_ptr)
                                                  : *((__global const DATA_TYPE *)off_value_ptr);
#elif AXIS == 3
  const uint index = *(__global const uint *)tensor3D_offset(&indices, px, py, pz);
  *(__global DATA_TYPE *)output.ptr = index == pw ? *((__global const DATA_TYPE *)on_value_ptr)
                                                  : *((__global const DATA_TYPE *)off_value_ptr);
#endif // AXIS
}

/** Performs the OneHot operation along the chosen axis as off_value being zero
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g.
 * -DDATA_TYPE=short
 * @note Axis should be given as a preprocessor argument using -DAXIS=axis. e.g. -DAXIS=1
 * @attention Output tensor depth should be given as a preprocessor argument using
 * -DOUTPUT_DIM_Z=size. e.g. -DOUTPUT_DIM_Z=16
 * @attention Input tensor depth should be given as a preprocessor argument using
 * -DINPUT_DIM_Z=size. e.g. -DINPUT_DIM_Z=16
 *
 *
 * @param[in]  indices_ptr                              Pointer to the source tensor. Supported data
 * types: S32
 * @param[in]  indices_stride_x                         Stride of the source tensor in X dimension
 * (in bytes)
 * @param[in]  indices_step_x                           indices_stride_x * number of elements along
 * X processed per work item (in bytes)
 * @param[in]  indices_stride_y                         Stride of the source tensor in Y dimension
 * (in bytes)
 * @param[in]  indices_step_y                           indices_stride_y * number of elements along
 * Y processed per work item (in bytes)
 * @param[in]  indices_stride_z                         Stride of the source tensor in Y dimension
 * (in bytes)
 * @param[in]  indices_step_z                           indices_stride_z * number of elements along
 * Z processed per work item (in bytes)
 * @param[in]  indices_offset_first_element_in_bytes    Offset of the first element in the source
 * tensor
 * @param[in]  on_value_ptr                             Pointer to the on_value vector. Supported
 * data types: U8/S8/U16/S16/F16/U32/S32/F32.
 * @param[in]  on_value_stride_x                        Stride of the on_value vector in X dimension
 * (in bytes)
 * @param[in]  on_value_step_x                          on_value_stride_x * number of elements along
 * X processed per work item (in bytes)
 * @param[in]  on_value_offset_first_element_in_bytes   Offset of the first element in the on_value
 * vector
 * @param[out] output_ptr                               Pointer to the destination tensor. Supported
 * data types: same as @p on_value
 * @param[in]  output_stride_x                          Stride of the destination tensor in X
 * dimension (in bytes)
 * @param[in]  output_step_x                            output_stride_x * number of elements along X
 * processed per work item (in bytes)
 * @param[in]  output_stride_y                          Stride of the destination tensor in Y
 * dimension (in bytes)
 * @param[in]  output_step_y                            output_stride_y * number of elements along Y
 * processed per work item (in bytes)
 * @param[in]  output_stride_z                          Stride of the destination tensor in Z
 * dimension (in bytes)
 * @param[in]  output_step_z                            output_stride_z * number of elements along Z
 * processed per work item (in bytes)
 * @param[in]  output_stride_w                          Stride of the destination tensor in W
 * dimension (in bytes)
 * @param[in]  output_step_w                            output_stride_w * number of elements along W
 * processed per work item (in bytes)
 * @param[in]  output_offset_first_element_in_bytes     Offset of the first element in the
 * destination tensor
 */
__kernel void one_hot_only_on_value(TENSOR3D_DECLARATION(indices), VECTOR_DECLARATION(on_value),
                                    TENSOR4D_DECLARATION(output))
{
  const int px = get_global_id(0);
  const int py = get_global_id(1);
  const int pz = get_global_id(2);

  const Tensor3D indices = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(indices);
  const Tensor4D output = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(output, OUTPUT_DIM_Z);

  const int index = *(__global const int *)tensor3D_offset(&indices, px, py, pz);

  if (index < 0 || index >= DEPTH)
    return;

#if AXIS == 0
  *(__global DATA_TYPE *)tensor4D_offset(&output, index, px, py, pz) =
    *((__global const DATA_TYPE *)on_value_ptr);
#elif AXIS == 1
  *(__global DATA_TYPE *)tensor4D_offset(&output, px, index, py, pz) =
    *((__global const DATA_TYPE *)on_value_ptr);
#elif AXIS == 2
  *(__global DATA_TYPE *)tensor4D_offset(&output, px, py, index, pz) =
    *((__global const DATA_TYPE *)on_value_ptr);
#elif AXIS == 3
  *(__global DATA_TYPE *)tensor4D_offset(&output, px, py, pz, index) =
    *((__global const DATA_TYPE *)on_value_ptr);
#endif // AXIS
}

#endif // defined(DATA_TYPE) && defined(AXIS) && defined(DEPTH) && defined(OUTPUT_DIM_Z)
