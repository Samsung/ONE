/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRIX_CONVERT_H__
#define __ONERT_BACKEND_TRIX_CONVERT_H__

#include <backend/IPortableTensor.h>
#include <ir/DataType.h>
#include <ir/Layout.h>
#include <libnpuhost.h>

#include <type_traits>

namespace onert::backend::trix
{

/**
 * @brief Convert type of data from onert type to npu type
 *
 * @param type Data type in onert
 * @return data_type Data type in npu
 */
data_type convertDataType(const ir::DataType type);

/**
 * @brief Set the tensors_data_info object
 *
 * @tparam T Type of tensor based of IPortableTensor
 * @param tensors Tensors that have data information
 * @param info    tensors_data_info to be set
 */
template <typename T, std::enable_if_t<std::is_base_of_v<IPortableTensor, T>, bool> = true>
void setDataInfo(const std::vector<T *> &tensors, tensors_data_info *info)
{
  info->num_info = static_cast<uint32_t>(tensors.size());

  for (uint32_t idx = 0; idx < info->num_info; ++idx)
  {
    info->info[idx].layout = DATA_LAYOUT_NHWC;
    info->info[idx].type = convertDataType(tensors[idx]->data_type());
  }
}

/**
 * @brief Set the generic_buffers object
 *
 * @tparam T Type of tensor based of IPortableTensor
 * @param tensors Tensors that have buffer information
 * @param buf     generic_buffers to be set
 */
template <typename T, std::enable_if_t<std::is_base_of_v<IPortableTensor, T>, bool> = true>
void setBuffers(const std::vector<T *> &tensors, generic_buffers *buf)
{
  buf->num_buffers = static_cast<uint32_t>(tensors.size());

  for (uint32_t idx = 0; idx < buf->num_buffers; ++idx)
  {
    buf->bufs[idx].addr = tensors[idx]->buffer();
    buf->bufs[idx].size = static_cast<uint64_t>(tensors[idx]->total_size());
    buf->bufs[idx].type = BUFFER_MAPPED;
  }
}

} // namespace onert::backend::trix

#endif // __ONERT_BACKEND_TRIX_CONVERT_H__
