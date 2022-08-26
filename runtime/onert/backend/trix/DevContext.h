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

#ifndef __ONERT_BACKEND_TRIX_DEV_CONTEXT_H__
#define __ONERT_BACKEND_TRIX_DEV_CONTEXT_H__

#include <ir/Layout.h>
#include <ir/DataType.h>

#include <libnpuhost.h>
#include <stdexcept>
#include <vector>

namespace onert
{
namespace backend
{
namespace trix
{

class DevContext
{
public:
  DevContext(int32_t batch_num);

  ~DevContext()
  {
    if (_dev_handle != nullptr)
    {
      unregisterNPUmodel_all(_dev_handle);
      putNPUdevice(_dev_handle);
    }
  }

  npudev_h getDev() const { return _dev_handle; }

  template <typename T> void setDataInfo(tensors_data_info *info, std::vector<T *> &tensors)
  {
    info->num_info = static_cast<uint32_t>(tensors.size());

    for (uint32_t idx = 0; idx < info->num_info; ++idx)
    {
      info->info[idx].layout = convertDataLayout(tensors[idx]->layout());
      info->info[idx].type = convertDataType(tensors[idx]->data_type());
    }
  }

  template <typename T> void setBuffer(generic_buffers *buf, std::vector<T *> &tensors)
  {
    buf->num_buffers = static_cast<uint32_t>(tensors.size());

    for (uint32_t idx = 0; idx < buf->num_buffers; ++idx)
    {
      buf->bufs[idx].addr = tensors[idx]->buffer();
      buf->bufs[idx].size = static_cast<uint64_t>(tensors[idx]->total_size());
      buf->bufs[idx].type = BUFFER_MAPPED;
    }
  }

private:
  data_layout convertDataLayout(const ir::Layout layout)
  {
    switch (layout)
    {
      case ir::Layout::NCHW:
        return DATA_LAYOUT_NCHW;
      case ir::Layout::NHWC:
        return DATA_LAYOUT_NHWC;
      default:
        throw std::runtime_error("Unknown Layout");
    }
  }

  data_type convertDataType(const ir::DataType type)
  {
    switch (type)
    {
      case ir::DataType::QUANT_UINT8_ASYMM:
        return DATA_TYPE_QASYMM8;
      case ir::DataType::QUANT_INT16_SYMM:
        return DATA_TYPE_QSYMM16;
      default:
        throw std::runtime_error("Unsupported data type");
    }
  }

private:
  // NPU device handle
  npudev_h _dev_handle;
};

} // namespace trix
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRIX_DEV_CONTEXT_H__
