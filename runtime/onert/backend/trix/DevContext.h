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

#include <libnpuhost.h>

namespace onert
{
namespace backend
{
namespace trix
{

class DevContext
{
public:
  DevContext()
  {
    auto device_count = getnumNPUdeviceByType(NPUCOND_TRIV2_CONN_SOCIP);
    // TODO: x64 platform has 3 cores. We do not support more that 2 cores for now.
    if (device_count > 2)
    {
      device_count = 2;
    }

    if (device_count <= 0)
    {
      throw std::runtime_error("Unable to find TRIX NPU device");
    }

    for (int i = 0; i < device_count; i++)
    {
      npudev_h h;
      if (getNPUdeviceByType(&h, NPUCOND_TRIV2_CONN_SOCIP, i) < 0)
      {
        throw std::runtime_error("Failed to get TRIX NPU device handle");
      }
      _dev_handles.push_back(h);
    }
  }

  ~DevContext()
  {
    for (auto h : _dev_handles)
    {
      if (h != nullptr)
      {
        unregisterNPUmodel_all(h);
        putNPUdevice(h);
      }
    }
  }

  npudev_h getDev(int i) { return _dev_handles[i]; }
  int getDevSize() { return _dev_handles.size(); }

  template <typename T> void setDataInfo(tensors_data_info *info, std::vector<T *> &tensors)
  {
    info->num_info = static_cast<uint32_t>(tensors.size());

    for (uint32_t idx = 0; idx < info->num_info; ++idx)
    {
      info->info[idx].layout = convertDataLayout(tensors[idx]->layout());
      info->info[idx].type = convertDataType(tensors[idx]->data_type());
    }
  }

  template <typename T>
  void setBuffer(generic_buffers *buf, std::vector<T *> &tensors, int batch_size, int batch_index)
  {
    buf->num_buffers = static_cast<uint32_t>(tensors.size());

    for (uint32_t idx = 0; idx < buf->num_buffers; ++idx)
    {
      buf->bufs[idx].size = static_cast<uint64_t>(tensors[idx]->total_size() / batch_size);
      buf->bufs[idx].addr = tensors[idx]->buffer() + (batch_index * buf->bufs[idx].size);
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
  // TODO Support multicore npu device
  std::vector<npudev_h> _dev_handles;
};

} // namespace trix
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRIX_DEV_CONTEXT_H__
