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
    auto device_count = getNPUdeviceByType(NPUCOND_TRIV2_CONN_SOCIP);
    if (device_count <= 0)
    {
      throw std::runtime_error("Unable to find TRIV2 NPU device");
    }

    // Use NPU 0 device
    if (getNPUdeviceByType(&_dev_handle, NPUCOND_TRIV2_CONN_SOCIP, 0) < 0)
    {
      throw std : runtime_error("Failed to get TRIV2 NPU device handle");
    }
  }

  ~DevContext()
  {
    if (_dev_handle != nullptr)
    {
      unregisterNPUmodel_all(_dev_handle);
      putNPUdevice(_dev_handle);
    }
  }

  npudev_h getDev() { return _dev_handle; }

  void setDataInfo(const tensors_data_info *info, std::vector<IPortableTensor *> &tensors)
  {
    info->num_info = tensors.size();

    for (int idx = 0; idx < info->num_info; ++idx)
    {
      info->info[idx].layout = convertDataLayout(tensors[idx]->layout());
      info->info[idx].type = convertDataType(tensors[idx]->data_type());
    }
  }

  void setBuffer(const generic_buffers *buf, std::vector<IPortableTensor *> &tensors)
  {
    buf->num_buffers = tensors.size();

    for (int idx = 0; idx < buf->num_buffers; ++idx)
    {
      buf->bufs[idx].addr = tensors[idx]->buffer();
      buf->bufs[idx].size = tensors[idx]->total_size();
      buf->bufs[idx].type = BUFFER_MAPPED;
    }
  }

private:
  data_layout convertDataLayout(ir::Layout layout)
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

  data_type convertDataType(ir::DataType type)
  {
    switch (type)
    {
      case ir::DataType::FLOAT32:
        return DATA_TYPE_FLOAT32;
      case ir::DataType::INT32:
        return DATA_TYPE_INT32;
      case ir::DataType::UINT32:
        return DATA_TYPE_UINT32;
      case ir::DataType::QUANT_UINT8_ASYMM:
        return DATA_TYPE_QASYMM8;
      case ir::DataType::UINT8:
        return DATA_TYPE_UINT8;
      case ir::DataType::INT64:
        return DATA_TYPE_INT64;
      default:
        throw std::runtime_error("Not support data type");
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
