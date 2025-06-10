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

#include "BatchThreadPool.h"

#include <libnpuhost.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace onert::backend::trix
{

using ModelID = uint32_t;

/**
 * @brief NPU device context of trix backend
 *
 */
class DevContext
{
public:
  /**
   * @brief Construct a new device Context object
   *
   */
  DevContext();

  /**
   * @brief Destroy the device Context object
   *
   */
  ~DevContext();

  DevContext(const DevContext &) = delete;
  DevContext &operator=(const DevContext &) = delete;

  /**
   * @brief Register a trix model for all NPU devices
   *
   * @param model_file_path File path of a trix model
   * @return ModelID Internal ID of the trix model
   */
  ModelID registerModel(const std::string &model_file_path);

  /**
   * @brief Unregister a trix model
   *
   * @param model_id Internal ID of the trix model to be unregistered
   */
  void unRegisterModel(ModelID model_id);

  /**
   * @brief Request a trix model to be run on NPU
   *
   * @param model_id    Internal ID of a trix model
   * @param input_bufs  Buffer data of inputs
   * @param in_info     Data info of inputs
   * @param output_bufs Buffer data of outputs
   * @param out_info    data info of outputs
   * @param batch_size  Batch size
   */
  void requestRun(ModelID model_id, input_buffers *input_bufs, tensors_data_info *in_info,
                  output_buffers *output_bufs, tensors_data_info *out_info, size_t batch_size);

private:
  /**
   * @brief Rquest one batch of a trix model to be run on a device of NPU
   *
   * @param dev_num     Device number
   * @param model_id    Internal ID of a trix model
   * @param input_bufs  Buffer data of inputs
   * @param in_info     Data info of inputs
   * @param output_bufs Buffer data of outputs
   * @param out_info    data info of outputs
   */
  void runOneBatch(uint32_t dev_num, ModelID model_id, input_buffers *input_bufs,
                   tensors_data_info *in_info, output_buffers *output_bufs,
                   tensors_data_info *out_info);

  /**
   * @brief Set the buffer object by batch
   *
   * @param origin_buf   Buffer object that has all batches
   * @param batch_num    Batch number
   * @param batch_offset Size of a batch
   * @param batch_buf    One batch buffer object to be set
   */
  void setBufferByBatch(const generic_buffer &origin_buf, uint32_t batch_num, uint64_t batch_offset,
                        generic_buffer *batch_buf);

private:
  /**
   * @brief NPU device handles
   *
   */
  std::vector<npudev_h> _dev_handles;

  /**
   * @brief Threadpool for batch-by-batch multi-threading
   *
   */
  std::unique_ptr<BatchThreadPool> _batch_thread_pool;

  // TODO Change key to internal trix model context(?) if it is needed
  /**
   * @brief Map for ID of models
   *        Internal Model ID : Model ID array for each device
   *
   */
  std::unordered_map<ModelID, std::vector<uint32_t>> _model_ids;

  /**
   * @brief Map for meta data
   *        Model ID at each device : meta data
   *
   */
  std::unordered_map<uint32_t, std::shared_ptr<npubin_meta>> _meta_map;

  /**
   * @brief Exception pointer captured whthin threads
   *
   */
  std::exception_ptr _eptr;
};

} // namespace onert::backend::trix

#endif // __ONERT_BACKEND_TRIX_DEV_CONTEXT_H__
