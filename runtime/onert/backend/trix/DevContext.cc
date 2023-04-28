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

#include "DevContext.h"

#include "Convert.h"

#include <stdexcept>

namespace onert
{
namespace backend
{
namespace trix
{

// All things related to npu device handle are gathered this Class, but when implementing npu
// deamon, others except the context roles should be seperated.
DevContext::DevContext() : _dev_handles{}, _model_ids{}, _meta_map{}
{
  auto dev_count = getnumNPUdeviceByType(NPUCOND_TRIV2_CONN_SOCIP);
  if (dev_count <= 0)
  {
    throw std::runtime_error("Unable to find TRIX NPU device");
  }

  // Get NPU device handles
  for (int i = 0; i < dev_count; ++i)
  {
    npudev_h handle;
    if (getNPUdeviceByType(&handle, NPUCOND_TRIV2_CONN_SOCIP, i) < 0)
    {
      throw std::runtime_error("Failed to get TRIX NPU device handle");
    }
    _dev_handles.emplace_back(handle);
  }

  // NOTE Do not change the number of threads as long as jobs in thread call
  //      the synchronous APIs such as submitNPU_request()
  _batch_thread_pool = std::make_unique<BatchThreadPool>(_dev_handles.size());
  // We need to careful not to create multiple `BatchThreadPool`. In case of multiple models, there
  // may be a problem having multiple `BatchThreadPool` in current implementation. But if this
  // creating thread pool is moved to npu deamon, I think this problem will be solved smoothly.
}

DevContext::~DevContext()
{
  // NOTE Must release _batch_thread_pool before releasing _dev_handles to wait for all threads to
  //      be terminated
  _batch_thread_pool.reset(nullptr);

  for (const auto &dev_handle : _dev_handles)
  {
    unregisterNPUmodel_all(dev_handle);
    putNPUdevice(dev_handle);
  }
}

ModelID DevContext::registerModel(const std::string &model_file_path)
{
  if (_dev_handles.size() == 0)
  {
    throw std::runtime_error("No npu device is available");
  }

  auto meta = getNPUmodel_metadata(model_file_path.c_str(), false);

  if (meta == nullptr)
  {
    throw std::runtime_error("Unable to extract the model metadata");
  }

  generic_buffer file_info;
  file_info.type = BUFFER_FILE;
  file_info.filepath = model_file_path.c_str();
  file_info.size = meta->size;

  ModelID model_id;

  for (uint32_t dev_num = 0; dev_num < _dev_handles.size(); ++dev_num)
  {
    // Register model for each device
    uint32_t model_id_at_device;
    if (registerNPUmodel(_dev_handles.at(dev_num), &file_info, &model_id_at_device) < 0)
    {
      if (meta != nullptr)
        free(meta);
      throw std::runtime_error("Failed to register npu model");
    }

    if (dev_num == 0)
    {
      model_id = model_id_at_device;
      _meta_map[model_id_at_device] = std::shared_ptr<npubin_meta>(meta);
    }
    else
    {
      _meta_map[model_id_at_device] = _meta_map[model_id];
      meta = nullptr;
    }

    _model_ids[model_id].resize(dev_num + 1);
    _model_ids[model_id].at(dev_num) = model_id_at_device;
  }

  // Return the model id for device 0 only
  return model_id;
}

void DevContext::unRegisterModel(ModelID model_id)
{
  for (uint32_t dev_num = 0; dev_num < _dev_handles.size(); ++dev_num)
  {
    const auto model_id_at_device = _model_ids.at(model_id).at(dev_num);
    const auto &dev_handle = _dev_handles.at(dev_num);

    // Remove meta data
    _meta_map.erase(model_id_at_device);

    // Unregister Model for each device
    unregisterNPUmodel(dev_handle, model_id_at_device);
  }
  // Remove model IDs
  _model_ids.erase(model_id);
}

void DevContext::requestRun(ModelID model_id, input_buffers *input_bufs, tensors_data_info *in_info,
                            output_buffers *output_bufs, tensors_data_info *out_info,
                            size_t batch_size)
{
  if (batch_size > 1)
  {
    if (in_info->num_info != 1)
    {
      throw std::runtime_error("Supported only an input that has batch now");
    }
    if (out_info->num_info != 1)
    {
      throw std::runtime_error("Supported only one output now");
    }

    if (input_bufs->bufs[0].size % batch_size != 0)
    {
      throw std::runtime_error("Invalid batch size. batch size :" + std::to_string(batch_size) +
                               ", input buffer size : " + std::to_string(input_bufs->bufs[0].size));
    }

    if (output_bufs->bufs[0].size % batch_size != 0)
    {
      throw std::runtime_error(
        "Invalid batch size. batch size :" + std::to_string(batch_size) +
        ", output tensor size : " + std::to_string(output_bufs->bufs[0].size));
    }

    // inputs/outputs for each batch
    std::vector<input_buffers> in_buffers_vec(batch_size);
    std::vector<output_buffers> out_buffers_vec(batch_size);

    // Run on thread pool
    std::vector<std::future<int32_t>> batch_futures;
    for (uint32_t batch_num = 0; batch_num < batch_size; ++batch_num)
    {
      // Enqueue jobs
      // The in_info and out_info are always the same even if they are divided by batch, so they are
      // used as they are.
      auto future = _batch_thread_pool->enqueueJob(
        [batch_size, in_info, out_info,
         this](uint32_t dev_num, ModelID model_id, const input_buffers *input_bufs,
               const output_buffers *output_bufs, uint32_t batch_num) -> int32_t {
          // Set buffers of inputs/outputs for each batch
          // TODO Support multiple inputs/outputs
          input_buffers in_batch_buffers;
          in_batch_buffers.num_buffers = input_bufs->num_buffers;
          const uint64_t in_batch_offset = input_bufs->bufs[0].size / batch_size;
          setBufferByBatch(input_bufs->bufs[0], batch_num, in_batch_offset,
                           &in_batch_buffers.bufs[0]);

          output_buffers out_batch_buffers;
          out_batch_buffers.num_buffers = output_bufs->num_buffers;
          const uint64_t out_batch_offset = output_bufs->bufs[0].size / batch_size;
          setBufferByBatch(output_bufs->bufs[0], batch_num, out_batch_offset,
                           &out_batch_buffers.bufs[0]);

          try
          {
            // dev_num is the same as the thread number in _batch_thread_pool
            this->runOneBatch(dev_num, model_id, &in_batch_buffers, in_info, &out_batch_buffers,
                              out_info);
          }
          catch (...)
          {
            _eptr = std::current_exception();
          }

          return batch_num;
        },
        model_id, input_bufs, output_bufs, batch_num);
      batch_futures.emplace_back(std::move(future));
    }

    for (auto &&future : batch_futures)
    {
      future.get();
    }

    if (_eptr)
    {
      std::exception_ptr eptr(nullptr);
      _eptr.swap(eptr);
      std::rethrow_exception(eptr);
    }
  }
  else
  {
    runOneBatch(0, model_id, input_bufs, in_info, output_bufs, out_info);
  }
}

void DevContext::runOneBatch(uint32_t dev_num, ModelID model_id, input_buffers *input_bufs,
                             tensors_data_info *in_info, output_buffers *output_bufs,
                             tensors_data_info *out_info)
{
  const auto &model_id_at_device = _model_ids.at(model_id).at(dev_num);

  const auto meta = _meta_map.at(model_id_at_device);
  if (meta->input_seg_num != in_info->num_info)
  {
    throw std::runtime_error("The number of inputs does not match to model input seg num");
  }

  if (meta->output_seg_num != out_info->num_info)
  {
    throw std::runtime_error("The number of outputs does not match to model output seg num");
  }

  const auto &dev_handle = _dev_handles.at(dev_num);
  int req_id;

  if (auto error_code = createNPU_request(dev_handle, model_id_at_device, &req_id))
  {
    throw std::runtime_error("Unable to create NPU request with model id (" +
                             std::to_string(model_id_at_device) + ")" +
                             " error code : " + std::to_string(error_code));
  }

  if (auto error_code =
        setNPU_requestData(dev_handle, req_id, input_bufs, in_info, output_bufs, out_info))
  {
    removeNPU_request(dev_handle, req_id);
    throw std::runtime_error("Unable to create NPU request for model id (" +
                             std::to_string(model_id_at_device) + ")" +
                             " error code : " + std::to_string(error_code));
  }

  // NOTE submitNPU_request is not thread-safe(?). It is rarely hanging(unresponsive).
  //      Ultimately, to solve this problem, we have to either use other thread-safe API or
  //      change submitNPU_request to be thread-safe, but both works take time.
  //      As a workaround, let's allow hanging thread.
  // TODO Change submitNPU_request to be thread-safe or replaced with other thread-safe API
  std::packaged_task<int(npudev_h, int)> task(submitNPU_request);
  auto f = task.get_future();
  std::thread thread_submit_request(std::move(task), dev_handle, req_id);
  auto status = f.wait_until(std::chrono::system_clock::now() + std::chrono::seconds(60));
  if (status == std::future_status::timeout)
  {
    // There is no way to terminate hanging submitNPU_request from the outside.
    // If a hanging thread is detached, it will remain as a hanging thread. Even so, it's better
    // than having the main thread hanging.
    thread_submit_request.detach();

    // TODO Enable removeNPU_request after resolving hanging.
    // removeNPU_request(dev_handle, req_id);
    throw std::runtime_error("The npu API \"submitNPU_request\" timeout");
  }

  auto error_code = f.get();
  thread_submit_request.join();
  if (error_code != 0)
  {
    removeNPU_request(dev_handle, req_id);
    throw std::runtime_error("Unable to submit NPU request with req id (" + std::to_string(req_id) +
                             ")" + " error code : " + std::to_string(error_code));
  }

  if (auto error_code = removeNPU_request(dev_handle, req_id))
  {
    throw std::runtime_error("Unable to remove NPU request with req id (" + std::to_string(req_id) +
                             ")" + " error code : " + std::to_string(error_code));
  }
}

void DevContext::setBufferByBatch(const generic_buffer &origin_buf, uint32_t batch_num,
                                  uint64_t batch_offset, generic_buffer *batch_buf)
{
  batch_buf->addr = reinterpret_cast<uint8_t *>(origin_buf.addr) + batch_num * batch_offset;
  batch_buf->size = batch_offset;
  batch_buf->type = BUFFER_MAPPED;
}

} // namespace trix
} // namespace backend
} // namespace onert
