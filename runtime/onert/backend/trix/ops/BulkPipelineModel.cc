/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "BulkPipelineModel.h"

#include <iostream>
#include <cstring>
#include <algorithm>

namespace onert::backend::trix::ops
{

BulkPipelineModel::BulkPipelineModel(const std::string &model_path, int device_id,
                                     BufferOwnership ownership)
  : _model_path(model_path), _device_id(device_id), _ownership(ownership)
{
  // DO NOTHING
}

BulkPipelineModel::~BulkPipelineModel() { release(); }

bool BulkPipelineModel::initialize()
{
  if (_initialized.load())
  {
    return true;
  }

  if (!loadMetadata())
  {
    return false;
  }

  _initialized = true;
  return true;
}

bool BulkPipelineModel::prepare()
{
  if (_prepared.load())
  {
    return true;
  }

  try
  {
    if (_ownership == BufferOwnership::OWNER)
    {
      openDevice();
      allocateBuffers();
      fillBuffers();
      markBufferReady();
      registerModel();
    }

    _prepared = true;
    return true;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Failed to prepare model " << _model_path << ": " << e.what() << std::endl;
    release();
    return false;
  }
}

void BulkPipelineModel::release()
{
  if (!_prepared.load())
  {
    return;
  }

  // Cancel a asynchronous job
  if (_async_fill_future.valid())
  {
    _async_fill_future.wait();
  }

  if (_ownership == BufferOwnership::OWNER)
  {
    unregisterModel();
    closeDevice();
  }

  if (_fp)
  {
    fclose(_fp);
    _fp = nullptr;
  }

  _program_buffer.reset();
  _weight_buffer.reset();
  _meta.reset();
  _meta_size = 0;
  _model_id = 0;

  _prepared = false;
}

void BulkPipelineModel::run(const std::vector<const IPortableTensor *> &inputs,
                            std::vector<IPortableTensor *> &outputs)
{
  if (!_prepared.load())
  {
    throw std::runtime_error("Model is not prepared: " + _model_path);
  }

  if (!_meta)
  {
    throw std::runtime_error("Model metadata is not loaded: " + _model_path);
  }

  // Prepare input buffers
  input_buffers input;
  input.num_buffers = _meta->input_seg_num;
  for (uint32_t i = 0; i < input.num_buffers; i++)
  {
    uint32_t idx = _meta->input_seg_idx[i];
    input.bufs[i].addr = inputs[i]->buffer();
    input.bufs[i].type = BUFFER_MAPPED;
    input.bufs[i].size = _meta->segment_size[idx];
  }

  // Prepare output buffers
  output_buffers output;
  output.num_buffers = _meta->output_seg_num;
  for (uint32_t i = 0; i < output.num_buffers; i++)
  {
    uint32_t idx = _meta->output_seg_idx[i];
    output.bufs[i].addr = outputs[i]->buffer();
    output.bufs[i].type = BUFFER_MAPPED;
    output.bufs[i].size = _meta->segment_size[idx];
  }

  // Execute the model
  int ret = runNPU_model(_dev, _model_id, NPU_INFER_BLOCKING, &input, &output, nullptr, nullptr);
  if (ret < 0)
  {
    throw std::runtime_error("runNPU_model() failed for " + _model_path +
                             ", ret: " + std::to_string(ret));
  }
}

void BulkPipelineModel::shareBuffersFrom(const BulkPipelineModel &owner)
{
  if (_ownership == BufferOwnership::OWNER)
  {
    throw std::runtime_error("Cannot share buffers with owner model: " + _model_path);
  }

  if (!owner.isPrepared())
  {
    throw std::runtime_error("Owner model is not prepared: " + owner.modelPath());
  }

  // Sharing the buffers
  _program_buffer = owner._program_buffer;
  _weight_buffer = owner._weight_buffer;

  // Sharing the device and model id
  _dev = owner.device();
  _model_id = owner.modelId();
}

void BulkPipelineModel::setNextModel(std::shared_ptr<BulkPipelineModel> next)
{
  _next_model = next;
}

void BulkPipelineModel::waitForBufferReady()
{
  std::unique_lock<std::mutex> lock(_buffer_mutex);
  _buffer_cv.wait(lock, [this] { return _buffer_ready.load(); });
}

void BulkPipelineModel::markBufferReady()
{
  {
    std::lock_guard<std::mutex> lock(_buffer_mutex);
    _buffer_ready = true;
  }
  _buffer_cv.notify_all();
}

void BulkPipelineModel::startAsyncBufferFill()
{
  _buffer_ready = false;
  _async_fill_future = std::async(std::launch::async, [this] {
    try
    {
      fillBuffers();
      markBufferReady();
    }
    catch (const std::exception &e)
    {
      std::cerr << "Failed to fill buffers asynchronously: " << e.what() << std::endl;
    }
  });
}

bool BulkPipelineModel::loadMetadata()
{
  _fp = fopen(_model_path.c_str(), "rb");
  if (!_fp)
  {
    throw std::runtime_error("Failed to open model file: " + _model_path);
  }

  _meta = std::make_unique<npubin_meta>();
  if (fread(_meta.get(), NPUBIN_META_SIZE, 1, _fp) != 1)
  {
    throw std::runtime_error("Failed to read metadata from: " + _model_path);
  }

  _meta_size = _meta->extended_metasize ? sizeof(npubin_meta) + _meta->extended_metasize
                                        : NPUBIN_META_TOTAL_SIZE(_meta->magiccode);

  return true;
}

void BulkPipelineModel::allocateBuffers()
{
  if (_ownership != BufferOwnership::OWNER)
  {
    throw std::runtime_error("Not allowed to allocate buffers for non-owner model: " + _model_path);
  }

  if (!_meta)
  {
    throw std::runtime_error("Metadata not loaded for: " + _model_path);
  }

  _program_buffer =
    std::make_shared<BulkPipelineBuffer>(BulkPipelineBuffer::BufferType::DMABUF_CONT,
                                         static_cast<size_t>(_meta->program_size), _device_id);

  _weight_buffer =
    std::make_shared<BulkPipelineBuffer>(BulkPipelineBuffer::BufferType::DMABUF_IOMMU,
                                         static_cast<size_t>(_meta->weight_size), _device_id);

  _program_buffer->allocate();
  if (_meta->weight_size > 0)
  {
    _weight_buffer->allocate();
  }
}

void BulkPipelineModel::fillBuffers()
{
  if (!_fp || !_program_buffer || !_weight_buffer)
  {
    throw std::runtime_error("Buffers not properly initialized for: " + _model_path);
  }

  // Fill program buffer
  _program_buffer->fillFromFile(_fp, _meta_size);

  // Fill weight buffer
  if (_weight_buffer->size() > 0)
  {
    _weight_buffer->fillFromFile(_fp, _meta_size + _meta->program_size);
  }
}

void BulkPipelineModel::registerModel()
{
  if (!_dev || !_program_buffer || !_weight_buffer)
  {
    throw std::runtime_error("Device or buffers not ready for: " + _model_path);
  }

  generic_buffer modelfile;
  modelfile.type = BUFFER_FILE;
  modelfile.filepath = _model_path.c_str();
  modelfile.size = _meta->size;

  int ret = registerNPUmodel_ext(_dev, &modelfile, _program_buffer->getGenericBuffer(),
                                 _weight_buffer->getGenericBuffer(), &_model_id);
  if (ret < 0)
  {
    throw std::runtime_error("Failed to register model: " + _model_path +
                             ", ret: " + std::to_string(ret));
  }
}

void BulkPipelineModel::unregisterModel()
{
  if (_dev && _model_id > 0)
  {
    int ret = unregisterNPUmodel(_dev, _model_id);
    if (ret < 0)
    {
      std::cerr << "Failed to unregister model: " << _model_path << ", ret: " << ret << std::endl;
    }
    _model_id = 0;
  }
}

void BulkPipelineModel::openDevice()
{
  int ret = getNPUdeviceByType(&_dev, NPUCOND_TRIV24_CONN_SOCIP, _device_id);
  if (ret < 0)
  {
    throw std::runtime_error("Failed to open NPU device for: " + _model_path +
                             ", ret: " + std::to_string(ret));
  }
}

void BulkPipelineModel::closeDevice()
{
  if (_dev)
  {
    putNPUdevice(_dev);
    _dev = nullptr;
  }
}

} // namespace onert::backend::trix::ops
