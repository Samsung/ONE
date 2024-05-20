/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "exporter/CircleExporter.h"

#include "exec/Execution.h"
#include "ir/train/TrainingInfo.h"
#include "MMappedFile.h"
#include "TrainInfoBuilder.h"

#include <fstream>
#include <iostream>

namespace onert
{
namespace exporter
{

CircleExporter::CircleExporter(const std::string &source, const std::string &path) : _path{path}
{
  std::ifstream src(source.c_str(), std::ios::binary);
  std::ofstream dst(path.c_str(), std::ios::binary);
  dst << src.rdbuf();

  _mmapfile = std::make_unique<MMappedFile>(path.c_str());
  if (!_mmapfile->ensure_mmap())
    throw std::runtime_error("Failed to ensure mmap file");

  // make sure the architecture is little endian before direct access to flatbuffers
  assert(FLATBUFFERS_LITTLEENDIAN);
}

CircleExporter::~CircleExporter() { finish(); }

void CircleExporter::updateWeight(const std::unique_ptr<exec::Execution> &exec)
{
  exec->iterateTrainableTensors(
    [&](const ir::OperandIndex &idx, const backend::train::ITrainableTensor *tensor) {
      auto model = ::circle::GetModel(_mmapfile->buf());
      if (!model)
        throw std::runtime_error("Failed to get model from circle");

      auto subgs = model->subgraphs();
      if (!subgs || subgs->size() != 1)
        throw std::runtime_error("Circle does not has valid subgraph or has multiple subgraphs");

      auto subg = subgs->Get(0); // Get 1st subgraph
      if (!idx.valid() || idx.value() >= subg->tensors()->size())
        throw std::runtime_error("Trainable tensor index is out of range");

      auto buf_idx = subg->tensors()->Get(idx.value())->buffer();
      const ::circle::Buffer *buffer = (*model->buffers())[buf_idx];
      if (!buffer || !buffer->data())
        throw std::runtime_error("Buffer for trainable tensors is invalid");

      const flatbuffers::Vector<uint8_t> *array = buffer->data();
      if (!array)
        throw std::runtime_error("Data for trainable tensor's buffer is invalid");

      auto org_buf_sz = array->size();
      if (org_buf_sz != tensor->total_size())
        throw std::runtime_error("Trained tensor buffer size does not match original tensor's one");

      uint8_t *org_buf = const_cast<uint8_t *>(array->Data());
      if (!org_buf)
        throw std::runtime_error("Data for trainable tensor's buffer is invalid");

      memcpy(const_cast<uint8_t *>(org_buf), tensor->buffer(), org_buf_sz);
    });
}

void CircleExporter::updateMetadata(const std::unique_ptr<ir::train::TrainingInfo> &training_info)
{
  const auto model = ::circle::GetModel(_mmapfile->buf());
  const auto metadata = model->metadata();
  const auto metabuffers = model->buffers();
  const flatbuffers::Vector<uint8_t> *train_data = nullptr;
  for (uint32_t i = 0; i < metadata->size(); ++i)
  {
    const auto meta = metadata->Get(i);
    const auto meta_name = meta->name();
    if (meta_name->str() == std::string{"CIRCLE_TRAINING"})
    {
      const uint32_t buf_idx = meta->buffer();
      const ::circle::Buffer *meta_buf = metabuffers->Get(buf_idx);
      train_data = meta_buf->data();
      break;
    }
  }

  TrainInfoBuilder builder(training_info);
  // if (builder.size() != train_data->size())
  // {
  //   std::ostringstream errMsg;
  //   errMsg << "TrainInfo buffer size(";
  //   errMsg << builder.size();
  //   errMsg << ") does not match original TrainInfo's one(";
  //   errMsg << train_data->size();
  //   errMsg << ").";
  //   // auto errMsg = std::string{"TrainInfo buffer size("} + std::string{builder.size()} +
  //   std::string{") does not match original TrainInfo's one("}, std::string{train_data->size()},
  //   std::string{")."}; throw std::runtime_error(errMsg.str());
  // }

  if (train_data)
  {
    const ::circle::ModelTraining *meta_traininfo = ::circle::GetModelTraining(train_data->Data());
    VERBOSE(CircleExporter) << "TrainInfo: " << meta_traininfo->version() << std::endl;
    memcpy(const_cast<uint8_t *>(train_data->Data()), builder.get(), train_data->size());
  }
}

void CircleExporter::finish()
{
  if (_mmapfile->sync() == false)
    throw std::runtime_error("Failed to sync Circle file");

  if (_mmapfile->close() == false)
    throw std::runtime_error("Failed to close Circle file");
}
} // namespace exporter
} // namespace onert
