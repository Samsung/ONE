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

#include "TrainInfoBuilder.h"
#include "circle_schema_generated.h"
#include "exec/Execution.h"
#include "ir/train/TrainingInfo.h"
#include "loader/TrainInfoLoader.h"

#include <fstream>
#include <iostream>

namespace onert::exporter
{

CircleExporter::CircleExporter(const std::string &source, const std::string &path)
  : _path{path}, _data{}, _model{nullptr}
{
  // make sure the architecture is little endian before direct access to flatbuffers
  assert(FLATBUFFERS_LITTLEENDIAN);

  std::ifstream src(source.c_str(), std::ios::binary);
  if (src.is_open())
  {
    src.seekg(0, std::ios::end);
    _data.resize(src.tellg());
    src.seekg(0, std::ios::beg);
    src.read(&_data[0], static_cast<std::streamsize>(_data.size()));
    src.close();
  }

  if (_data.size() == 0)
    throw std::runtime_error("Invalid source file");

  const auto model = ::circle::GetModel(_data.data());
  if (!model)
    throw std::runtime_error("Failed to load original circle file");
  _model.reset(model->UnPack());
}

CircleExporter::~CircleExporter() { finish(); }

void CircleExporter::updateWeight(const std::unique_ptr<exec::Execution> &exec)
{
  exec->iterateTrainableTensors(
    [&](const ir::OperandIndex &idx, const backend::train::ITrainableTensor *tensor) {
      std::lock_guard<std::mutex> guard(_mutex);
      const auto &subgs = _model->subgraphs;
      if (subgs.size() != 1)
        throw std::runtime_error("Circle does not has valid subgraph or has multiple subgraphs");

      if (!idx.valid())
        throw std::runtime_error("Trainable tensor is invalid");

      uint32_t buf_idx = -1;
      const auto &subg = subgs.at(0); // Get 1st subgraph
      if (idx.value() >= subg->tensors.size())
      {
        auto buffer = std::make_unique<::circle::BufferT>();
        buffer->size = tensor->total_size();
        buffer->data.resize(buffer->size);

        buf_idx = _model->buffers.size();
        _model->buffers.push_back(std::move(buffer));
      }
      else
      {
        buf_idx = subg->tensors.at(idx.value())->buffer;
        if (buf_idx >= _model->buffers.size())
          throw std::runtime_error("Buffer for trainable tensors is invalid");
      }

      const auto &buffer = _model->buffers.at(buf_idx);

      auto org_buf_sz = buffer->data.size();
      if (org_buf_sz != tensor->total_size())
        throw std::runtime_error("Trained tensor buffer size does not match original tensor's one");

      memcpy(buffer->data.data(), tensor->buffer(), org_buf_sz);
    });
}

void CircleExporter::updateMetadata(const std::unique_ptr<ir::train::TrainingInfo> &training_info)
{
  TrainInfoBuilder tbuilder(training_info);
  bool found = false;
  for (const auto &meta : _model->metadata)
  {
    if (meta->name == std::string{loader::TRAININFO_METADATA_NAME})
    {
      std::lock_guard<std::mutex> guard(_mutex);
      const uint32_t buf_idx = meta->buffer;
      auto &buffer = _model->buffers.at(buf_idx);

      if (tbuilder.size() != buffer->data.size())
      {
        buffer->data.resize(tbuilder.size());
        buffer->size = tbuilder.size();
      }

      memcpy(buffer->data.data(), tbuilder.get(), tbuilder.size());
      found = true;
      break;
    }
  }

  if (!found)
  {
    std::lock_guard<std::mutex> guard(_mutex);
    auto buffer = std::make_unique<::circle::BufferT>();
    buffer->size = tbuilder.size();
    buffer->data.resize(buffer->size);
    memcpy(buffer->data.data(), tbuilder.get(), buffer->size);

    auto meta = std::make_unique<::circle::MetadataT>();
    meta->name = std::string{loader::TRAININFO_METADATA_NAME};
    meta->buffer = _model->buffers.size();

    _model->buffers.push_back(std::move(buffer));
    _model->metadata.push_back(std::move(meta));
  }
}

void CircleExporter::finish()
{
  flatbuffers::FlatBufferBuilder builder(1024);
  builder.Finish(::circle::Model::Pack(builder, _model.get()), ::circle::ModelIdentifier());

  std::ofstream dst(_path.c_str(), std::ios::binary);
  dst.write(reinterpret_cast<const char *>(builder.GetBufferPointer()),
            static_cast<std::streamsize>(builder.GetSize()));
  dst.close();
}
} // namespace onert::exporter
