/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_TRAIN_CIRCLE_EXPORTER_H__
#define __ONERT_TRAIN_CIRCLE_EXPORTER_H__

#include "traininfobuilder.h"
#include "circle_schema_generated.h"

#include <string>
#include <fstream>
#include <iostream>
#include <mutex>

namespace onert_train
{

class CircleExporter
{
public:
  CircleExporter(const std::string &source, const std::string &path)
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
      src.read(&_data[0], _data.size());
      src.close();
    }

    const auto model = ::circle::GetModel(_data.data());
    if (!model)
      throw std::runtime_error("Failed to load original circle file");
    _model = model->UnPack();
  }

  ~CircleExporter() { finish(); }

  void updateMetadata(const nnfw_train_info &traininfo)
  {
    const char *const TRAININFO_METADATA_NAME = "CIRCLE_TRAINING";

    TrainInfoBuilder tbuilder(traininfo);
    std::mutex mutex;

    bool found = false;
    for (const auto &meta : _model->metadata)
    {
      if (meta->name == std::string{TRAININFO_METADATA_NAME})
      {
        const uint32_t buf_idx = meta->buffer;
        auto &buffer = _model->buffers.at(buf_idx);
        memcpy(&buffer->data[0], tbuilder.get(), tbuilder.size());
        found = true;
        break;
      }
    }

    if (!found)
    {
      std::lock_guard<std::mutex> guard(mutex);
      auto buffer = std::make_unique<::circle::BufferT>();
      buffer->size = tbuilder.size();
      buffer->data.resize(buffer->size);
      memcpy(&buffer->data[0], tbuilder.get(), buffer->size);

      auto meta = std::make_unique<::circle::MetadataT>();
      meta->name = std::string{TRAININFO_METADATA_NAME};
      meta->buffer = _model->buffers.size();

      _model->buffers.push_back(std::move(buffer));
      _model->metadata.push_back(std::move(meta));
    }
  }

private:
  void finish()
  {
    flatbuffers::FlatBufferBuilder builder(1024);
    builder.Finish(::circle::Model::Pack(builder, _model), ::circle::ModelIdentifier());

    std::ofstream dst(_path.c_str(), std::ios::binary);
    dst.write((const char *)builder.GetBufferPointer(), builder.GetSize());
    dst.close();
  }

private:
  std::string _path;
  std::string _data;
  ::circle::ModelT *_model;
};

} // namespace onert_train

#endif // __ONERT_TRAIN_CIRCLE_EXPORTER_H__
