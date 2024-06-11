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

#include "circle_schema_generated.h"

#include <fstream>
#include <iostream>
#include <mutex>

namespace onert
{
namespace exporter
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
    src.read(&_data[0], _data.size());
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

void CircleExporter::finish()
{
  flatbuffers::FlatBufferBuilder builder(1024);
  builder.Finish(::circle::Model::Pack(builder, _model.get()), ::circle::ModelIdentifier());

  std::ofstream dst(_path.c_str(), std::ios::binary);
  dst.write(reinterpret_cast<const char *>(builder.GetBufferPointer()), builder.GetSize());
  dst.close();
}
} // namespace exporter
} // namespace onert
