/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ModelData.h"

#include <mio/circle/schema_generated.h>
#include <luci/Importer.h>

using namespace circle_resizer;

namespace
{
std::unique_ptr<luci::Module> load_module(const std::vector<uint8_t> &model_buffer)
{
  flatbuffers::Verifier verifier{model_buffer.data(), model_buffer.size()};
  if (!circle::VerifyModelBuffer(verifier))
  {
    throw std::runtime_error("Verification of the model failed");
  }

  const luci::GraphBuilderSource *source_ptr = &luci::GraphBuilderRegistry::get();
  luci::Importer importer(source_ptr);
  return importer.importModule(model_buffer.data(), model_buffer.size());
}
} // namespace

ModelData::ModelData(const std::vector<uint8_t> &buffer) { invalidate(buffer); }

void ModelData::invalidate(const std::vector<uint8_t> &buffer)
{
  _buffer = buffer;
  _module = load_module(buffer);
}

std::vector<uint8_t> &ModelData::buffer() { return _buffer; }

luci::Module *ModelData::module() { return _module.get(); }

ModelData::~ModelData() = default;
