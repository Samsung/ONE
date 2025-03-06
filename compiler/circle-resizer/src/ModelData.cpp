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
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>

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
class BufferModelContract : public luci::CircleExporter::Contract
{
public:
  BufferModelContract(luci::Module *module)
    : _module(module), _buffer{std::make_unique<std::vector<uint8_t>>()}
  {
  }

  luci::Module *module() const override { return _module; }

  bool store(const char *ptr, const size_t size) const override
  {
    _buffer->resize(size);
    std::copy(ptr, ptr + size, _buffer->begin());
    return true;
  }

  std::vector<uint8_t> get_buffer() { return *_buffer; }

private:
  luci::Module *_module;
  std::unique_ptr<std::vector<uint8_t>> _buffer;
};

} // namespace

ModelData::ModelData(const std::vector<uint8_t> &buffer)
  : _buffer{buffer}, _module{load_module(buffer)}
{
}

void ModelData::invalidate_module() { _module_invalidated = true; }

void ModelData::invalidate_buffer() { _buffer_invalidated = true; }

std::vector<uint8_t> &ModelData::buffer()
{
  if (_buffer_invalidated)
  {
    luci::CircleExporter exporter;
    BufferModelContract contract(module());

    if (!exporter.invoke(&contract))
    {
      throw std::runtime_error("Exporting buffer from the model failed");
    }
    _buffer = contract.get_buffer();
    _buffer_invalidated = false;
  }
  return _buffer;
}

luci::Module *ModelData::module()
{
  if (_module_invalidated)
  {
    _module = load_module(_buffer);
    _module_invalidated = false;
  }
  return _module.get();
}

ModelData::~ModelData() = default;
