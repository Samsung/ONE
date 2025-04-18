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

#include "CircleModel.h"

#include <mio/circle/schema_generated.h>

#include <luci/Importer.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>

#include <fstream>

using namespace circle_resizer;

namespace
{

std::vector<uint8_t> read_model(const std::string &model_path)
{
  std::ifstream file_stream(model_path, std::ios::in | std::ios::binary | std::ifstream::ate);
  if (!file_stream.is_open())
  {
    throw std::runtime_error("Failed to open file: " + model_path);
  }

  std::streamsize size = file_stream.tellg();
  file_stream.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  if (!file_stream.read(reinterpret_cast<char *>(buffer.data()), size))
  {
    throw std::runtime_error("Failed to read file: " + model_path);
  }

  return buffer;
}

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
    assert(_module); // FIX_CALLER_UNLESS
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
  std::unique_ptr<std::vector<uint8_t>> _buffer; // note that the store method has to be const
};

template <typename NodeType>
std::vector<Shape> extract_shapes(const std::vector<loco::Node *> &nodes)
{
  std::vector<Shape> shapes;
  for (const auto &loco_node : nodes)
  {
    std::vector<Dim> dims;
    const auto circle_node = loco::must_cast<const NodeType *>(loco_node);
    for (uint32_t dim_idx = 0; dim_idx < circle_node->rank(); dim_idx++)
    {
      if (circle_node->dim(dim_idx).known())
      {
        const int32_t dim_val = circle_node->dim(dim_idx).value();
        dims.push_back(Dim{dim_val});
      }
      else
      {
        dims.push_back(Dim{-1});
      }
    }
    shapes.push_back(Shape{dims});
  }
  return shapes;
}

} // namespace

CircleModel::CircleModel(const std::vector<uint8_t> &buffer) : _module{load_module(buffer)} {}

CircleModel::CircleModel(const std::string &model_path) : CircleModel(read_model(model_path)) {}

luci::Module *CircleModel::module() { return _module.get(); }

void CircleModel::save(std::ostream &stream)
{
  BufferModelContract contract(module());
  luci::CircleExporter exporter;
  if (!exporter.invoke(&contract))
  {
    throw std::runtime_error("Exporting buffer from the model failed");
  }

  auto model_buffer = contract.get_buffer();
  stream.write(reinterpret_cast<const char *>(model_buffer.data()), model_buffer.size());
  if (!stream.good())
  {
    throw std::runtime_error("Failed to write to output stream");
  }
}

void CircleModel::save(const std::string &output_path)
{
  std::ofstream out_stream(output_path, std::ios::out | std::ios::binary);
  save(out_stream);
}

std::vector<Shape> CircleModel::input_shapes() const
{
  return extract_shapes<luci::CircleInput>(loco::input_nodes(_module->graph()));
}

std::vector<Shape> CircleModel::output_shapes() const
{
  return extract_shapes<luci::CircleOutput>(loco::output_nodes(_module->graph()));
}

CircleModel::~CircleModel() = default;
