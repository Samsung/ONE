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

#include "trix_loader.h"

#include <libnpuhost.h>
#include <npubinfmt.h>
#include <typedef.h>

namespace onert
{
namespace trix_loader
{

/**
 * @brief A tvn metadata reader
 */
class TrixMetaReader
{
public:
  TrixMetaReader(const char *path) : _model_path(path) {}
  ~TrixMetaReader() { free(_meta); }

  /**
   * @throw runtime_error when path is wrong or metadata is not valid
   */
  void open();
  data_layout input_seg_layout(uint32_t n) const { return _meta->input_seg_layout[n]; }
  data_layout output_seg_layout(uint32_t n) const { return _meta->output_seg_layout[n]; }
  uint32_t input_seg_num() const { return _meta->input_seg_num; }
  uint32_t output_seg_num() const { return _meta->output_seg_num; }

private:
  const char *_model_path;
  npubin_meta *_meta = nullptr;
};

void TrixMetaReader::open()
{
  assert(_model_path);
  auto _meta = getNPUmodel_metadata(_model_path, false);
  if (_meta == nullptr)
  {
    throw std::runtime_error("Failed to get TRIV2 model metadata");
  }
  if (NPUBIN_VERSION(_meta->magiccode) != 3)
  {
    throw std::runtime_error("TRIV2 model metadata version mismatched.");
  }
}

class TrixLoader final : public TrixLoaderBase
{
public:
  explicit TrixLoader(std::unique_ptr<ir::Subgraphs> &subgs) : TrixLoaderBase(subgs) {}

protected:
  bool loadModel() override;
};

bool TrixLoader::loadModel()
{
  // No need to consider multiple subgraphs
  auto subg = std::make_unique<ir::Graph>();

  TrixMetaReader meta_reader(_model_path.c_str());
  meta_reader.open();
  auto in_num = meta_reader.input_seg_num();
  auto out_num = meta_reader.output_seg_num();
  auto total_num = in_num + out_num;

  std::vector<ir::OperandIndex> _tensor_to_operand;
  _tensor_to_operand.resize(total_num);
  for (uint32_t i = 0; i < total_num; ++i)
  {
  }
  return true;
}

std::unique_ptr<ir::Subgraphs> loadModel(const std::string &filename)
{
  auto subgraphs = std::make_unique<ir::Subgraphs>();
  TrixLoader loader(subgraphs);
  loader.loadFromFile(filename);
  return subgraphs;
}
} // namespace trix_loader
} // namespace onert
