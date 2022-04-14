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

#include "ir/Graph.h"
#include "ir/operation/Bulk.h"

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
  TrixMetaReader() = default;
  ~TrixMetaReader() { free(_meta); }

  void init(const char *path);
  data_layout input_seg_layout(uint32_t n) const { return _meta->input_seg_layout[n]; }
  data_layout output_seg_layout(uint32_t n) const { return _meta->output_seg_layout[n]; }
  uint32_t input_seg_num() const { return _meta->input_seg_num; }
  uint32_t output_seg_num() const { return _meta->output_seg_num; }

private:
  npubin_meta *_meta = nullptr;
};

void TrixMetaReader::init(const char *path)
{
  assert(path);
  _meta = getNPUmodel_metadata(path, false);
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
  void loadModel() override;

private:
  void loadSubgraphs();
  std::unique_ptr<ir::Graph> loadSubgraph();
  void loadOperands(ir::Graph &subg);
  void loadBulk(ir::Graph &subg);
  void loadOperationIO(ir::OperandIndexSequence &inputs, ir::OperandIndexSequence &outputs);
  ir::OperandIndex inputIdxToOperandIdx(uint32_t i) const;
  ir::OperandIndex outputIdxToOperandIdx(uint32_t i) const;

private:
  TrixMetaReader _meta;
};

ir::OperandIndex TrixLoader::inputIdxToOperandIdx(uint32_t i) const { return ir::OperandIndex(i); }
ir::OperandIndex TrixLoader::outputIdxToOperandIdx(uint32_t i) const
{
  return ir::OperandIndex(_meta.input_seg_num() + i);
}

void TrixLoader::loadOperationIO(ir::OperandIndexSequence &inputs,
                                 ir::OperandIndexSequence &outputs)
{
  for (uint32_t i = 0; i < _meta.input_seg_num(); ++i)
  {
    inputs.append(inputIdxToOperandIdx(i));
  }

  for (uint32_t i = 0; i < _meta.output_seg_num(); ++i)
  {
    outputs.append(outputIdxToOperandIdx(i));
  }
}

void TrixLoader::loadBulk(ir::Graph &subg)
{
  ir::operation::Bulk::Param param;
  param.binary_path = _model_path;

  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(inputs, outputs);

  std::unique_ptr<ir::operation::Bulk> bulk(new ir::operation::Bulk(inputs, outputs, param));
  subg.addOperation(std::move(bulk));
}

void TrixLoader::loadOperands(ir::Graph &subg)
{
  (void)subg;
  auto in_num = _meta.input_seg_num();
  for (uint32_t i = 0; i < in_num; ++i)
  {
    // TODO: create operand
    // const auto operand_index = subg.addOperand(shape, type_info);
    // ...
  }
  auto out_num = _meta.output_seg_num();
  for (uint32_t i = 0; i < out_num; ++i)
  {
    // TODO: create operand
    // const auto operand_index = subg.addOperand(shape, type_info);
    // ...
  }
}

std::unique_ptr<ir::Graph> TrixLoader::loadSubgraph()
{
  auto subg = std::make_unique<ir::Graph>();
  _meta.init(_model_path.c_str());

  // Load tensors
  loadOperands(*subg);

  // Set inputs
  for (uint32_t i = 0; i < _meta.input_seg_num(); ++i)
  {
    subg->addInput(inputIdxToOperandIdx(i), "tvn_input" + std::to_string(i));
  }
  // Set outputs
  for (uint32_t i = 0; i < _meta.output_seg_num(); ++i)
  {
    subg->addOutput(outputIdxToOperandIdx(i), "tvn_out" + std::to_string(i));
  }
  // Create operations
  loadBulk(*subg);

  // TODO: NHWC only supported at this moment.
  subg->setLayout(ir::Layout::NHWC);
  subg->verify();
  return subg;
}

void TrixLoader::loadSubgraphs()
{
  // one subgraph only
  auto subg = loadSubgraph();
  _subgraphs->push(ir::SubgraphIndex(0), std::move(subg));
}

void TrixLoader::loadModel()
{
  _meta.init(_model_path.c_str());
  loadSubgraphs();
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
