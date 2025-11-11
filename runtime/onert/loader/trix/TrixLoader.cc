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

#include "ir/Graph.h"
#include "ir/operation/Bulk.h"
#include "loader/ILoader.h"

#include <libnpuhost.h>
#include <npubinfmt.h>
#include <typedef.h>

namespace onert::trix_loader
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
  data_type input_seg_quant_type(uint32_t n) const { return _meta->input_seg_quant_type[n]; }
  data_type output_seg_quant_type(uint32_t n) const { return _meta->output_seg_quant_type[n]; }
  float input_seg_quant_scale(uint32_t n) const { return _meta->input_seg_quant_s[n]; }
  float output_seg_quant_scale(uint32_t n) const { return _meta->output_seg_quant_s[n]; }
  int32_t input_seg_quant_zp(uint32_t n) { return _meta->input_seg_quant_z[n]; }
  int32_t output_seg_quant_zp(uint32_t n) { return _meta->output_seg_quant_z[n]; }
  uint32_t input_seg_num() const { return _meta->input_seg_num; }
  uint32_t output_seg_num() const { return _meta->output_seg_num; }
  uint32_t input_seg_dims(uint32_t n, uint32_t axis) const
  {
    return _meta->input_seg_dims[n][axis];
  }
  uint32_t output_seg_dims(uint32_t n, uint32_t axis) const
  {
    return _meta->output_seg_dims[n][axis];
  }

private:
  npubin_meta *_meta = nullptr;
};

void TrixMetaReader::init(const char *path)
{
  assert(path);
  _meta = getNPUmodel_metadata(path, false);
  if (_meta == nullptr)
  {
    throw std::runtime_error("Failed to get TRIX model metadata");
  }
  if (NPUBIN_VERSION(_meta->magiccode) != 3)
  {
    throw std::runtime_error("TRIX model metadata version mismatched.");
  }
}

class TrixLoader : public onert::loader::ILoader
{
public:
  /**
   * @brief Construct a new Loader object
   */
  TrixLoader() = default;

  /**
   * @brief Load a model from file
   * @param file_path
   */
  std::unique_ptr<ir::Model> loadFromFile(const std::string &file_path) override;

private:
  /*
   * @brief Load actually
   * @throw runtime_error when tvn path is wrong or tvn is invalid
   */
  void loadModel(std::unique_ptr<ir::Model> &model);
  std::unique_ptr<ir::Graph> loadSubgraph();
  void loadBulk(ir::Graph &subg);
  ir::OperandIndex inputIdxToOperandIdx(uint32_t i) const;
  ir::OperandIndex outputIdxToOperandIdx(uint32_t i) const;
  ir::DataType toDataType(const data_type type) const;
  void loadInputOperands(TrixMetaReader &meta, ir::Graph &subg);
  void loadOutputOperands(TrixMetaReader &meta, ir::Graph &subg);

private:
  /** path to model (e.g. tvn) */
  std::vector<std::string> _model_path;
  /** original IO shapes */
  std::vector<ir::Shape> _origin_input_shapes;
  std::vector<ir::Shape> _origin_output_shapes;
  TrixMetaReader _meta;
};

ir::DataType TrixLoader::toDataType(const data_type type) const
{
  switch (type)
  {
    case DATA_TYPE_QASYMM8:
      return ir::DataType::QUANT_UINT8_ASYMM;
    case DATA_TYPE_QSYMM16:
      return ir::DataType::QUANT_INT16_SYMM;
    default:
      throw std::runtime_error("Unsupported data type from trix model");
  }
}

ir::OperandIndex TrixLoader::inputIdxToOperandIdx(uint32_t i) const { return ir::OperandIndex(i); }
ir::OperandIndex TrixLoader::outputIdxToOperandIdx(uint32_t i) const
{
  return ir::OperandIndex(_meta.input_seg_num() + i);
}

void TrixLoader::loadBulk(ir::Graph &subg)
{
  ir::operation::Bulk::Param param;
  param.binary_path = _model_path;
  param.origin_input_shapes = _origin_input_shapes;
  param.origin_output_shapes = _origin_output_shapes;

  std::unique_ptr<ir::operation::Bulk> bulk(
    new ir::operation::Bulk(subg.getInputs(), subg.getOutputs(), param));
  subg.addOperation(std::move(bulk));
}

void TrixLoader::loadInputOperands(TrixMetaReader &meta, ir::Graph &subg)
{
  for (uint32_t i = 0; i < meta.input_seg_num(); ++i)
  {
    // Shape
    ir::Shape shape;
    for (uint32_t d = 0; d < MAX_RANK; ++d)
      shape.append(meta.input_seg_dims(i, d));

    // TypeInfo
    ir::TypeInfo type_info(toDataType(meta.input_seg_quant_type(i)), meta.input_seg_quant_scale(i),
                           meta.input_seg_quant_zp(i));

    _origin_input_shapes.push_back(shape);
    // Create operand
    subg.addOperand(shape, type_info);
    subg.addInput(ir::OperandIndex(i), "tvn_input" + std::to_string(i));
  }
}

void TrixLoader::loadOutputOperands(TrixMetaReader &meta, ir::Graph &subg)
{
  for (uint32_t i = 0; i < meta.output_seg_num(); ++i)
  {
    // Shape
    ir::Shape shape;
    for (uint32_t d = 0; d < MAX_RANK; ++d)
      shape.append(meta.output_seg_dims(i, d));

    // TypeInfo
    ir::TypeInfo type_info(toDataType(meta.output_seg_quant_type(i)),
                           meta.output_seg_quant_scale(i), meta.output_seg_quant_zp(i));

    _origin_output_shapes.push_back(shape);
    // Create operand
    subg.addOperand(shape, type_info);
    subg.addOutput(ir::OperandIndex(subg.getInputs().size() + i), "tvn_out" + std::to_string(i));
  }
}

std::unique_ptr<ir::Graph> TrixLoader::loadSubgraph()
{
  auto subg = std::make_unique<ir::Graph>();

  if (_model_path.size() == 1)
  {
    // Single model
    TrixMetaReader meta;
    meta.init(_model_path.front().c_str());
    loadInputOperands(meta, *subg);
    loadOutputOperands(meta, *subg);
  }
  else
  {
    // Multiple models
    TrixMetaReader head_model_meta, tail_model_meta;
    head_model_meta.init(_model_path.front().c_str());
    tail_model_meta.init(_model_path.back().c_str());
    loadInputOperands(head_model_meta, *subg);
    loadOutputOperands(tail_model_meta, *subg);
  }

  // Create operations
  loadBulk(*subg);

  subg->verify();
  return subg;
}

void TrixLoader::loadModel(std::unique_ptr<ir::Model> &model)
{
  // one subgraph only
  auto subg = loadSubgraph();
  model->push(ir::SubgraphIndex(0), std::move(subg));
}

std::unique_ptr<ir::Model> TrixLoader::loadFromFile(const std::string &file_path)
{
  auto model = std::make_unique<ir::Model>();

  std::stringstream ss{file_path};
  std::string path;
  while (std::getline(ss, path, ';'))
  {
    // model path will be used to set Bulk param
    // , and it can be more than one for multiple models
    _model_path.push_back(path);
  }

  loadModel(model);

  return model;
}

} // namespace onert::trix_loader

extern "C" {

onert::loader::ILoader *onert_loader_create() { return new onert::trix_loader::TrixLoader; }

void onert_loader_destroy(onert::loader::ILoader *loader) { delete loader; }
}
