/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <cassert>
#include <iostream>
#include <map>
#include <memory>

#include "CircleModel.h"
#include "DataLookup.h"

#include <mio_tflite2121/Helper.h>

namespace tflite2circle
{

template <> void Offset<MetaDataBufferLink>::build(const TFLFlatBufVec *tflite_flatbuffer_vec)
{
  if (tflite_flatbuffer_vec == nullptr)
    return;
  std::vector<int32_t> metadata_buffer_vec{tflite_flatbuffer_vec->begin(),
                                           tflite_flatbuffer_vec->end()};
  _circle_flatbuffer_vec_offset = _fb->CreateVector(metadata_buffer_vec);
}

template <> void Offset<BufferLink>::build(const TFLFlatBufVec *tflite_flatbuffer_vec)
{
  std::vector<flatbuffers::Offset<circle::Buffer>> buffers_vec;

  for (auto it : *tflite_flatbuffer_vec)
  {
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> buffer_data;
    if (it->data())
    {
      std::vector<uint8_t> data_vec{it->data()->begin(), it->data()->end()};
      buffer_data = _fb->CreateVector(data_vec);
    }
    circle::BufferBuilder circle_buffer_builder{*_fb};
    circle_buffer_builder.add_data(buffer_data);
    auto circle_buffers = circle_buffer_builder.Finish();
    buffers_vec.emplace_back(circle_buffers);
  }
  _circle_flatbuffer_vec_offset = _fb->CreateVector(buffers_vec);
}

template <> void Offset<SubGraphLink>::build(const TFLFlatBufVec *tflite_flatbuffer_vec)
{
  std::vector<flatbuffers::Offset<circle::SubGraph>> subgprahs_vec;

  int32_t subgraph_index = 0;

  for (auto it_sg : *tflite_flatbuffer_vec)
  {
    // tensors of subgraph
    std::vector<flatbuffers::Offset<circle::Tensor>> tensor_vec;

    auto tflite_tensors = it_sg->tensors();
    for (auto it : *tflite_tensors)
    {
      // shape
      flatbuffers::Offset<flatbuffers::Vector<int32_t>> shape;
      if (it->shape())
      {
        auto shape_vec = std::vector<int32_t>({it->shape()->begin(), it->shape()->end()});
        shape = _fb->CreateVector(shape_vec);
      }
      // name
      flatbuffers::Offset<flatbuffers::String> name;
      if (it->name())
        name = _fb->CreateString(it->name()->str());
      // quantization
      flatbuffers::Offset<circle::QuantizationParameters> quantization;
      if (it->quantization())
      {
        std::vector<float> tfmin;
        std::vector<float> tfmax;
        std::vector<float> tfscale;
        std::vector<int64_t> tfzerop;
        flatbuffers::Offset<flatbuffers::Vector<float>> min;
        flatbuffers::Offset<flatbuffers::Vector<float>> max;
        flatbuffers::Offset<flatbuffers::Vector<float>> scale;
        flatbuffers::Offset<flatbuffers::Vector<int64_t>> zero_point;
        int32_t quantized_dimension = it->quantization()->quantized_dimension();

        if (it->quantization()->min() && it->quantization()->max())
        {
          auto rmin = it->quantization()->min();
          auto rmax = it->quantization()->max();
          tfmin = std::vector<float>{rmin->begin(), rmin->end()};
          tfmax = std::vector<float>{rmax->begin(), rmax->end()};
          min = _fb->CreateVector(tfmin);
          max = _fb->CreateVector(tfmax);
        }

        if (it->quantization()->scale() && it->quantization()->zero_point())
        {
          auto rs = it->quantization()->scale();
          auto rz = it->quantization()->zero_point();
          tfscale = std::vector<float>{rs->begin(), rs->end()};
          tfzerop = std::vector<int64_t>{rz->begin(), rz->end()};
          scale = _fb->CreateVector(tfscale);
          zero_point = _fb->CreateVector(tfzerop);
        }

        quantization = circle::CreateQuantizationParameters(*_fb, min, max, scale, zero_point,
                                                            circle::QuantizationDetails_NONE, 0,
                                                            quantized_dimension);
      }
      // is_variable
      bool is_variable = it->is_variable();

      flatbuffers::Offset<circle::SparsityParameters> sparsity;
      // sparsity
      if (it->sparsity())
      {
        flatbuffers::Offset<flatbuffers::Vector<int32_t>> traversal_order;
        flatbuffers::Offset<flatbuffers::Vector<int32_t>> block_map;
        flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<circle::DimensionMetadata>>>
          dim_metadata;

        // traversal_order
        if (it->sparsity()->traversal_order())
        {
          auto traversal_order_vec = std::vector<int32_t>{
            it->sparsity()->traversal_order()->begin(), it->sparsity()->traversal_order()->end()};
          traversal_order = _fb->CreateVector(traversal_order_vec);
        }

        // block_map
        if (it->sparsity()->block_map())
        {
          auto block_map_vec = std::vector<int32_t>{it->sparsity()->block_map()->begin(),
                                                    it->sparsity()->block_map()->end()};
          block_map = _fb->CreateVector(block_map_vec);
        }

        // dim_metadata
        std::vector<flatbuffers::Offset<circle::DimensionMetadata>> dim_metadata_vec;
        auto tflite_dim_metadata = it->sparsity()->dim_metadata();
        for (auto it : *tflite_dim_metadata)
        {
          // array_segments
          auto tflite_array_segments_type = it->array_segments_type();
          auto circle_array_segments =
            get_circle_sparse_index_vector(*_fb, it->array_segments(), tflite_array_segments_type);
          auto circle_array_segments_type =
            get_circle_sparse_index_vector_type(tflite_array_segments_type);

          // array_indices
          auto tflite_array_indices_type = it->array_indices_type();
          auto circle_array_indices =
            get_circle_sparse_index_vector(*_fb, it->array_indices(), tflite_array_indices_type);
          auto circle_array_indices_type =
            get_circle_sparse_index_vector_type(tflite_array_indices_type);

          auto circle_dim_metadata_builder = circle::DimensionMetadataBuilder{*_fb};

          circle_dim_metadata_builder.add_format(get_circle_dimension_type(it->format()));
          circle_dim_metadata_builder.add_dense_size(it->dense_size());
          circle_dim_metadata_builder.add_array_segments(circle_array_segments);
          circle_dim_metadata_builder.add_array_segments_type(circle_array_segments_type);
          circle_dim_metadata_builder.add_array_indices(circle_array_indices);
          circle_dim_metadata_builder.add_array_indices_type(circle_array_indices_type);
          auto dim_metadata = circle_dim_metadata_builder.Finish();
          dim_metadata_vec.emplace_back(dim_metadata);
        }
        dim_metadata = _fb->CreateVector(dim_metadata_vec);

        sparsity = circle::CreateSparsityParameters(*_fb, traversal_order, block_map, dim_metadata);
      }

      // shape signature
      flatbuffers::Offset<flatbuffers::Vector<int32_t>> shape_signature;
      if (it->shape_signature())
      {
        auto shape_signature_vec =
          std::vector<int32_t>({it->shape_signature()->begin(), it->shape_signature()->end()});
        shape_signature = _fb->CreateVector(shape_signature_vec);
      }

      circle::TensorBuilder tensor_builder{*_fb};
      tensor_builder.add_shape(shape);
      tensor_builder.add_type(get_circle_tensortype(it->type()));
      tensor_builder.add_buffer(it->buffer());
      tensor_builder.add_name(name);
      tensor_builder.add_quantization(quantization);
      tensor_builder.add_is_variable(is_variable);
      tensor_builder.add_sparsity(sparsity);
      tensor_builder.add_shape_signature(shape_signature);
      auto tensor = tensor_builder.Finish();
      tensor_vec.emplace_back(tensor);
    }
    auto circle_tensors = _fb->CreateVector(tensor_vec);

    // inputs of subgraph
    auto tflite_inputs = it_sg->inputs();
    std::vector<int32_t> input_vec{tflite_inputs->begin(), tflite_inputs->end()};

    // apply signature_def to input tensor index so that input orders follow like tensorflow lite
    // interpreter._get_full_signature_list() method, which is ordered(sorted) in name
    // NOTE we do not need this when circle format supports signature_def
    if (_tfl_signature_def_offsets != nullptr)
    {
      for (auto it_signdef : *_tfl_signature_def_offsets)
      {
        if (it_signdef->subgraph_index() == subgraph_index)
        {
          auto inputs = it_signdef->inputs();
          assert(inputs->size() == input_vec.size());

          std::map<std::string, uint32_t> map_name_index;
          for (auto it_tm : *inputs)
          {
            map_name_index[it_tm->name()->str()] = it_tm->tensor_index();
          }
          uint32_t input_vec_idx = 0;
          for (auto &item : map_name_index)
          {
            input_vec[input_vec_idx++] = item.second;
          }
        }
      }
    }

    auto circle_inputs = _fb->CreateVector(input_vec);

    // outputs of subgraph
    auto tflite_outputs = it_sg->outputs();
    std::vector<int32_t> output_vec{tflite_outputs->begin(), tflite_outputs->end()};

    if (_tfl_signature_def_offsets != nullptr)
    {
      // apply SignatureDef
      for (auto it_signdef : *_tfl_signature_def_offsets)
      {
        if (it_signdef->subgraph_index() == subgraph_index)
        {
          auto outputs = it_signdef->outputs();
          assert(outputs->size() == output_vec.size());

          std::map<std::string, uint32_t> map_name_index;
          for (auto it_tm : *outputs)
          {
            map_name_index[it_tm->name()->str()] = it_tm->tensor_index();
          }
          uint32_t output_vec_idx = 0;
          for (auto &item : map_name_index)
          {
            output_vec[output_vec_idx++] = item.second;
          }
        }
      }
    }

    auto circle_outputs = _fb->CreateVector(output_vec);

    // operators of subgraph
    std::vector<flatbuffers::Offset<circle::Operator>> operator_vec;

    auto tflite_operators = it_sg->operators();
    if (tflite_operators != nullptr)
    {
      for (auto it : *tflite_operators)
      {
        // inputs
        std::vector<int32_t> input_vec{it->inputs()->begin(), it->inputs()->end()};
        auto circle_inputs = _fb->CreateVector(input_vec);
        // outputs
        std::vector<int32_t> output_vec{it->outputs()->begin(), it->outputs()->end()};
        auto circle_outputs = _fb->CreateVector(output_vec);
        // builtin options
        auto circle_builtin_options = get_circle_builtin_options(*_fb, it);
        auto circle_builtin_options_type = get_circle_builtin_options_type(it);
        // custom options
        flatbuffers::Offset<flatbuffers::Vector<uint8_t>> circle_custom_options;
        if (it->custom_options())
        {
          std::vector<uint8_t> custom_options_vec{it->custom_options()->begin(),
                                                  it->custom_options()->end()};
          circle_custom_options = _fb->CreateVector(custom_options_vec);
        }
        // custom options format
        // TODO Make get_circle_custom_options_format
        assert(it->custom_options_format() == tflite::CustomOptionsFormat_FLEXBUFFERS);
        auto circle_custom_options_format = circle::CustomOptionsFormat_FLEXBUFFERS;

        circle::OperatorBuilder operator_builder{*_fb};
        operator_builder.add_opcode_index(it->opcode_index());
        operator_builder.add_inputs(circle_inputs);
        operator_builder.add_outputs(circle_outputs);
        operator_builder.add_builtin_options(circle_builtin_options);
        operator_builder.add_builtin_options_type(circle_builtin_options_type);
        operator_builder.add_custom_options(circle_custom_options);
        operator_builder.add_custom_options_format(circle_custom_options_format);
        // TODO mutating_variable_inputs
        auto opeartor = operator_builder.Finish();
        operator_vec.emplace_back(opeartor);
      }
    }
    auto circle_operators = _fb->CreateVector(operator_vec);

    // name of subgraph
    auto subgraphs_name = _fb->CreateString(it_sg->name());

    // subgraphs
    auto circle_subgraph_builder = circle::SubGraphBuilder{*_fb};

    circle_subgraph_builder.add_tensors(circle_tensors);
    circle_subgraph_builder.add_inputs(circle_inputs);
    circle_subgraph_builder.add_outputs(circle_outputs);
    circle_subgraph_builder.add_operators(circle_operators);
    circle_subgraph_builder.add_name(subgraphs_name);

    auto circle_subgraph = circle_subgraph_builder.Finish();
    subgprahs_vec.emplace_back(circle_subgraph);

    // next subgraph
    subgraph_index = subgraph_index + 1;
  }
  _circle_flatbuffer_vec_offset = _fb->CreateVector(subgprahs_vec);
}

template <> void Offset<OperatorCodeLink>::build(const TFLFlatBufVec *tflite_flatbuffer_vec)
{
  std::vector<flatbuffers::Offset<circle::OperatorCode>> operator_code_vec;

  for (auto it : *tflite_flatbuffer_vec)
  {
    auto custom_code = _fb->CreateString(it->custom_code());
    circle::OperatorCodeBuilder operator_code_builder{*_fb};
    auto de_code = it->deprecated_builtin_code();
    auto bt_code = it->builtin_code();

    // There are two builtin codes (deprecated_builtin, (extended) builtin)
    // deprecated builtin code uses 0~126
    // extended builtin code uses 127~
    // NOTE 127 = BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES
    if (de_code >= 0 and de_code < 127)
    {
      // Use deprecated builtin opcode.
      auto cir_de_code = get_circle_builtin_code(de_code);
      auto cir_bt_code = get_circle_builtin_code(bt_code);
      // correct bt_code where bt_code == 0 for old tflite format
      if (cir_bt_code == 0)
        cir_bt_code = static_cast<circle::BuiltinOperator>(cir_de_code);
      operator_code_builder.add_deprecated_builtin_code(cir_de_code);
      operator_code_builder.add_builtin_code(cir_bt_code);
    }
    else
    {
      // Use extended builtin opcode
      // Set 127 (PLACEHOLDER_FOR_GREATER_OP_CODES) for deprecated builtin code
      auto cir_bt_code = get_circle_builtin_code(bt_code);
      operator_code_builder.add_deprecated_builtin_code(
        tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES);
      operator_code_builder.add_builtin_code(cir_bt_code);
    }
    operator_code_builder.add_custom_code(custom_code);
    operator_code_builder.add_version(it->version());
    auto code = operator_code_builder.Finish();
    operator_code_vec.emplace_back(code);
  }
  _circle_flatbuffer_vec_offset = _fb->CreateVector(operator_code_vec);
}

CircleModel::CircleModel(FlatBufBuilder &fb)
  : _version{0}, _description{fb->CreateString("ONE-tflite2circle")}, _fb{fb}
{
  // NOTHING TODO
}

void CircleModel::load_offsets(const tflite::Model *tfl_model)
{
  _operator_codes_offset = std::make_unique<Offset<OperatorCodeLink>>(_fb);
  _subGraphs_offset = std::make_unique<Offset<SubGraphLink>>(_fb);
  _buffers_offset = std::make_unique<Offset<BufferLink>>(_fb);
  _metadata_buffer_offset = std::make_unique<Offset<MetaDataBufferLink>>(_fb);

  _subGraphs_offset->set_signature_defs(tfl_model->signature_defs());

  _operator_codes_offset->build(tfl_model->operator_codes());
  _subGraphs_offset->build(tfl_model->subgraphs());
  _buffers_offset->build(tfl_model->buffers());
  _metadata_buffer_offset->build(tfl_model->metadata_buffer());
}

void CircleModel::model_build(void) const
{
  circle::ModelBuilder model_builder{*_fb};

  model_builder.add_version(_version);
  model_builder.add_description(_description);
  model_builder.add_operator_codes(_operator_codes_offset->offset());
  model_builder.add_subgraphs(_subGraphs_offset->offset());
  model_builder.add_buffers(_buffers_offset->offset());
  model_builder.add_metadata_buffer(_metadata_buffer_offset->offset());

  auto model = model_builder.Finish();
  circle::FinishModelBuffer(*_fb, model);
}

const char *CircleModel::base(void) const
{
  return reinterpret_cast<const char *>(_fb->GetBufferPointer());
}

size_t CircleModel::size(void) const { return _fb->GetSize(); }

} // namespace tflite2circle
