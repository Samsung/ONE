/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "tflchef/ModelChef.h"
#include <souschef/RangedArguments.h>
#include <souschef/Registry.h>

#include "Convert.h"

#include <souschef/DataChefs.h>

#include "OpChef.h"
#include "OpChefs.h"

#include <souschef/Dataset.h>
#include <souschef/Dims.h>

#include "Log.h"

#include <iterator>
#include <map>
#include <string>
#include <vector>

#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

using namespace souschef;

namespace
{

struct DataChefRegistry final : public Registry<DataChefFactory>
{
};

DataChefRegistry &data_chef_registry(const tflchef::TensorType &type)
{
  static DataChefRegistry s32;
  static DataChefRegistry s64;
  static DataChefRegistry fp32;
  static DataChefRegistry u8;
  static DataChefRegistry string;
  static DataChefRegistry boolean;
  static DataChefRegistry s16;
  static DataChefRegistry fp16;
  static DataChefRegistry s8;
  static DataChefRegistry s4;

  switch (type)
  {
    case tflchef::INT32:
      return s32;
    case tflchef::INT64:
      return s64;
    case tflchef::FLOAT32:
      return fp32;
    case tflchef::FLOAT16:
      return fp16;
    case tflchef::UINT8:
      return u8;
    case tflchef::STRING:
      return string;
    case tflchef::BOOL:
      return boolean;
    case tflchef::INT16:
      return s16;
    case tflchef::INT8:
      return s8;
    case tflchef::INT4:
      return s4;
    default:
      break;
  }

  throw std::runtime_error{"Unknown tensor type"};
}

struct OpChefRegistry final : public Registry<OpChefFactory>
{
};

OpChefRegistry &op_chef_registry(void)
{
  static OpChefRegistry registry;
  return registry;
}

/// @brief This will prepare a map of unique builtin codes in the model recipe
std::map<tflite::BuiltinOperator, int32_t>
gather_builtincode_map(const ::tflchef::ModelRecipe &model_recipe)
{
  // Key and value of the map are BuiltinOperator and operator version
  std::map<tflite::BuiltinOperator, int32_t> builtin_map;

  for (const auto &operation : model_recipe.operation())
  {
    if (operation.type() == "Custom")
      continue;

    auto op_chef = op_chef_registry().lookup(operation.type()).create(&operation);
    // Various operation version is unified as the highest version among them
    if (builtin_map.find(op_chef->code()) == builtin_map.end() ||
        builtin_map[op_chef->code()] < operation.version())
      builtin_map[op_chef->code()] = operation.version();
  }

  // Add ops used in Graphs(subgraphs)
  for (int g = 0; g < model_recipe.graph_size(); ++g)
  {
    const auto &graph = model_recipe.graph(g);
    for (const auto &operation : graph.operation())
    {
      if (operation.type() == "Custom")
        continue;

      auto op_chef = op_chef_registry().lookup(operation.type()).create(&operation);
      // Various operation version is unified as the highest version among them
      if (builtin_map.find(op_chef->code()) == builtin_map.end() ||
          builtin_map[op_chef->code()] < operation.version())
        builtin_map[op_chef->code()] = operation.version();
    }
  }

  return builtin_map;
}

/// @brief This will prepare a set of unique custom codes in the mode recipe
std::set<std::string> gather_customcode_set(const ::tflchef::ModelRecipe &model_recipe)
{
  std::set<std::string> customcode_set;
  for (const auto &operation : model_recipe.operation())
  {
    if (operation.type() == "Custom")
    {
      assert(not operation.custom_code().empty());
      customcode_set.insert(operation.custom_code());
    }
  }

  // Add ops used in Graphs(subgraphs)
  for (int g = 0; g < model_recipe.graph_size(); ++g)
  {
    const auto &graph = model_recipe.graph(g);
    for (const auto &operation : graph.operation())
    {
      if (operation.type() == "Custom")
      {
        assert(not operation.custom_code().empty());
        customcode_set.insert(operation.custom_code());
      }
    }
  }

  return customcode_set;
}

} // namespace

namespace
{

using SymboleTable_t = std::map<std::string, int32_t>;
using SparsityParams_t = flatbuffers::Offset<tflite::SparsityParameters>;
using SparsityDims_t = std::vector<sparsity::TfLiteDimensionType>;
using QuantParams_t = flatbuffers::Offset<tflite::QuantizationParameters>;

class ModelChef
{
public:
  ModelChef() = default;

public:
  void init(void);
  void cook(const ::tflchef::ModelRecipe &model_recipe);

private:
  void prepare_initial_buffer(void);
  void gather_operator_codes(const ::tflchef::ModelRecipe &model_recipe);
  void gather_signature_defs(const ::tflchef::ModelRecipe &model_recipe);

  void buffer_sparse_f32(int32_t &buffer_index, DimsI32_t &dims,
                         std::vector<int> &traversal_order_vec, SparsityDims_t &format_vec,
                         souschef::Data &data_vec, SparsityParams_t &sparsity_index);

  void buffer_dense(int32_t &buffer_index, const tflchef::Operand &operand, int32_t count,
                    souschef::Data &data_vec);

  template <typename T> void cook_operands(const T &graph);

  template <typename T> void prepare_tensor_symbols(const T &graph, SymboleTable_t &symbol_table);

  template <typename T> void cook_operations(const T &graph, SymboleTable_t &symbol_table);

  template <typename T> void cook_graph(const T &graph, SymboleTable_t &symbol_table);

  bool finalize_ext_buffer(void);

public:
  const char *get_buffer_pointer(void) const;
  size_t get_size(void) const;

private:
  std::unique_ptr<flatbuffers::FlatBufferBuilder> _flatbuffer_builder;

  std::vector<flatbuffers::Offset<::tflite::SignatureDef>> _signdef_vec;
  std::vector<flatbuffers::Offset<::tflite::Buffer>> _buffer_vec;
  std::vector<flatbuffers::Offset<::tflite::OperatorCode>> _code_vec;
  std::vector<flatbuffers::Offset<::tflite::SubGraph>> _subgraph_vec;
  std::map<tflite::BuiltinOperator, int32_t> _builtin_code_map;
  std::vector<std::string> _custom_code_vec;
  // _symbol_tables stores symbol_table of each sub graph
  // this is used to find tensor ID(index) with tensor name
  std::vector<SymboleTable_t> _symbol_tables;

  // per graph that needs clear afer graph is processed
  // Operand-related
  std::vector<flatbuffers::Offset<::tflite::Tensor>> _tensor_vec;
  // Operation-related
  std::vector<flatbuffers::Offset<::tflite::Operator>> _operator_vec;

  std::string _graph_name;

  // store Buffer data to external of FB and use (Buffer) offset/size fields
  bool _ext_offset = false;
  std::map<int32_t, std::vector<uint8_t>> _buffer_data_map;
  std::string _ext_data;
};

void ModelChef::init(void)
{
  _flatbuffer_builder =
    std::unique_ptr<flatbuffers::FlatBufferBuilder>(new flatbuffers::FlatBufferBuilder(1024));
}

std::vector<flatbuffers::Offset<tflite::DimensionMetadata>>
make_dim_metadata_vec(flatbuffers::FlatBufferBuilder *flatbuffer_builder, int32_t dims_count,
                      const std::vector<int> &traversal_order_vec, const SparsityDims_t &format_vec,
                      const std::vector<std::vector<int32_t>> &dim_metadata_src)
{
  // Build sparsity parameter.
  std::vector<flatbuffers::Offset<tflite::DimensionMetadata>> dim_metadata_vec(dims_count);
  for (int32_t i = 0; i < dims_count; i++)
  {
    const int32_t metadata_idx = 2 * i;
    if (format_vec[traversal_order_vec[i]] == sparsity::kTfLiteDimSparseCSR)
    {
      auto array_segments =
        tflite::CreateInt32Vector(*flatbuffer_builder,
                                  flatbuffer_builder->CreateVector(dim_metadata_src[metadata_idx]))
          .Union();
      auto array_indices =
        tflite::CreateInt32Vector(
          *flatbuffer_builder, flatbuffer_builder->CreateVector(dim_metadata_src[metadata_idx + 1]))
          .Union();
      dim_metadata_vec[i] =
        tflite::CreateDimensionMetadata(*flatbuffer_builder, tflite::DimensionType_SPARSE_CSR, 0,
                                        tflite::SparseIndexVector_Int32Vector, array_segments,
                                        tflite::SparseIndexVector_Int32Vector, array_indices);
    }
    else
    {
      dim_metadata_vec[i] = tflite::CreateDimensionMetadata(
        *flatbuffer_builder, tflite::DimensionType_DENSE, dim_metadata_src[metadata_idx][0]);
    }
  }
  return dim_metadata_vec;
}

void ModelChef::buffer_sparse_f32(int32_t &buffer_index, DimsI32_t &dims,
                                  std::vector<int> &traversal_order_vec, SparsityDims_t &format_vec,
                                  souschef::Data &data_vec, SparsityParams_t &sparsity_index)
{
  const int32_t dims_count = dims.size();

  ::sparsity::FormatConverter<float> converter(dims, traversal_order_vec, format_vec);
  converter.DenseToSparse(reinterpret_cast<const float *>(data_vec.data()));
  const auto &sparse_data = converter.GetData();

  std::vector<uint8_t> sparse_uint8;
  for (int c = 0; c < sparse_data.size(); ++c)
  {
    const float value = sparse_data.at(c);
    const uint8_t *arr = reinterpret_cast<const uint8_t *>(&value);
    for (uint32_t b = 0; b < sizeof(float); ++b)
    {
      sparse_uint8.emplace_back(arr[b]);
    }
  }
  if (_ext_offset)
  {
    buffer_index = _buffer_vec.size();
    _buffer_data_map[buffer_index] = sparse_uint8;

    auto buffer = tflite::CreateBuffer(*_flatbuffer_builder, 0, 1, 1);
    _buffer_vec.emplace_back(buffer);
  }
  else
  {
    auto data = _flatbuffer_builder->CreateVector(sparse_uint8);
    // Create Buffer
    tflite::BufferBuilder buffer_builder{*_flatbuffer_builder};
    buffer_builder.add_data(data);
    auto buffer = buffer_builder.Finish();

    // Update Buffer Index & Vector
    buffer_index = _buffer_vec.size();
    _buffer_vec.emplace_back(buffer);
  }

  // save SparsityParameters
  auto traversal_order = _flatbuffer_builder->CreateVector(traversal_order_vec);

  // Create block map
  std::vector<int> block_map_vec{};
  auto block_map = _flatbuffer_builder->CreateVector(block_map_vec);

  // Create dimension metadata
  const auto &dim_metadata_src = converter.GetDimMetadata();
  auto dim_metadata_vec = make_dim_metadata_vec(_flatbuffer_builder.get(), dims_count,
                                                traversal_order_vec, format_vec, dim_metadata_src);
  auto dim_metadata = _flatbuffer_builder->CreateVector(dim_metadata_vec);
  sparsity_index = tflite::CreateSparsityParameters(*_flatbuffer_builder, traversal_order,
                                                    block_map, dim_metadata);
}

void ModelChef::buffer_dense(int32_t &buffer_index, const tflchef::Operand &operand, int32_t count,
                             souschef::Data &data_vec)
{
  // pack for INT4 and replace data_vec
  if (operand.type() == tflchef::TensorType::INT4)
  {
    uint32_t packed = (count + 1) / 2;
    std::vector<uint8_t> data_packed(packed);
    for (uint32_t idx = 0; idx < packed; ++idx)
    {
      uint32_t sidx = idx * 2;
      data_packed[idx] = data_vec[sidx++] & 0x0f;
      if (sidx < count)
        data_packed[idx] |= data_vec[sidx] << 4;
    }
    data_vec = data_packed;
  }

  if (_ext_offset)
  {
    buffer_index = _buffer_vec.size();
    _buffer_data_map[buffer_index] = data_vec;

    auto buffer = tflite::CreateBuffer(*_flatbuffer_builder, 0, 1, 1);
    _buffer_vec.emplace_back(buffer);
  }
  else
  {
    auto data = _flatbuffer_builder->CreateVector(data_vec);

    // Create Buffer
    tflite::BufferBuilder buffer_builder{*_flatbuffer_builder};
    buffer_builder.add_data(data);
    auto buffer = buffer_builder.Finish();

    // Update Buffer Index & Vector
    buffer_index = _buffer_vec.size();
    _buffer_vec.emplace_back(buffer);
  }
}

template <typename T> void ModelChef::cook_operands(const T &graph)
{
  int32_t buffer_start = _buffer_vec.size();
  int32_t buffer_index = 0;

  // Create buffer(s) 1~n(I) for input(s)
  const auto size_input = graph.input_size();
  for (int ci = 0; ci < size_input; ++ci)
  {
    tflite::BufferBuilder buffer_builder{*_flatbuffer_builder};
    _buffer_vec.emplace_back(buffer_builder.Finish());
  }
  // Create buffer(s) n(I)+1~n(I)+n(O) for output(s)
  const auto size_output = graph.output_size();
  for (int co = 0; co < size_output; ++co)
  {
    tflite::BufferBuilder buffer_builder{*_flatbuffer_builder};
    _buffer_vec.emplace_back(buffer_builder.Finish());
  }

  auto input_names = as_dataset(graph.input()).vectorize();
  auto output_names = as_dataset(graph.output()).vectorize();

  for (const auto &operand : graph.operand())
  {
    assert(operand.has_name());
    assert(operand.has_type());

    SparsityParams_t sparsity_index;

    flatbuffers::Offset<flatbuffers::Vector<int32_t>> shape;
    std::vector<int32_t> dims;
    if (operand.has_shape())
    {
      dims = as_dims(operand.shape());
      shape = _flatbuffer_builder->CreateVector(dims);
    }

    auto name = _flatbuffer_builder->CreateString(operand.name());

    buffer_index = 0;

    // Create Buffer if filler is specified
    if (operand.has_filler())
    {
      // prohibit constant as graph input/output
      for (auto it = input_names.begin(); it != input_names.end(); ++it)
      {
        if (*it == operand.name())
          throw std::runtime_error{"Constant '" + *it + "' cannot be graph I/O"};
      }
      for (auto it = output_names.begin(); it != output_names.end(); ++it)
      {
        if (*it == operand.name())
          throw std::runtime_error{"Constant '" + *it + "' cannot be graph I/O"};
      }

      const auto &filler = operand.filler();

      assert(filler.has_tag());

      auto args = ranged_arguments(filler.arg().begin(), filler.arg().end());
      auto chef = data_chef_registry(operand.type()).lookup(filler.tag()).create(args);

      assert(chef != nullptr);

      // Create Data
      int32_t count = (element_count(dims) > 0) ? element_count(dims) : filler.arg_size();
      auto data_vec = chef->generate(count);

      if (operand.has_make_sparse() && operand.make_sparse())
      {
        assert(not operand.has_sparsity());
        assert(operand.has_shape());
        assert(operand.type() != tflchef::TensorType::INT4);

        const int32_t dims_count = dims.size();
        std::vector<int> traversal_order_vec;
        SparsityDims_t format_vec;
        for (int32_t o = 0; o < dims_count; ++o)
          traversal_order_vec.push_back(o);
        for (int32_t o = 0; o < dims_count - 1; ++o)
          format_vec.push_back(sparsity::kTfLiteDimDense);
        format_vec.push_back(sparsity::kTfLiteDimSparseCSR);

        if (operand.type() == tflchef::FLOAT32)
        {
          buffer_sparse_f32(buffer_index, dims, traversal_order_vec, format_vec, data_vec,
                            sparsity_index);
        }
        else if (operand.type() == tflchef::FLOAT16)
        {
          ::sparsity::FormatConverter<uint16_t> converter(dims, traversal_order_vec, format_vec);
          converter.DenseToSparse(reinterpret_cast<const uint16_t *>(data_vec.data()));
          const auto &sparse_data = converter.GetData();

          std::vector<uint8_t> sparse_uint8;
          for (int c = 0; c < sparse_data.size(); ++c)
          {
            const uint16_t value = sparse_data.at(c);
            const uint8_t *arr = reinterpret_cast<const uint8_t *>(&value);
            for (uint32_t b = 0; b < sizeof(uint16_t); ++b)
            {
              sparse_uint8.emplace_back(arr[b]);
            }
          }
          if (_ext_offset)
          {
            buffer_index = _buffer_vec.size();
            _buffer_data_map[buffer_index] = sparse_uint8;

            auto buffer = tflite::CreateBuffer(*_flatbuffer_builder, 0, 1, 1);
            _buffer_vec.emplace_back(buffer);
          }
          else
          {
            auto data = _flatbuffer_builder->CreateVector(sparse_uint8);

            // Create Buffer
            tflite::BufferBuilder buffer_builder{*_flatbuffer_builder};
            buffer_builder.add_data(data);
            auto buffer = buffer_builder.Finish();

            // Update Buffer Index & Vector
            buffer_index = _buffer_vec.size();
            _buffer_vec.emplace_back(buffer);
          }

          // save SparsityParameters
          auto traversal_order = _flatbuffer_builder->CreateVector(traversal_order_vec);

          // Create block map
          std::vector<int> block_map_vec{};
          auto block_map = _flatbuffer_builder->CreateVector(block_map_vec);

          // Create dimension metadata
          const auto &dim_metadata_src = converter.GetDimMetadata();
          auto dim_metadata_vec =
            make_dim_metadata_vec(_flatbuffer_builder.get(), dims_count, traversal_order_vec,
                                  format_vec, dim_metadata_src);
          auto dim_metadata = _flatbuffer_builder->CreateVector(dim_metadata_vec);
          sparsity_index = tflite::CreateSparsityParameters(*_flatbuffer_builder, traversal_order,
                                                            block_map, dim_metadata);
        }
        else
        {
          throw std::runtime_error{"NYI: unsupported operand type"};
        }
      }
      else
      {
        buffer_dense(buffer_index, operand, count, data_vec);
      }
    }
    else
    {
      // if this is input or output, assign to that buffer_index
      int idx = 0;
      for (auto it = input_names.begin(); it != input_names.end(); ++it, ++idx)
      {
        if (*it == operand.name())
        {
          buffer_index = buffer_start + idx;
          break;
        }
      }
      if (buffer_index == 0)
      {
        idx = 0;
        for (auto it = output_names.begin(); it != output_names.end(); ++it, ++idx)
        {
          if (*it == operand.name())
          {
            buffer_index = buffer_start + size_input + idx;
            break;
          }
        }
      }
      if (buffer_index == 0)
      {
        // we couldn't find the buffer; create an empty buffer for this tensor
        buffer_index = _buffer_vec.size();

        tflite::BufferBuilder buffer_builder{*_flatbuffer_builder};
        _buffer_vec.emplace_back(buffer_builder.Finish());
      }
    }
    assert(buffer_index != 0);

    QuantParams_t quant_index;

    // Create QuantizationParameters if quant is specified
    if (operand.has_quant())
    {
      const auto &quant = operand.quant();

      // Create each parameters
      // NOTE if some parameters are not given, those will be set to default value
      std::vector<float> quant_max_vec(quant.max_size());
      std::vector<float> quant_min_vec(quant.min_size());
      std::vector<float> quant_scale_vec(quant.scale_size());
      std::vector<int64_t> quant_zero_point_vec(quant.zero_point_size());

      for (uint32_t i = 0; i < quant.max_size(); ++i)
        quant_max_vec.at(i) = quant.max(i);
      for (uint32_t i = 0; i < quant.min_size(); ++i)
        quant_min_vec.at(i) = quant.min(i);
      for (uint32_t i = 0; i < quant.scale_size(); ++i)
        quant_scale_vec.at(i) = quant.scale(i);
      for (uint32_t i = 0; i < quant.zero_point_size(); ++i)
        quant_zero_point_vec.at(i) = quant.zero_point(i);

      auto quant_max = _flatbuffer_builder->CreateVector(quant_max_vec);
      auto quant_min = _flatbuffer_builder->CreateVector(quant_min_vec);
      auto quant_scale = _flatbuffer_builder->CreateVector(quant_scale_vec);
      auto quant_zero_point = _flatbuffer_builder->CreateVector(quant_zero_point_vec);

      // Create QuantizationParameters
      tflite::QuantizationParametersBuilder quant_builder{*_flatbuffer_builder};
      quant_builder.add_max(quant_max);
      quant_builder.add_min(quant_min);
      quant_builder.add_scale(quant_scale);
      quant_builder.add_zero_point(quant_zero_point);
      quant_builder.add_quantized_dimension(quant.quantized_dimension());

      // Update QuantizationParameters Index
      quant_index = quant_builder.Finish();
    }

    if (operand.has_sparsity())
    {
      const auto &sparsity = operand.sparsity();

      // Create traversal order
      std::vector<int> traversal_order_vec{sparsity.traversal_order().dim().begin(),
                                           sparsity.traversal_order().dim().end()};
      auto traversal_order = _flatbuffer_builder->CreateVector(traversal_order_vec);

      // Create block map
      std::vector<int> block_map_vec{sparsity.block_map().dim().begin(),
                                     sparsity.block_map().dim().end()};
      auto block_map = _flatbuffer_builder->CreateVector(block_map_vec);

      // Create dimension metadata
      std::vector<flatbuffers::Offset<tflite::DimensionMetadata>> dim_metadata_vec;
      auto recipe_dim_metadata = sparsity.dim_metadata();
      for (const auto &dm : recipe_dim_metadata)
      {
        // Create array segments
        auto tflite_array_segments =
          as_tflite_sparse_index_vec(*_flatbuffer_builder, dm.array_segments());

        // Create array indices
        auto tflite_array_indices =
          as_tflite_sparse_index_vec(*_flatbuffer_builder, dm.array_indices());

        auto tflite_dim_metadata_builder = tflite::DimensionMetadataBuilder{*_flatbuffer_builder};
        tflite_dim_metadata_builder.add_format(as_tflite_dimensiontype(dm.format()));
        tflite_dim_metadata_builder.add_dense_size(dm.dense_size());
        tflite_dim_metadata_builder.add_array_segments(tflite_array_segments);
        tflite_dim_metadata_builder.add_array_segments_type(
          as_tflite_sparse_idx_vec_type(dm.array_segments().type()));
        tflite_dim_metadata_builder.add_array_indices(tflite_array_indices);
        tflite_dim_metadata_builder.add_array_indices_type(
          as_tflite_sparse_idx_vec_type(dm.array_indices().type()));
        auto tflite_dim_metadata = tflite_dim_metadata_builder.Finish();
        dim_metadata_vec.emplace_back(tflite_dim_metadata);
      }
      auto dim_metadata = _flatbuffer_builder->CreateVector(dim_metadata_vec);

      sparsity_index = tflite::CreateSparsityParameters(*_flatbuffer_builder, traversal_order,
                                                        block_map, dim_metadata);
    }

    flatbuffers::Offset<flatbuffers::Vector<int32_t>> shape_signature;
    if (operand.has_shape_signature())
    {
      auto signature = as_dims(operand.shape_signature());
      shape_signature = _flatbuffer_builder->CreateVector(signature);
    }

    // Create Tensor
    tflite::TensorBuilder tensor_builder{*_flatbuffer_builder};

    tensor_builder.add_shape(shape);
    tensor_builder.add_type(as_tflite_tensortype(operand.type()));
    tensor_builder.add_buffer(buffer_index);
    tensor_builder.add_name(name);
    tensor_builder.add_is_variable(operand.is_variable());
    if (operand.has_quant())
      tensor_builder.add_quantization(quant_index);
    tensor_builder.add_sparsity(sparsity_index);
    if (operand.has_shape_signature())
      tensor_builder.add_shape_signature(shape_signature);

    // Append!
    _tensor_vec.emplace_back(tensor_builder.Finish());
  }
}

template <typename T>
void ModelChef::prepare_tensor_symbols(const T &graph, SymboleTable_t &symbol_table)
{
  LOGGER(l);

  for (const auto &operand : graph.operand())
  {
    // Update Tensor Name -> Tensor Index Map
    int32_t tensor_index = symbol_table.size();
    const auto &tensor_name = operand.name();

    INFO(l) << "Symbol [" << tensor_name << "] = Tensor " << tensor_index << std::endl;

    symbol_table[tensor_name] = tensor_index;
  }
}

template <typename T> void ModelChef::cook_operations(const T &graph, SymboleTable_t &symbol_table)
{
  auto lookup = [&](const std::string &name) {
    if (symbol_table.find(name) != symbol_table.end())
      return symbol_table.at(name);
    else if (name == "")
      return -1; // -1 in TFLite means that optional input tensor is empty.
    else
    {
      std::string msg = "tflchef : input not found in " + _graph_name + " graph";
      throw std::runtime_error(msg.c_str());
    }
  };

  // Create Operator
  for (const auto &operation : graph.operation())
  {
    assert(operation.has_type());

    std::string op_type = operation.type();
    if (not operation.custom_code().empty())
      op_type = operation.custom_code();

    auto op_chef = op_chef_registry().lookup(op_type).create(&operation);

    // Create 'inputs'
    std::vector<int32_t> input_vec = as_dataset(operation.input()).map(lookup).vectorize();
    auto inputs = _flatbuffer_builder->CreateVector(input_vec);

    // Create 'outputs'
    std::vector<int32_t> output_vec = as_dataset(operation.output()).map(lookup).vectorize();
    auto outputs = _flatbuffer_builder->CreateVector(output_vec);

    // Create Option
    auto options = op_chef->value(*_flatbuffer_builder);

    // Create Custom option
    auto circle_custom_options = op_chef->custom_value(*_flatbuffer_builder);

    // Create Operator
    tflite::OperatorBuilder op_builder{*_flatbuffer_builder};

    // Note that opcode_index is an index into the operator_codes vector.
    // operator_codes consists of buildtin_code and custom_code, which is inserted sequentially.
    uint32_t opcode_index = 0;
    auto op_it = _builtin_code_map.find(op_chef->code());
    // builtin operator
    if (op_it != _builtin_code_map.end())
    {
      opcode_index = std::distance(_builtin_code_map.begin(), op_it);
    }
    // custom operator
    else
    {
      assert(not operation.custom_code().empty());
      const auto &custom_code = operation.custom_code();
      auto op_it = std::find(_custom_code_vec.begin(), _custom_code_vec.end(), custom_code);
      assert(op_it != _custom_code_vec.end());
      opcode_index = _builtin_code_map.size();
      opcode_index += std::distance(_custom_code_vec.begin(), op_it);
    }

    op_builder.add_opcode_index(opcode_index);
    op_builder.add_inputs(inputs);
    op_builder.add_outputs(outputs);
    op_builder.add_builtin_options_type(op_chef->type());
    op_builder.add_builtin_options(options);
    op_builder.add_custom_options(circle_custom_options);
    op_builder.add_custom_options_format(tflite::CustomOptionsFormat_FLEXBUFFERS);
    // Append Operator
    _operator_vec.emplace_back(op_builder.Finish());
  }
}

template <typename T> void ModelChef::cook_graph(const T &graph, SymboleTable_t &symbol_table)
{
  LOGGER(l);

  assert(symbol_table.empty());  // FIX_CALLER_UNLESS
  assert(_tensor_vec.empty());   // FIX_CALLER_UNLESS
  assert(_operator_vec.empty()); // FIX_CALLER_UNLESS

  // default name for graph
  if (graph.has_name())
    _graph_name = graph.name();

  auto lookup = [&](const std::string &name) {
    if (symbol_table.find(name) != symbol_table.end())
      return symbol_table.at(name);
    else if (name == "")
      return -1; // -1 in TFLite means that optional input tensor is empty.
    else
    {
      std::string msg = "tflchef : input not found in " + _graph_name + " graph";
      throw std::runtime_error(msg.c_str());
    }
  };

  cook_operands(graph);

  prepare_tensor_symbols(graph, symbol_table);

  cook_operations(graph, symbol_table);

  // Create network input/output vector
  std::vector<int32_t> input_vec = as_dataset(graph.input()).map(lookup).vectorize();
  std::vector<int32_t> output_vec = as_dataset(graph.output()).map(lookup).vectorize();

  // Create "SubGraph" arguments
  auto tensors = _flatbuffer_builder->CreateVector(_tensor_vec);
  auto inputs = _flatbuffer_builder->CreateVector(input_vec);
  auto outputs = _flatbuffer_builder->CreateVector(output_vec);
  auto operators = _flatbuffer_builder->CreateVector(_operator_vec);
  auto name = _flatbuffer_builder->CreateString(_graph_name);

  tflite::SubGraphBuilder subgraph_builder{*_flatbuffer_builder};

  subgraph_builder.add_tensors(tensors);
  subgraph_builder.add_inputs(inputs);
  subgraph_builder.add_outputs(outputs);
  subgraph_builder.add_operators(operators);
  subgraph_builder.add_name(name);

  _subgraph_vec.emplace_back(subgraph_builder.Finish());
}

void ModelChef::gather_operator_codes(const ::tflchef::ModelRecipe &model_recipe)
{
  // Create OperatorCode with Builtin Operator
  _builtin_code_map = gather_builtincode_map(model_recipe);
  for (auto const &opcode : _builtin_code_map)
  {
    tflite::OperatorCodeBuilder code_builder{*_flatbuffer_builder};
    // 127 is BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES
    // This is the way to handle deprecated builtin code
    // See
    // https://github.com/tensorflow/tensorflow/blob/a0afe8f9218be5eb9ed5dffc2dff652996da8c28/tensorflow/lite/schema/schema.fbs#L1061-L1077
    if (opcode.first < 127)
    {
      code_builder.add_deprecated_builtin_code(opcode.first);
    }
    else
    {
      code_builder.add_deprecated_builtin_code(
        ::tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES);
    }
    code_builder.add_version(opcode.second);
    code_builder.add_builtin_code(opcode.first);
    auto code = code_builder.Finish();
    // Update OperatorCode vector
    _code_vec.emplace_back(code);
  }

  // Create OperatorCode with Custom Operator
  {
    std::set<std::string> custom_code_set = gather_customcode_set(model_recipe);
    std::vector<std::string> custom_code_vec{custom_code_set.begin(), custom_code_set.end()};
    _custom_code_vec = custom_code_vec;
  }

  for (const auto &opcode : _custom_code_vec)
  {
    auto custom_code = _flatbuffer_builder->CreateString(opcode);
    tflite::OperatorCodeBuilder code_builder{*_flatbuffer_builder};
    code_builder.add_deprecated_builtin_code(tflite::BuiltinOperator_CUSTOM);
    code_builder.add_custom_code(custom_code);
    code_builder.add_builtin_code(tflite::BuiltinOperator_CUSTOM);
    auto code = code_builder.Finish();
    // Update OperatorCode vector
    _code_vec.emplace_back(code);
  }
}

void ModelChef::prepare_initial_buffer(void)
{
  // Create an Empty Buffer
  //
  // Buffer 0 SHOULD be an empty buffer in TensorFlow Lite model file
  // (Please refer to the comment for Tensor.buffer field in schema)
  tflite::BufferBuilder buffer_builder{*_flatbuffer_builder};
  _buffer_vec.emplace_back(buffer_builder.Finish());
}

void ModelChef::gather_signature_defs(const ::tflchef::ModelRecipe &model_recipe)
{
  for (int s = 0; s < model_recipe.signature_def_size(); ++s)
  {
    // load from recipe
    const auto &rec_signature_def = model_recipe.signature_def(s);

    std::vector<flatbuffers::Offset<::tflite::TensorMap>> tensormap_inputs;
    std::vector<flatbuffers::Offset<::tflite::TensorMap>> tensormap_outputs;

    // which subgraph index to cook
    auto subgraph_index = 0;
    if (rec_signature_def.has_subgraph_index())
    {
      subgraph_index = rec_signature_def.subgraph_index();
    }
    assert(subgraph_index < _symbol_tables.size());
    auto &symbol_table = _symbol_tables[subgraph_index];

    // cook for inputs
    for (int si = 0; si < rec_signature_def.inputs_size(); ++si)
    {
      // recipe for input TensorMap
      const auto &rec_tm_input = rec_signature_def.inputs(si);
      auto name = _flatbuffer_builder->CreateString(rec_tm_input.name());
      uint32_t tensor_index = 0;
      // either tensor or tensor_index should exist
      assert(rec_tm_input.has_tensor() || rec_tm_input.has_tensor_index());
      if (rec_tm_input.has_tensor())
      {
        // we can get tensor_index from symbol_table
        const auto &tensor = rec_tm_input.tensor();
        tensor_index = symbol_table[tensor];
      }
      else
      {
        // or we can use tensor_index itself
        tensor_index = rec_tm_input.tensor_index();
      }

      ::tflite::TensorMapBuilder tensormap_builder{*_flatbuffer_builder};
      tensormap_builder.add_name(name);
      tensormap_builder.add_tensor_index(tensor_index);
      tensormap_inputs.push_back(tensormap_builder.Finish());
    }
    // cook for outputs, same as inputs
    for (int so = 0; so < rec_signature_def.outputs_size(); ++so)
    {
      const auto &rec_tm_output = rec_signature_def.outputs(so);
      auto name = _flatbuffer_builder->CreateString(rec_tm_output.name());
      uint32_t tensor_index = 0;
      assert(rec_tm_output.has_tensor() || rec_tm_output.has_tensor_index());
      if (rec_tm_output.has_tensor())
      {
        const auto &tensor = rec_tm_output.tensor();
        tensor_index = symbol_table[tensor];
      }
      else
      {
        tensor_index = rec_tm_output.tensor_index();
      }

      ::tflite::TensorMapBuilder tensormap_builder{*_flatbuffer_builder};
      tensormap_builder.add_name(name);
      tensormap_builder.add_tensor_index(tensor_index);
      tensormap_outputs.push_back(tensormap_builder.Finish());
    }

    auto inputs = _flatbuffer_builder->CreateVector(tensormap_inputs);
    auto outputs = _flatbuffer_builder->CreateVector(tensormap_outputs);
    auto signature_key = _flatbuffer_builder->CreateString(rec_signature_def.signature_key());
    // TODO add validation for signature_key

    ::tflite::SignatureDefBuilder signature_def_builder{*_flatbuffer_builder};
    signature_def_builder.add_inputs(inputs);
    signature_def_builder.add_outputs(outputs);
    signature_def_builder.add_signature_key(signature_key);
    signature_def_builder.add_subgraph_index(rec_signature_def.subgraph_index());

    _signdef_vec.emplace_back(signature_def_builder.Finish());
  }
}

bool ModelChef::finalize_ext_buffer(void)
{
  // NOTE modification of std::string object in the middle may reallocate it.
  // we will use std::string::reserve() to prevent this.

  auto align16 = [](size_t &v) {
    while (v % 16 != 0)
      v++;
  };

  // get total memory for flatbuffer + all buffer_data
  size_t result_size = _flatbuffer_builder->GetSize();
  align16(result_size);
  for (auto &it : _buffer_data_map)
  {
    std::vector<uint8_t> &buffer_data = it.second;
    result_size += buffer_data.size();
    align16(result_size);
  }
  align16(result_size);
  result_size += 16; // additional for safety

  std::string result;
  auto *buff_ptr = reinterpret_cast<const char *>(_flatbuffer_builder->GetBufferPointer());

  auto padalign16 = [](std::string &str) {
    while (str.size() % 16 != 0)
      str += '\0';
  };

  result.reserve(result_size);
  result.append(buff_ptr, _flatbuffer_builder->GetSize());

  auto mutable_model = tflite::GetMutableModel(result.data());
  auto mutable_buffers = mutable_model->mutable_buffers();
  bool ret = true;

  padalign16(result);
  for (auto &it : _buffer_data_map)
  {
    int32_t buffer_index = it.first;
    std::vector<uint8_t> &buffer_data = it.second;
    uint64_t offset = result.size();
    uint64_t size = buffer_data.size();

    tflite::Buffer *mutable_buffer = mutable_buffers->GetMutableObject(buffer_index);
    ret &= mutable_buffer->mutate_offset(offset);
    ret &= mutable_buffer->mutate_size(size);

    result.append(buffer_data.begin(), buffer_data.end());
    padalign16(result);
  }
  padalign16(result);

  // use final result
  _ext_data = result;

  return ret;
}

void ModelChef::cook(const ::tflchef::ModelRecipe &model_recipe)
{
  // use Custom/Buffer offset
  _ext_offset = model_recipe.has_ext_offset() ? model_recipe.ext_offset() : false;

  prepare_initial_buffer();

  gather_operator_codes(model_recipe);

  //
  // Create Main graph
  //

  _graph_name = "main";
  // Tensor Name -> Tensor ID mapping (per Graph)
  SymboleTable_t symbol_table;
  cook_graph<::tflchef::ModelRecipe>(model_recipe, symbol_table);
  _symbol_tables.push_back(symbol_table);

  //
  // Create subgraphs if exist
  //
  for (int g = 0; g < model_recipe.graph_size(); ++g)
  {
    const auto &graph = model_recipe.graph(g);

    std::ostringstream stringStream;
    stringStream << "sub_" << (g + 1);

    _graph_name = stringStream.str();

    symbol_table.clear();
    _tensor_vec.clear();
    _operator_vec.clear();
    cook_graph<::tflchef::Graph>(graph, symbol_table);
    _symbol_tables.push_back(symbol_table);
  }

  gather_signature_defs(model_recipe);

  // Create "Model" arguments
  auto buffers = _flatbuffer_builder->CreateVector(_buffer_vec);
  auto signdefs = _flatbuffer_builder->CreateVector(_signdef_vec);
  auto operator_codes = _flatbuffer_builder->CreateVector(_code_vec);
  auto subgraphs = _flatbuffer_builder->CreateVector(_subgraph_vec);
  auto description = _flatbuffer_builder->CreateString("Generated by tflchef");

  // Create "Model"
  tflite::ModelBuilder model_builder{*_flatbuffer_builder};

  model_builder.add_version(3);
  model_builder.add_operator_codes(operator_codes);
  model_builder.add_signature_defs(signdefs);
  model_builder.add_subgraphs(subgraphs);
  model_builder.add_description(description);
  model_builder.add_buffers(buffers);

  auto model = model_builder.Finish();

  // Finalize
  ::tflite::FinishModelBuffer(*_flatbuffer_builder, model);

  if (_ext_offset)
    finalize_ext_buffer();
}

const char *ModelChef::get_buffer_pointer(void) const
{
  if (_ext_offset)
    return _ext_data.data();
  return reinterpret_cast<const char *>(_flatbuffer_builder->GetBufferPointer());
}

size_t ModelChef::get_size(void) const
{
  if (_ext_offset)
    return _ext_data.size();
  return _flatbuffer_builder->GetSize();
}

} // namespace

namespace
{

class GeneratedModelImpl final : public tflchef::GeneratedModel::Impl
{
public:
  GeneratedModelImpl()
  {
    // DO NOTHING
  }

public:
  const char *base(void) const override { return _mc.get_buffer_pointer(); }

  size_t size(void) const override { return _mc.get_size(); }

public:
  ModelChef &model_chef(void) { return _mc; }

private:
  ModelChef _mc;
};

} // namespace

namespace tflchef
{

/**
 * @brief Generate a (in-memory) TensorFlow Lite model from a given model recipe
 */
GeneratedModel cook(const ::tflchef::ModelRecipe &model_recipe)
{
// Initialize Op Chef Registry
#define OP_CHEF(NAME, FACTORY_CLASS) \
  op_chef_registry().add(#NAME, std::unique_ptr<FACTORY_CLASS>(new FACTORY_CLASS()));
#include "OpChef.def"
#undef OP_CHEF

// Initialize Data Chef Registry
#define DATA_CHEF(TYPE, NAME, FACTORY_CLASS) \
  data_chef_registry(::tflchef::TYPE)        \
    .add(#NAME, std::unique_ptr<FACTORY_CLASS>(new FACTORY_CLASS()));
#include "DataChef.def"
#undef DATA_CHEF

  std::unique_ptr<GeneratedModelImpl> gen_model(new GeneratedModelImpl());

  ModelChef &mc = gen_model->model_chef();

  mc.init();
  mc.cook(model_recipe);

  // Return "GenerateModel"
  return GeneratedModel{std::move(gen_model)};
}

} // namespace tflchef
