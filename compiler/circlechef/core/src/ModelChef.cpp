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

#include "circlechef/ModelChef.h"
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

class GeneratedModelImpl final : public circlechef::GeneratedModel::Impl
{
public:
  GeneratedModelImpl(std::unique_ptr<flatbuffers::FlatBufferBuilder> &&builder)
    : _builder{std::move(builder)}
  {
    // DO NOTHING
  }

public:
  const char *base(void) const override
  {
    // Return the base address of generated flatbuffer model
    return reinterpret_cast<const char *>(_builder->GetBufferPointer());
  }

public:
  size_t size(void) const override
  {
    // Return the size of generated flatbuffer model
    return _builder->GetSize();
  }

private:
  std::unique_ptr<flatbuffers::FlatBufferBuilder> _builder;
};

} // namespace

namespace
{

struct DataChefRegistry final : public Registry<DataChefFactory>
{
};

DataChefRegistry &data_chef_registry(const circlechef::TensorType &type)
{
  static DataChefRegistry s32;
  static DataChefRegistry s64;
  static DataChefRegistry fp32;
  static DataChefRegistry u8;
  static DataChefRegistry string;
  static DataChefRegistry boolean;
  static DataChefRegistry s16;

  switch (type)
  {
    case circlechef::INT32:
      return s32;
    case circlechef::INT64:
      return s64;
    case circlechef::FLOAT32:
      return fp32;
    case circlechef::UINT8:
      return u8;
    case circlechef::STRING:
      return string;
    case circlechef::BOOL:
      return boolean;
    case circlechef::INT16:
      return s16;
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
std::map<circle::BuiltinOperator, int32_t>
gather_builtincode_map(const ::circlechef::ModelRecipe &model_recipe)
{
  // Key and value of the map are BuiltinOperator and operator version
  std::map<circle::BuiltinOperator, int32_t> builtin_map;

  for (const auto &operation : model_recipe.operation())
  {
    if (operation.type() == "Custom" || (operation.has_extype() && operation.extype() == "Custom"))
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
      if (operation.type() == "Custom" ||
          (operation.has_extype() && operation.extype() == "Custom"))
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
std::set<std::string> gather_customcode_set(const ::circlechef::ModelRecipe &model_recipe)
{
  std::set<std::string> customcode_set;
  for (const auto &operation : model_recipe.operation())
  {
    if (operation.type() == "Custom" || (operation.has_extype() && operation.extype() == "Custom"))
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
      if (operation.type() == "Custom" ||
          (operation.has_extype() && operation.extype() == "Custom"))
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

struct CookParams
{
  std::vector<flatbuffers::Offset<::circle::Buffer>> &buffer_vec;
  std::vector<flatbuffers::Offset<::circle::OperatorCode>> &code_vec;
  std::vector<flatbuffers::Offset<::circle::SubGraph>> &subgraph_vec;
  std::unique_ptr<flatbuffers::FlatBufferBuilder> &flatbuffer_builder;
  std::map<circle::BuiltinOperator, int32_t> &builtin_code_map;
  std::string noname;
};

template <typename T> void cook_graph(const T &graph, CookParams &cp)
{
  LOGGER(l);

  std::vector<flatbuffers::Offset<::circle::Buffer>> &buffer_vec = cp.buffer_vec;
  std::vector<flatbuffers::Offset<::circle::OperatorCode>> &code_vec = cp.code_vec;
  std::vector<flatbuffers::Offset<::circle::SubGraph>> &subgraph_vec = cp.subgraph_vec;
  std::unique_ptr<flatbuffers::FlatBufferBuilder> &flatbuffer_builder = cp.flatbuffer_builder;
  std::map<circle::BuiltinOperator, int32_t> &builtin_code_map = cp.builtin_code_map;

  // Operand-related
  std::vector<flatbuffers::Offset<::circle::Tensor>> tensor_vec;

  // Operation-related
  std::vector<flatbuffers::Offset<::circle::Operator>> operator_vec;

  // default name for graph
  std::string graph_name = cp.noname;
  if (graph.has_name())
    graph_name = graph.name();

  // Tensor Name -> Tensor ID mapping (per Graph)
  std::map<std::string, int32_t> symbol_table;

  auto lookup = [&symbol_table, &graph_name](const std::string &name) {
    if (symbol_table.find(name) != symbol_table.end())
      return symbol_table.at(name);
    else if (name == "")
      return -1; // -1 in circle means that optional input tensor is empty.
    else
    {
      std::string msg = "circlechef : input not found in " + graph_name + " graph";
      throw std::runtime_error(msg.c_str());
    }
  };

  int32_t buffer_start = buffer_vec.size();
  int32_t buffer_index = 0;

  // Create buffer(s) 1~n(I) for input(s)
  const auto size_input = graph.input_size();
  for (int ci = 0; ci < size_input; ++ci)
  {
    circle::BufferBuilder buffer_builder{*flatbuffer_builder};
    buffer_vec.emplace_back(buffer_builder.Finish());
  }
  // Create buffer(s) n(I)+1~n(I)+n(O) for output(s)
  const auto size_output = graph.output_size();
  for (int co = 0; co < size_output; ++co)
  {
    circle::BufferBuilder buffer_builder{*flatbuffer_builder};
    buffer_vec.emplace_back(buffer_builder.Finish());
  }

  auto input_names = as_dataset(graph.input()).vectorize();
  auto output_names = as_dataset(graph.output()).vectorize();

  for (const auto &operand : graph.operand())
  {
    assert(operand.has_name());

    assert(operand.has_type());

    flatbuffers::Offset<flatbuffers::Vector<int32_t>> shape;
    std::vector<int32_t> dims;
    if (operand.has_shape())
    {
      dims = as_dims(operand.shape());
      shape = flatbuffer_builder->CreateVector(dims);
    }

    auto name = flatbuffer_builder->CreateString(operand.name());

    buffer_index = 0;

    // Create Buffer if filler is specified
    if (operand.has_filler())
    {
      const auto &filler = operand.filler();

      assert(filler.has_tag());

      auto args = ranged_arguments(filler.arg().begin(), filler.arg().end());
      auto chef = data_chef_registry(operand.type()).lookup(filler.tag()).create(args);

      assert(chef != nullptr);

      // Create Data
      int32_t count = (element_count(dims) > 0) ? element_count(dims) : filler.arg_size();
      auto data_vec = chef->generate(count);
      auto data = flatbuffer_builder->CreateVector(data_vec);

      // Create Buffer
      circle::BufferBuilder buffer_builder{*flatbuffer_builder};
      buffer_builder.add_data(data);
      auto buffer = buffer_builder.Finish();

      // Update Buffer Index & Vector
      buffer_index = buffer_vec.size();
      buffer_vec.emplace_back(buffer);
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
        buffer_index = buffer_vec.size();

        circle::BufferBuilder buffer_builder{*flatbuffer_builder};
        buffer_vec.emplace_back(buffer_builder.Finish());
      }
    }
    assert(buffer_index != 0);

    flatbuffers::Offset<circle::QuantizationParameters> quant_index;

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

      auto quant_max = flatbuffer_builder->CreateVector(quant_max_vec);
      auto quant_min = flatbuffer_builder->CreateVector(quant_min_vec);
      auto quant_scale = flatbuffer_builder->CreateVector(quant_scale_vec);
      auto quant_zero_point = flatbuffer_builder->CreateVector(quant_zero_point_vec);

      // Create QuantizationParameters
      circle::QuantizationParametersBuilder quant_builder{*flatbuffer_builder};
      quant_builder.add_max(quant_max);
      quant_builder.add_min(quant_min);
      quant_builder.add_scale(quant_scale);
      quant_builder.add_zero_point(quant_zero_point);
      quant_builder.add_quantized_dimension(quant.quantized_dimension());

      // Update QuantizationParameters Index
      quant_index = quant_builder.Finish();
    }

    flatbuffers::Offset<flatbuffers::Vector<int32_t>> shape_signature;
    if (operand.has_shape_signature())
    {
      auto signature = as_dims(operand.shape_signature());
      shape_signature = flatbuffer_builder->CreateVector(signature);
    }

    // Create Tensor
    circle::TensorBuilder tensor_builder{*flatbuffer_builder};

    tensor_builder.add_shape(shape);
    tensor_builder.add_type(as_circle_tensortype(operand.type()));
    tensor_builder.add_buffer(buffer_index);
    tensor_builder.add_name(name);
    if (operand.has_quant())
      tensor_builder.add_quantization(quant_index);
    if (operand.has_shape_signature())
      tensor_builder.add_shape_signature(shape_signature);

    // Append!
    tensor_vec.emplace_back(tensor_builder.Finish());

    // Update Tensor Name -> Tensor Index Map
    int32_t tensor_index = symbol_table.size();
    const auto &tensor_name = operand.name();

    INFO(l) << "Symbol [" << tensor_name << "] = Tensor " << tensor_index << std::endl;

    symbol_table[tensor_name] = tensor_index;
  }

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
    auto inputs = flatbuffer_builder->CreateVector(input_vec);

    // Create 'outputs'
    std::vector<int32_t> output_vec = as_dataset(operation.output()).map(lookup).vectorize();
    auto outputs = flatbuffer_builder->CreateVector(output_vec);

    // Create Option
    auto options = op_chef->value(*flatbuffer_builder);

    // Create Custom option
    auto circle_custom_options = op_chef->custom_value(*flatbuffer_builder);

    // Create Operator
    circle::OperatorBuilder op_builder{*flatbuffer_builder};

    // Get operator code index from builtin_code_map with assumption, order of
    // builtin_code_map is same as that of code_vec
    auto op_it = builtin_code_map.find(op_chef->code());
    assert(op_it != builtin_code_map.end());
    uint32_t opcode_index = std::distance(builtin_code_map.begin(), op_it);

    op_builder.add_opcode_index(opcode_index);
    op_builder.add_inputs(inputs);
    op_builder.add_outputs(outputs);
    op_builder.add_builtin_options_type(op_chef->type());
    op_builder.add_builtin_options(options);
    op_builder.add_custom_options(circle_custom_options);
    op_builder.add_custom_options_format(circle::CustomOptionsFormat_FLEXBUFFERS);
    // Append Operator
    operator_vec.emplace_back(op_builder.Finish());
  }

  // Create network input/output vector
  std::vector<int32_t> input_vec = as_dataset(graph.input()).map(lookup).vectorize();
  std::vector<int32_t> output_vec = as_dataset(graph.output()).map(lookup).vectorize();

  // Create "SubGraph" arguments
  auto tensors = flatbuffer_builder->CreateVector(tensor_vec);
  auto inputs = flatbuffer_builder->CreateVector(input_vec);
  auto outputs = flatbuffer_builder->CreateVector(output_vec);
  auto operators = flatbuffer_builder->CreateVector(operator_vec);
  auto name = flatbuffer_builder->CreateString(graph_name);

  circle::SubGraphBuilder subgraph_builder{*flatbuffer_builder};

  subgraph_builder.add_tensors(tensors);
  subgraph_builder.add_inputs(inputs);
  subgraph_builder.add_outputs(outputs);
  subgraph_builder.add_operators(operators);
  subgraph_builder.add_name(name);

  subgraph_vec.emplace_back(subgraph_builder.Finish());
}

} // namespace

namespace circlechef
{

/**
 * @brief Generate a (in-memory) TensorFlow Lite model from a given model recipe
 */
GeneratedModel cook(const ::circlechef::ModelRecipe &model_recipe)
{
// Initialize Op Chef Registry
#define OP_CHEF(NAME, FACTORY_CLASS) \
  op_chef_registry().add(#NAME, std::unique_ptr<FACTORY_CLASS>(new FACTORY_CLASS()));
#include "OpChef.def"
#undef OP_CHEF

// Initialize Data Chef Registry
#define DATA_CHEF(TYPE, NAME, FACTORY_CLASS) \
  data_chef_registry(::circlechef::TYPE)     \
    .add(#NAME, std::unique_ptr<FACTORY_CLASS>(new FACTORY_CLASS()));
#include "DataChef.def"
#undef DATA_CHEF

  //
  // Create FlatBufferBuilder
  //
  auto flatbuffer_builder =
    std::unique_ptr<flatbuffers::FlatBufferBuilder>(new flatbuffers::FlatBufferBuilder(1024));

  // Operand-related
  std::vector<flatbuffers::Offset<::circle::Buffer>> buffer_vec;

  // Operation-related
  std::vector<flatbuffers::Offset<::circle::OperatorCode>> code_vec;

  // Graphs-related
  std::vector<flatbuffers::Offset<::circle::SubGraph>> subgraph_vec;

  // Create OperatorCode with Builtin Operator
  std::map<circle::BuiltinOperator, int32_t> builtin_code_map =
    gather_builtincode_map(model_recipe);
  for (auto const &opcode : builtin_code_map)
  {
    circle::OperatorCodeBuilder code_builder{*flatbuffer_builder};
    int8_t dep_code = 127; // BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES
    if (opcode.first < circle::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES)
      dep_code = static_cast<int8_t>(opcode.first);
    code_builder.add_deprecated_builtin_code(dep_code);
    code_builder.add_builtin_code(opcode.first);
    code_builder.add_version(opcode.second);
    auto code = code_builder.Finish();
    // Update OperatorCode vector
    code_vec.emplace_back(code);
  }

  // Create OperatorCode with Custom Operator
  std::set<std::string> custom_code_set = gather_customcode_set(model_recipe);
  if (custom_code_set.size() &&
      builtin_code_map.find(circle::BuiltinOperator_CUSTOM) == builtin_code_map.end())
    builtin_code_map[circle::BuiltinOperator_CUSTOM] = 1;

  for (auto opcode : custom_code_set)
  {
    auto custom_code = flatbuffer_builder->CreateString(opcode);
    circle::OperatorCodeBuilder code_builder{*flatbuffer_builder};
    code_builder.add_builtin_code(circle::BuiltinOperator_CUSTOM);
    code_builder.add_custom_code(custom_code);
    auto code = code_builder.Finish();
    // Update OperatorCode vector
    code_vec.emplace_back(code);
  }

  // Create an Empty Buffer
  //
  // Buffer 0 SHOULD be an empty buffer in TensorFlow Lite model file
  // (Please refer to the comment for Tensor.buffer field in schema)
  {
    circle::BufferBuilder buffer_builder{*flatbuffer_builder};
    buffer_vec.emplace_back(buffer_builder.Finish());
  }

  //
  // Create Main graph
  //
  CookParams cp{buffer_vec, code_vec, subgraph_vec, flatbuffer_builder, builtin_code_map, "main"};

  cook_graph<::circlechef::ModelRecipe>(model_recipe, cp);

  //
  // Create subgraphs if exist
  //
  for (int g = 0; g < model_recipe.graph_size(); ++g)
  {
    const auto &graph = model_recipe.graph(g);

    std::ostringstream stringStream;
    stringStream << "sub_" << (g + 1);

    CookParams cp{buffer_vec,         code_vec,         subgraph_vec,
                  flatbuffer_builder, builtin_code_map, stringStream.str()};

    cook_graph<::circlechef::Graph>(graph, cp);
  }

  // Create "Model" arguments
  auto buffers = flatbuffer_builder->CreateVector(buffer_vec);
  auto operator_codes = flatbuffer_builder->CreateVector(code_vec);
  auto subgraphs = flatbuffer_builder->CreateVector(subgraph_vec);
  auto description = flatbuffer_builder->CreateString("Generated by circlechef");

  // Create "Model"
  circle::ModelBuilder model_builder{*flatbuffer_builder};

  model_builder.add_version(3);
  model_builder.add_operator_codes(operator_codes);
  model_builder.add_subgraphs(subgraphs);
  model_builder.add_description(description);
  model_builder.add_buffers(buffers);

  auto model = model_builder.Finish();

  // Finalize
  ::circle::FinishModelBuffer(*flatbuffer_builder, model);

  // Return "GenerateModel"
  return GeneratedModel{
    std::unique_ptr<GeneratedModelImpl>(new GeneratedModelImpl(std::move(flatbuffer_builder)))};
}

} // namespace circlechef
