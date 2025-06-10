/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

// from tensorflow/lite/core/macros.h
// maximum size of a valid flatbuffer
inline constexpr unsigned int flatbuffer_size_max = 2147483648;

// from tensorflow/compiler/mlir/lite/flatbuffer_import.cc

#include "circle-mlir/import/CircleImport.h"

#include "CircleOperator.h"

#include <circle-mlir/dialect/CircleDialect.h>
#include <circle-mlir/utils/FlatbufferOperator.h>
#include <circle-mlir/utils/ConvertType.h>
#include <circle-mlir/utils/SizeUtils.h>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringExtras.h> // llvm::join
#include <llvm/ADT/STLExtras.h>    // llvm::is_contained
#include <llvm/Support/Endian.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/IR/Matchers.h> // m_Constant
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>

#include <circle_schema/schema_generated.h>

#include <cassert>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#define ASSIGN_OR_RETURN(lhs, rexpr)   \
  auto optvalue = (rexpr);             \
  if (!optvalue.has_value())           \
  {                                    \
    llvm::errs() << "Invalid value\n"; \
    return {};                         \
  }                                    \
  lhs = std::move(optvalue).value()

namespace circle
{

// Node edge.second depends on node edge.first.
using ControlEdge = std::pair<int32_t, int32_t>;
using ControlEdges = std::vector<ControlEdge>;

} // namespace circle

// from tensorflow/lite/experimental/remat/metadata_util.h
namespace circle
{

/// Control dependencies for the model is the collection of control dependencies
/// for its subgraphs.
using ModelControlDependencies = std::vector<ControlEdges>;

} // namespace circle

namespace mlir
{
namespace Circle
{

namespace
{

// from tensorflow/compiler/mlir/lite/offset_buffer.h
inline bool IsValidBufferOffset(const int64_t offset) { return offset > 1; }

// from tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h
constexpr StringRef kModelIndexPathAttr = "model.index_path";
constexpr StringRef kModelExportedNamesAttr = "model.exported_names";

bool IsQuantized(const circle::TensorT &tensor)
{
  return (tensor.quantization != nullptr) && !tensor.quantization->zero_point.empty();
}

// Create the MLIR NamedLoc location corresponding to a given tensor
mlir::Location TensorLoc(const circle::TensorT &tensor, mlir::Builder builder, mlir::Location base)
{
  if (tensor.name.empty())
  {
    return base;
  }
  return mlir::NameLoc::get(builder.getStringAttr(tensor.name), base);
}

// Create the MLIR Location corresponding to a given op. This is an
// experimental/debugging feature and production code should not rely on names
// of intermediate tensors since importer doesn't guarantee to preserve tensor
// names except output tensors.
Location OpLoc(const circle::OperatorT &op,
               const std::vector<std::unique_ptr<circle::TensorT>> &tensors, mlir::Builder builder,
               mlir::Location base)
{
  if (op.outputs.empty())
    return base;

  llvm::SmallVector<Location, 4> locations;
  locations.reserve(op.outputs.size());
  for (auto tensor_index : op.outputs)
  {
    locations.push_back(TensorLoc(*tensors[tensor_index], builder, base));
  }
  return mlir::FusedLoc::get(builder.getContext(), locations);
}

std::optional<mlir::TensorType> GetTensorType(const circle::TensorT &tensor, Builder builder,
                                              bool is_constant = false,
                                              bool is_intermediate = false,
                                              bool get_storage = false)
{
  mlir::Type elem_type = circle::ConvertElementType(tensor.type, builder);
  if (tensor.type == circle::TensorType_VARIANT)
  {
    // TODO implement with mlir::TF::VariantType conversion
    llvm::errs() << "NYI GetTensorType VARIANT.\n";
    return {};
  }
  if (IsQuantized(tensor))
  {
    llvm::errs() << "NYI GetTensorType Quantized.\n";
    return {};
  }

  // Intermediate tensors with calibration value (but not scale and zero points)
  // should return calibrated quantized type.
  if (is_intermediate && tensor.quantization != nullptr && !IsQuantized(tensor))
  {
    llvm::errs() << "NYI GetTensorType Calibrated Quantized.\n";
    return {};
  }

  if (tensor.shape.empty() && (is_constant || tensor.has_rank))
  {
    return RankedTensorType::get({}, elem_type);
  }

  if (!tensor.shape_signature.empty())
  {
    llvm::errs() << "NYI GetTensorType Tensor shape_signature.\n";
    return {};
  }

  if (!tensor.shape.empty())
  {
    llvm::SmallVector<int64_t, 4> shape(tensor.shape.begin(), tensor.shape.end());
    return GetTypeFromTensorShape(shape, elem_type);
  }

  return UnrankedTensorType::get(elem_type);
}

mlir::Operation *ConvertMinMaxToStatsOp(const circle::TensorT &tensor, mlir::OpBuilder b,
                                        mlir::Value res)
{
  // TODO implement
  (void)tensor;
  (void)b;
  (void)res;
  return nullptr;
}

// Returns true if this is a basic LSTM op.
bool IsBasicLSTMOp(circle::BuiltinOptionsUnion op_union)
{
  if (const auto *op = op_union.AsLSTMOptions())
  {
    return op->kernel_type == circle::LSTMKernelType_BASIC;
  }
  else
  {
    return false;
  }
}

// Gets the MLIR op name with the dialect name for the flatbuffer operator.
std::string GetMlirOpName(const circle::OperatorT &op, const circle::OperatorCodeT &op_code)
{
  if (IsBasicLSTMOp(op.builtin_options))
  {
    return std::string("Circle.basic_lstm");
  }
  return GetMlirOpNameFromOpCode(op_code);
}

// The buffers in Circle flatbuffers have their contents stored as a vector of
// bytes that represent host endianness values.
// The read_size parameter is present to allow reading both float16 and float32s
// without a case split.
template <typename T> llvm::SmallVector<mlir::APInt> ReadAsHostEndian(llvm::ArrayRef<uint8_t> bytes)
{
  llvm::SmallVector<mlir::APInt> ret;
  size_t read_size = sizeof(T);
  // NOTE original code used `int` for `bytes_len`
  size_t bytes_len = bytes.size();
  assert(bytes_len % read_size == 0);

  size_t elem_count = bytes_len / read_size;
  ret.reserve(elem_count);

  const char *data_ptr = reinterpret_cast<const char *>(bytes.data());
  for (size_t i = 0; i < elem_count; i++)
  {
    T val = llvm::support::endian::readNext<T, llvm::endianness::native, llvm::support::unaligned>(
      data_ptr);
    ret.push_back(mlir::APInt(sizeof(T) * 8, val));
  }
  return ret;
}

std::optional<mlir::ElementsAttr> ConvertFloatBuffer(mlir::RankedTensorType shaped_type,
                                                     const std::vector<uint8_t> &buffer)
{
  size_t bytes_len = buffer.size();
  mlir::Type elem_type = shaped_type.getElementType();

  // The bytes of floats are stored little-endian.
  switch (elem_type.getIntOrFloatBitWidth())
  {
    // TODO 16
    case 32:
    {
      assert(bytes_len % 4 == 0);
      int elem_count = bytes_len / 4;
      std::vector<float> values;
      values.reserve(elem_count);

      const char *data = reinterpret_cast<const char *>(buffer.data());

      for (int i = 0; i < elem_count; i++)
      {
        uint32_t bit_repr = llvm::support::endian::readNext<uint32_t, llvm::endianness::native,
                                                            llvm::support::unaligned>(data);
        values.push_back(absl::bit_cast<float>(bit_repr));
      }
      auto num = shaped_type.getNumElements();
      return mlir::ElementsAttr(DenseElementsAttr::get(shaped_type, ArrayRef<float>(values)));
    }
  }
  llvm::errs() << "Unsupported bit width: " << elem_type.getIntOrFloatBitWidth() << "\n";
  return {};
}

std::optional<mlir::ElementsAttr> ConvertIntBuffer(mlir::RankedTensorType shaped_type,
                                                   const std::vector<uint8_t> &buffer,
                                                   bool truncate = false)
{
  mlir::Type elem_type = shaped_type.getElementType();
  unsigned bit_width;
  if (auto itype = mlir::dyn_cast<mlir::IntegerType>(elem_type))
  {
    bit_width = itype.getWidth();
  }
  else if (auto qtype = mlir::dyn_cast<mlir::quant::QuantizedType>(elem_type))
  {
    llvm::errs() << "NYI ConvertIntBuffer QuantizedType\n";
    return {};
  }
  else
  {
    llvm::errs() << "Unsupported integer constant type\n";
    return {};
  }

  llvm::SmallVector<mlir::APInt> values;
  switch (bit_width)
  {
    case 8:
      return mlir::ElementsAttr(
        mlir::DenseElementsAttr::get(shaped_type, ArrayRef<uint8_t>(buffer)));
    case 16:
      values = ReadAsHostEndian<uint16_t>(buffer);
      break;
    case 32:
      values = ReadAsHostEndian<uint32_t>(buffer);
      break;
    case 64:
      values = ReadAsHostEndian<uint64_t>(buffer);
      break;
    default:
      llvm::errs() << "Cannot handle bit width " << bit_width << "\n";
      return {};
  }

  if (truncate)
  {
    llvm::errs() << "NYI ConvertIntBuffer truncate.\n";
    return {};
  }

  return mlir::ElementsAttr(mlir::DenseElementsAttr::get(shaped_type, values));
}

std::optional<Operation *> BuildExternalConstOp(const circle::TensorT &tensor, int32_t buffer_index,
                                                mlir::OpBuilder builder, mlir::Location loc)
{
  // TODO implement
  (void)tensor;
  (void)buffer_index;
  (void)builder;
  (void)loc;
  llvm::errs() << "NYI BuildExternalConstOp\n";
  assert(false); // assert is used to know when this is used for some model
  return {};
}

std::optional<Operation *> BuildConstOp(const circle::TensorT &tensor,
                                        const std::vector<uint8_t> &buffer, bool is_variable,
                                        mlir::OpBuilder builder, mlir::Location loc,
                                        bool use_stablehlo_constant)
{
  if (tensor.sparsity != nullptr)
  {
    // TODO support sparsity
    llvm::errs() << "NYI BuildConstOp sparse\n";
    return {};
  }

  if (is_variable)
  {
    // TODO support variable
    llvm::errs() << "NYI BuildConstOp variable\n";
    return {};
  }

  ASSIGN_OR_RETURN(auto type, GetTensorType(tensor, builder,
                                            /*is_constant=*/true,
                                            /*is_intermediate=*/false,
                                            /*get_storage=*/true));
  auto shaped_type = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!shaped_type)
  {
    llvm::errs() << "Constant doesn't have a shape\n";
    return {};
  }

  mlir::ElementsAttr value;
  if (IsQuantized(tensor))
  {
    // TODO support quantized
    llvm::errs() << "NYI BuildConstOp quantized\n";
    return {};
  }

  auto elem_type = shaped_type.getElementType();
  if (auto float_type = mlir::dyn_cast<mlir::FloatType>(elem_type))
  {
    ASSIGN_OR_RETURN(value, ConvertFloatBuffer(shaped_type, buffer));
  }
  else if (mlir::isa<mlir::IntegerType>(elem_type))
  {
    ASSIGN_OR_RETURN(value, ConvertIntBuffer(shaped_type, buffer));
  }
  // TODO support StringType (TensorType_STRING)
  // TODO support ComplexType
  else
  {
    llvm::errs() << "Constant of unsupported type\n";
    return {};
  }

  if (use_stablehlo_constant)
  {
    // TODO support stablehlo
    llvm::errs() << "NYI BuildConstOp stablehlo\n";
    return {};
  }
  auto op = builder.create<mlir::Circle::ConstOp>(loc, value);
  return op.getOperation();
}

// TODO(krzysd) Handle function calls
std::optional<Operation *>
ConvertOp(const circle::OperatorT &op, const std::vector<Value> &vals_map,
          const std::vector<mlir::TensorType> &intermediate_types, mlir::Value optional_arg_marker,
          const std::vector<std::unique_ptr<circle::OperatorCodeT>> &op_codes,
          const std::vector<std::string> &func_names,
          const std::vector<std::unique_ptr<circle::TensorT>> &tensors, mlir::Location loc,
          mlir::OpBuilder builder, const circle::Model *model_ptr)
{
  const circle::OperatorCodeT &op_code = *op_codes.at(op.opcode_index);

  const std::string op_name = GetMlirOpName(op, op_code);

  mlir::OperationState op_state(loc, op_name);

  for (auto input_num : op.inputs)
  {
    if (input_num == -1)
    {
      assert(optional_arg_marker != nullptr);
      op_state.addOperands({optional_arg_marker});
    }
    else
    {
      op_state.addOperands({vals_map.at(input_num)});
    }
  }

  for (auto output_num : op.outputs)
  {
    auto &tensor = *tensors.at(output_num);
    auto type_or_err = GetTensorType(tensor, builder);
    if (!type_or_err.has_value())
    {
      return {};
    }
    auto type = std::move(type_or_err).value();

    if (op_name == "Circle.quantize")
    {
      // Special case for quantize: return type must also be in qtype attribute
      op_state.addAttribute("qtype", mlir::TypeAttr::get(type));
    }
    else if (op_name == "Circle.reshape" && op_state.operands.size() == 1)
    {
      // Special case for reshape: the second op is optional in the old
      // converter and kernel, so we create the second operand, which is
      // required by the new converter, from the reshape op's option.
      auto new_shape = op.builtin_options.AsReshapeOptions()->new_shape;
      auto shape_type = GetTypeFromTensorShape({static_cast<int64_t>(new_shape.size())},
                                               builder.getIntegerType(32));

      mlir::SmallVector<mlir::Attribute, 4> shape;
      for (auto s : new_shape)
      {
        shape.push_back(builder.getI32IntegerAttr(ConvertToCircleSize(s)));
      }
      auto output_shape = DenseElementsAttr::get(shape_type, shape);
      auto shape_op = builder.create<mlir::Circle::ConstOp>(loc, output_shape);
      op_state.addOperands({shape_op});
    }

    op_state.addTypes({type});
  }

  // While the last several tensors could be optional tensors for an circle op, the
  // number of input operands could vary. Gets the min/max number of
  // operands from circle op name.
  // Also, since the above code special-handles the `circle.reshape` op and add an
  // additional input, we put these function block here.
  llvm::MinMax input_min_max = mlir::OperandNumbersMinMax(op_name);
  int input_max_num = input_min_max.Max;
  int op_input_num = op_state.operands.size();
  if (input_max_num != 0 && input_max_num > op_input_num)
  {
    // If the number of current inputs is less than the op definition, fill in
    // with `none` value,
    llvm::SmallVector<mlir::Value, 4> none_operands(
      input_max_num - op_input_num,
      builder.create<mlir::Circle::NoValueOp>(loc, builder.getNoneType(), builder.getUnitAttr()));
    op_state.addOperands(llvm::ArrayRef<mlir::Value>(none_operands));
  }

  // TODO support lstm
  // TODO support while
  // TODO support unidirectional_sequence_lstm
  if (op_name == "Circle.reshape")
  {
    // Flattern reshape ops when more than one dimension shape operand is given.
    mlir::DenseIntElementsAttr shape_attr;
    if (matchPattern(op_state.operands[1], m_Constant(&shape_attr)))
    {
      auto shape_ty = mlir::dyn_cast<RankedTensorType>(op_state.operands[1].getType());
      if (shape_ty != nullptr && shape_ty.hasRank() && shape_ty.getRank() > 1)
      {
        llvm::SmallVector<mlir::Attribute, 4> shape;
        int32_t dim_size = 0;
        for (const auto &dim : llvm::enumerate(shape_attr.getValues<llvm::APInt>()))
        {
          shape.push_back(
            builder.getI32IntegerAttr(ConvertToCircleSize(dim.value().getSExtValue())));
          ++dim_size;
        }
        auto shape_type =
          GetTypeFromTensorShape({static_cast<int32_t>(dim_size)}, builder.getIntegerType(32));
        auto output_shape = mlir::DenseElementsAttr::get(shape_type, shape);
        auto shape_op = builder.create<mlir::Circle::ConstOp>(loc, output_shape);
        op_state.operands[1] = shape_op;
      }
    }
  }
  // TODO check why stablehlo is used
  /*
  if (op_name == "stablehlo.reduce" || op_name == "stablehlo.reduce_window" ||
      op_name == "stablehlo.sort" || op_name == "stablehlo.scatter") {
    op_state.addRegion();
  }
  if (op_name == "stablehlo.while") {
    op_state.addRegion();
    op_state.addRegion();
  }
  */

  llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
  auto builtin_code = circle::GetBuiltinCode(&op_code);
  if (builtin_code == circle::BuiltinOperator_CUSTOM)
  {
    auto status = true;

    std::vector<uint8_t> custom_options;

    // TODO enable to support large custom options
    assert(not IsValidBufferOffset(op.large_custom_options_offset));
    /*
    if (IsValidBufferOffset(op.large_custom_options_offset))
    {
      custom_options.resize(op.large_custom_options_size);
      memcpy(custom_options.data(),
             reinterpret_cast<const uint8_t *>(model_ptr) + op.large_custom_options_offset,
             op.large_custom_options_size);
    }
    else
    */
    {
      custom_options = op.custom_options;
    }

    status =
      mlir::CustomOptionsToAttributes(op_code.custom_code, custom_options, builder, loc, &attrs);
    if (!status)
    {
      return {};
    }
  }
  else
  {
    mlir::BuiltinOptionsToAttributes(op.builtin_options, builder, attrs);
    // TODO enable BuiltinOptions2
    // mlir::BuiltinOptions2ToAttributes(op.builtin_options_2, builder, attrs);
  }
  op_state.addAttributes(attrs);

  // TODO handle CallOnce, If and While subgraphs
  // TODO handle StableHLO

  (void)intermediate_types;
  (void)func_names;
  (void)model_ptr;

  return builder.create(op_state);
}

// Returns indices of the given tensors in the subgraph. Returns error if a
// tensor name cannot be found in the subgraph.
std::optional<std::vector<int>> GetTensorIndices(const circle::SubGraphT &subgraph,
                                                 const std::vector<std::string> &tensor_names)
{
  throw std::runtime_error("NYI GetTensorIndices");
  // NOTE enable codes when necessary
  /*
  absl::flat_hash_map<std::string, int> name_to_index;

  for (const auto &index_and_tensor : llvm::enumerate(subgraph.tensors))
  {
    name_to_index[index_and_tensor.value()->name] = index_and_tensor.index();
  }

  std::vector<int> indices;
  indices.reserve(tensor_names.size());

  for (const auto &name : tensor_names)
  {
    auto found = name_to_index.find(name);
    if (found != name_to_index.end())
    {
      indices.push_back(found->second);
    }
    else
    {
      llvm::errs() << "Could not find tensor in subgraph: " << name << "\n";
      return {};
    }
  }

  return indices;
  */
}

// Given a list of tensor indices, returns true if any of the tensors have
// non-empty name strings.
bool HasNonEmptyNames(const circle::SubGraphT &subgraph, llvm::ArrayRef<int32_t> indices)
{
  return llvm::any_of(indices, [&](int i) { return !subgraph.tensors.at(i)->name.empty(); });
}

// Given a list of tensor indices, returns a array of strings of tensor names
// wrapped in a NamedAttribute.
mlir::Attribute BuildEntryFunctionAttribute(const circle::SubGraphT &subgraph,
                                            mlir::Builder *builder, llvm::ArrayRef<int32_t> indices)
{
  auto tensor_names = llvm::map_range(indices, [&](int i) { return subgraph.tensors.at(i)->name; });
  // NOTE single line "argumments(tensor_names.begin(), tensor_names.end());" gives corrupted names
  auto names_vect = llvm::to_vector(tensor_names);
  llvm::SmallVector<StringRef> argumments;
  for (auto &item : names_vect)
    argumments.push_back(item);
  return builder->getStrArrayAttr(argumments);
}

// We want to adjust the func op according to some cross ops information.
std::optional<mlir::func::FuncOp> PostProcessFuncOp(mlir::func::FuncOp func)
{
  OpBuilder builder(func);
  // TODO walk with QConstOp when ready
  return func;
}

// There are control nodes at each end of each control edge. For each of them,
// we store the source vertices of the incoming edges (if any) and the control
// node's output token. To improve testability, we use an ordered set for the
// source vertices.
struct ControlNodeDesc
{
  std::set<int> incoming;
  std::optional<mlir::Value> outgoing;
};

using ControlNodes = llvm::DenseMap<int, ControlNodeDesc>;

// Helper function: After op has been emitted as the MLIR representation of
// a subgraph's operators[op_index], check *control_nodes whether it needs to be
// wrapped in a ControlNode because it's at either end of a control edge from
// the metadata. If it is, wrap it in a ControlNode, store the resulting
// ControlType token in *control_nodes, and return the non-ControlType (i.e.,
// tensor) results.  If it isn't, just return the original operator's results.
mlir::ResultRange MaybeWrapInControlNode(mlir::Operation *op, OpBuilder op_builder, int op_index,
                                         Location op_loc, ControlNodes *control_nodes)
{
  const ControlNodes::iterator maybe_control_node = control_nodes->find(op_index);
  if (maybe_control_node == control_nodes->end())
  {
    return op->getResults();
  }
  // TODO enable control node
  (void)op_builder;
  (void)op_loc;
  llvm::errs() << "NYI MaybeWrapInControlNode\n";
  assert(false); // assert is used to know when this is used for some model
  return op->getResults();
}

// Build a FuncOp from a circle SubGraph
// The buffers are directly taken
// from the deserialized flatbuffer as we do not have the type information to
// interpret them until this point. The base_loc parameter is the location of
// the flatbuffer as a whole (usually a file). If ordered_output_arrays is not
// empty, then the imported mlir function will only return nodes in
// ordered_output_arrays in the same order. If signature is not null, then the
// inputs/outputs in signature will be attached to the FuncOp.
std::optional<mlir::func::FuncOp>
ConvertSubgraph(const circle::SubGraphT &subgraph, llvm::StringRef name,
                const std::vector<std::unique_ptr<circle::OperatorCodeT>> &op_codes,
                const std::vector<std::string> &func_names,
                const std::vector<std::unique_ptr<circle::BufferT>> &buffers, Location base_loc,
                mlir::Builder builder, bool is_entry_point, bool use_external_constant,
                const std::vector<std::string> &ordered_input_arrays,
                const std::vector<std::string> &ordered_output_arrays,
                bool experimental_prune_unreachable_nodes_unconditionally,
                const circle::SignatureDefT *signature, const circle::ControlEdges &control_edges,
                const circle::Model *model_ptr, bool use_stablehlo_constant)
{
  // Populate from metadata.
  ControlNodes control_nodes;
  for (const auto [from, to] : control_edges)
  {
    control_nodes.try_emplace(from);
    control_nodes[to].incoming.insert(from);
  }

  llvm::SmallVector<mlir::Type, 2> ret_types;
  llvm::SmallVector<mlir::Type, 4> input_types;

  auto func_loc = mlir::NameLoc::get(builder.getStringAttr(name), base_loc);
  std::vector<int> func_inputs = subgraph.inputs;
  if (is_entry_point && !ordered_input_arrays.empty())
  {
    if (!experimental_prune_unreachable_nodes_unconditionally)
    {
      // TODO(b/149922113): Resolve input-arrays/pruning flags interaction.
      llvm::errs() << "input-arrays should be used with experimental pruning flag\n";
      return {};
    }
    ASSIGN_OR_RETURN(func_inputs, GetTensorIndices(subgraph, ordered_input_arrays));
  }

  for (int input : func_inputs)
  {
    auto &tensor = *subgraph.tensors.at(input);
    auto type_or_err = GetTensorType(tensor, builder);
    if (!type_or_err.has_value())
    {
      llvm::errs() << "Error reading argument types\n";
      return {};
    }
    auto type = std::move(type_or_err).value();
    input_types.push_back(type);
  }

  llvm::SmallVector<bool, 16> is_op_output(subgraph.tensors.size(), false);
  for (auto &op : subgraph.operators)
  {
    for (auto output : op->outputs)
    {
      is_op_output[output] = true;
    }
  }

  std::vector<int> func_outputs = subgraph.outputs;
  if (is_entry_point && !ordered_output_arrays.empty())
  {
    ASSIGN_OR_RETURN(func_outputs, GetTensorIndices(subgraph, ordered_output_arrays));
  }

  for (auto output : func_outputs)
  {
    const bool is_func_input =
      std::find(func_inputs.begin(), func_inputs.end(), output) != func_inputs.end();
    bool is_constant = !is_op_output[output] && !is_func_input;

    auto type_or_err = GetTensorType(*subgraph.tensors.at(output), builder, is_constant);
    if (!type_or_err.has_value())
    {
      llvm::errs() << "Error reading return types\n";
      return {};
    }
    auto type = std::move(type_or_err).value();
    ret_types.push_back(type);
  }
  auto func_type = builder.getFunctionType(input_types, ret_types);

  // Construct function object
  auto func = mlir::func::FuncOp::create(func_loc, name, func_type, /* attrs= */ {});
  func.addEntryBlock();
  auto &body = func.getBody();
  mlir::OpBuilder op_builder{body};

  std::vector<mlir::Value> vals_map(subgraph.tensors.size(), nullptr);
  Value maybe_optional_arg_marker = nullptr;

  // Get or construct MLIR values for each input
  for (int i = 0, e = func_inputs.size(); i < e; i++)
  {
    auto input_tensor = func_inputs[i];
    const auto &tensor = *subgraph.tensors.at(input_tensor);
    auto loc = TensorLoc(tensor, builder, base_loc);
    if (vals_map[input_tensor])
    {
      llvm::errs() << "Duplicate input arguments\n";
      return {};
    }
    mlir::Value input_value = func.getArgument(i);

    // If the `tensor` has min/max and doesn't have scale/zero_point
    // information, a stats op is created to use the input_value, then the
    // `tensor` should be mapped to the result of this new stats op.
    if (auto stats_op = ConvertMinMaxToStatsOp(tensor, op_builder, input_value))
    {
      vals_map[input_tensor] = stats_op->getResult(0);
    }
    else
    {
      vals_map[input_tensor] = input_value;
    }
  }

  // Set entry_function attribute
  if (is_entry_point)
  {
    // NOTE we need attribute something like this, in MLIR
    //    attributes {input_names = ["input_1", "input_2"], output_names = ["output_1"]}
    if (HasNonEmptyNames(subgraph, func_inputs))
    {
      auto names = BuildEntryFunctionAttribute(subgraph, &builder, func_inputs);
      func->setAttr("input_names", names);
    }
    if (HasNonEmptyNames(subgraph, func_outputs))
    {
      auto names = BuildEntryFunctionAttribute(subgraph, &builder, func_outputs);
      func->setAttr("output_names", names);
    }
  }
  else
  {
    func.setPrivate();
  }

  // Set signature on function.
  if (signature)
  {
    throw std::runtime_error("'signature' is expected to be nullptr");
    // TODO revive SetSignature
    // SetSignature(func, signature, subgraph.tensors);
  }

  absl::flat_hash_set<const circle::OperatorT *> pruned_subgraph_ops;
  if (experimental_prune_unreachable_nodes_unconditionally)
  {
    throw std::runtime_error("'experimental_prune...' is expected to be false");
    // TODO prune subgraph for experimental_prune_unreachable_nodes_unconditionally
    // ASSIGN_OR_RETURN(pruned_subgraph_ops, PruneSubgraph(subgraph, func_inputs, func_outputs));
  }

  // Construct MLIR operators from Circle operators
  for (const auto &it : llvm::enumerate(subgraph.operators))
  {
    auto &op = it.value();

    if (experimental_prune_unreachable_nodes_unconditionally && !pruned_subgraph_ops.contains(op))
    {
      continue;
    }

    for (auto input_num : op->inputs)
    {
      // The operators in a graph are topologically sorted
      // and so if no previous operation has produced a tensor
      // it must be a constant.
      if (input_num == -1)
      {
        if (maybe_optional_arg_marker == nullptr)
        {
          maybe_optional_arg_marker = op_builder
                                        .create<mlir::Circle::NoValueOp>(
                                          base_loc, builder.getNoneType(), builder.getUnitAttr())
                                        .getResult();
        }
      }
      else if (!vals_map.at(input_num))
      {
        auto &const_tensor = *subgraph.tensors[input_num];
        auto const_loc = TensorLoc(const_tensor, builder, base_loc);
        std::optional<Operation *> op_or_err;
        std::vector<uint8_t> buffer;
        // TODO enable to support external tensor files
        /*
        // Check if constant tensor is stored outside of the flatbuffers.
        if (IsValidBufferOffset(buffers[const_tensor.buffer]->offset))
        {
          const uint8_t *file_begin_ptr =
            reinterpret_cast<const uint8_t *>(model_ptr->allocation()->base());
          buffer = std::vector<uint8_t>(file_begin_ptr + buffers[const_tensor.buffer]->offset,
                                        file_begin_ptr + buffers[const_tensor.buffer]->offset +
                                          buffers[const_tensor.buffer]->size);

          auto shape = const_tensor.shape;
        }
        else
        */
        {
          buffer = buffers[const_tensor.buffer]->data;
        }
        op_or_err =
          use_external_constant
            ? BuildExternalConstOp(const_tensor, const_tensor.buffer, op_builder, const_loc)
            : BuildConstOp(const_tensor, buffer, const_tensor.is_variable, op_builder, const_loc,
                           use_stablehlo_constant);
        if (!op_or_err.has_value())
        {
          llvm::errs() << "Failed to create ConstOp\n";
          return {};
        }
        vals_map[input_num] = op_or_err.value()->getResult(0);
      }
    }

    // Intermediate tensors for LSTMs are used to carry quantization range
    // in their types, so we only need and extract their types.
    std::vector<mlir::TensorType> intermediate_types;
    intermediate_types.reserve(5);
    for (auto intermediate : op->intermediates)
    {
      ASSIGN_OR_RETURN(auto type, GetTensorType(*subgraph.tensors[intermediate], builder,
                                                /*is_constant=*/false, /*is_intermediate=*/true));
      intermediate_types.emplace_back(type);
    }

    auto op_loc = OpLoc(*op, subgraph.tensors, builder, base_loc);

    // If there's an optional argument, maybe_optional_arg_marker has been set
    // to a valid Value
    ASSIGN_OR_RETURN(auto *mlir_op, ConvertOp(*op, vals_map, intermediate_types,
                                              maybe_optional_arg_marker, op_codes, func_names,
                                              subgraph.tensors, op_loc, op_builder, model_ptr));

    // Add the results to the value maps. There are two cases: 1. the result
    // tensor does not have min/max values, the original op result is used
    // directly; 2. the result tensor has some min/max values, a stats op is
    // created, then the result of the stats op is used.
    for (const auto &pair : llvm::enumerate(
           MaybeWrapInControlNode(mlir_op, op_builder, it.index(), op_loc, &control_nodes)))
    {
      int output_tensor_index = op->outputs[pair.index()];
      auto &tensor = *subgraph.tensors[output_tensor_index];
      if (auto stats_op = ConvertMinMaxToStatsOp(tensor, op_builder, pair.value()))
      {
        vals_map[output_tensor_index] = stats_op->getResult(0);
      }
      else
      {
        vals_map[output_tensor_index] = pair.value();
      }
    }
  }

  // Construct return values
  llvm::SmallVector<Value, 4> return_operands;
  for (auto index : func_outputs)
  {
    if (!vals_map.at(index))
    {
      auto &const_tensor = *subgraph.tensors[index];
      auto const_loc = TensorLoc(const_tensor, builder, base_loc);
      std::optional<Operation *> op_or_err;
      std::vector<uint8_t> buffer;
      // TODO enable to support external tensor files
      /*
      // Check if constant tensor is stored outside of the flatbuffers.
      if (IsValidBufferOffset(buffers[const_tensor.buffer]->offset))
      {
        const uint8_t *file_begin_ptr = reinterpret_cast<const uint8_t *>(model_ptr);

        buffer = std::vector<uint8_t>(file_begin_ptr + buffers[const_tensor.buffer]->offset,
                                      file_begin_ptr + buffers[const_tensor.buffer]->offset +
                                      buffers[const_tensor.buffer]->size);

        auto shape = const_tensor.shape;
      }
      else
      */
      {
        buffer = buffers[const_tensor.buffer]->data;
      }
      op_or_err = use_external_constant
                    ? BuildExternalConstOp(const_tensor, const_tensor.buffer, op_builder, const_loc)
                    : BuildConstOp(const_tensor, buffer, const_tensor.is_variable, op_builder,
                                   const_loc, use_stablehlo_constant);
      if (!op_or_err.has_value())
      {
        llvm::errs() << "Failed to create ConstOp\n";
        return {};
      }
      vals_map[index] = op_or_err.value()->getResult(0);
    }
    return_operands.push_back(vals_map[index]);
  }

  op_builder.create<mlir::func::ReturnOp>(base_loc, return_operands);

  return PostProcessFuncOp(func);
}

std::string SubgraphName(bool set_implicit_main_func, unsigned index,
                         const circle::SubGraphT &subgraph)
{
  if (index == 0 && set_implicit_main_func)
  {
    return "main_graph";
  }
  if (subgraph.name.empty())
  {
    return llvm::formatv("fn_{0}", index).str();
  }
  return subgraph.name;
}

} // namespace

mlir::OwningOpRef<mlir::ModuleOp>
FlatBufferToMlir(absl::string_view buffer, mlir::MLIRContext *context, mlir::Location base_loc,
                 bool use_external_constant, const std::vector<std::string> &ordered_input_arrays,
                 const std::vector<std::string> &ordered_output_arrays,
                 bool experimental_prune_unreachable_nodes_unconditionally)
{
  // Only run validator on models less than 2GB
  if (buffer.length() < flatbuffer_size_max)
  {
    flatbuffers::Verifier base_verifier(reinterpret_cast<const uint8_t *>(buffer.data()),
                                        buffer.size());
    if (!circle::VerifyModelBuffer(base_verifier))
    {
      llvm::errs() << "The model is not a valid Flatbuffer buffer.\n";
      return nullptr;
    }
  }

  auto circle_model = circle::GetModel(buffer.data());
  std::unique_ptr<circle::ModelT> model(circle_model->UnPack());

  auto builder = Builder(context);

  circle::ModelControlDependencies model_control_dependencies(model->subgraphs.size());

  bool use_stablehlo_constant = false;

  // TODO iterate model->metadata

  std::vector<std::string> func_names;
  for (auto &subgraph : model->subgraphs)
  {
    func_names.push_back(subgraph->name);
  }

  auto module = mlir::ModuleOp::create(base_loc);

  // We currently don't use this to make decisions, but we could
  // use it in exports or if there are breaking changes
  module->setAttr("circle.schema_version", builder.getI32IntegerAttr(model->version));
  if (!model->description.empty())
  {
    module->setAttr("circle.description", builder.getStringAttr(model->description));
  }

  absl::flat_hash_map<uint32_t, circle::SignatureDefT *> subgraph_to_signature_map;
  for (int i = 0; i < model->signature_defs.size(); i++)
  {
    auto *signature_def = model->signature_defs[i].get();
    const uint32_t subgraph_index = signature_def->subgraph_index;
    subgraph_to_signature_map[subgraph_index] = signature_def;
  }

  const bool set_implicit_main_func = subgraph_to_signature_map.size() <= 1;
  for (const auto &e : llvm::enumerate(model->subgraphs))
  {
    auto &subgraph = e.value();
    std::string name = SubgraphName(set_implicit_main_func, e.index(), *subgraph);
    uint32_t subgraph_index = static_cast<uint32_t>(e.index());
    bool is_entry_point =
      set_implicit_main_func ? e.index() == 0 : subgraph_to_signature_map.contains(subgraph_index);
    circle::SignatureDefT *signature_def = subgraph_to_signature_map.contains(subgraph_index)
                                             ? subgraph_to_signature_map.at(subgraph_index)
                                             : nullptr;

    auto func_or_error = ConvertSubgraph(
      *subgraph, name, model->operator_codes, func_names, model->buffers, base_loc, builder,
      is_entry_point, use_external_constant, ordered_input_arrays, ordered_output_arrays,
      experimental_prune_unreachable_nodes_unconditionally, signature_def,
      model_control_dependencies[subgraph_index], circle_model, use_stablehlo_constant);

    if (!func_or_error.has_value())
    {
      llvm::errs() << "Could not translate function '" << subgraph->name << "'\n";
      return nullptr;
    }
    // NOTE std::move is from TF for 'StatusOr' but here we use 'std::optional'
    // TODO revise this for any issues that may happen
    module.push_back(std::move(func_or_error).value());
  }

  return mlir::OwningOpRef<mlir::ModuleOp>(module);
}

} // namespace Circle
} // namespace mlir
