/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

// from tensorflow/compiler/mlir/lite/flatbuffer_export.cc

#define DEBUG_TYPE "o2c"
#include <llvm/Support/Debug.h>

#include "circle-mlir/export/CircleExport.h"
#include "OpOrArgNameMapper.h"

#include <circle-mlir/dialect/CircleDialect.h>
#include <circle-mlir/utils/FlatbufferOperator.h>
#include <circle_schema/schema_generated.h>

#define CIRCLE_SCHEMA_VERSION (0)
#define kCircleOptionalTensor (-1)

const int64_t kCIRDynamicSize = -1;

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/strings/str_cat.h>

#include <mlir/Dialect/Arith/IR/Arith.h> // from @llvm-project

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

// limitation of current flatbuffers file size
inline constexpr uint64_t FLATBUFFERS_SIZE_MAX = 2147483648UL;

template <typename T> using BufferOffset = flatbuffers::Offset<T>;

template <typename T> using VectorBufferOffset = flatbuffers::Offset<flatbuffers::Vector<T>>;

using CustomOptionsOffset = VectorBufferOffset<uint8_t>;

// Use initial buffer size in flatbuffer builder to be same as the initial size
// used by the TOCO export. (It does not explain rationale for this choice.)
constexpr size_t kInitialBufferSize = 10240;

// Set `isSigned` to false if the `type` is an 8-bit unsigned integer type.
// Since circle doesn't support unsigned for other types, returns error if
// `isSigned` is set to false for other types.
static circle::TensorType GetCircleType(mlir::Type type, bool is_signed = true)
{
  if (!is_signed && type.isSignlessInteger(8))
  {
    return circle::TensorType_UINT8;
  }
  if (!is_signed)
  {
    throw std::runtime_error("'isSigned' can only be set for 8-bits integer type");
  }

  if (type.isF32())
  {
    return circle::TensorType_FLOAT32;
  }
  else if (type.isF16())
  {
    return circle::TensorType_FLOAT16;
  }
  else if (type.isF64())
  {
    return circle::TensorType_FLOAT64;
  }
  else if (auto itype = mlir::dyn_cast<mlir::IntegerType>(type))
  {
    switch (itype.getWidth())
    {
      case 1:
        return circle::TensorType_BOOL;
      case 4:
        if (itype.isUnsigned())
        {
          throw std::runtime_error("Unsupported 4bit unsigned int type");
        }
        else
        {
          return circle::TensorType_INT4;
        }
      case 8:
        return itype.isUnsigned() ? circle::TensorType_UINT8 : circle::TensorType_INT8;
      case 16:
        return itype.isUnsigned() ? circle::TensorType_UINT16 : circle::TensorType_INT16;
      case 32:
        return itype.isUnsigned() ? circle::TensorType_UINT32 : circle::TensorType_INT32;
      case 64:
        return itype.isUnsigned() ? circle::TensorType_UINT64 : circle::TensorType_INT64;
    }
  }
  // TODO support quantized

  // Circle export fills FLOAT32 for unknown data types. Returning an error
  // for now for safety and this could be revisited when required.
  throw std::runtime_error("Unsupported type");
}

static bool IsConst(mlir::Operation *op)
{
  return llvm::isa<mlir::func::ConstantOp, mlir::arith::ConstantOp, mlir::Circle::ConstOp,
                   mlir::Circle::NoValueOp>(op);
  // TODO add QConstOp, SparseConstOp, SparseQConstOp
}

namespace mlir
{
namespace Circle
{

namespace
{

// Helper struct that wraps inputs/outputs of a single SignatureDef.
struct SignatureDefData
{
  // Note, we are using maps here to make order deterministic
  // for easily testing only.

  // Inputs defined in the signature def mapped to tensor names.
  std::map<std::string, std::string> inputs;
  // Outputs defined in the signature def mapped to tensor names.
  std::map<std::string, std::string> outputs;
  // Signature key.
  std::string signature_key;
  // Subgraph index.
  uint32_t subgraph_index;
};

// Translates an MLIR module in Circle dialect to Circle FlatBuffer.
class Translator
{
public:
  // Translates the given MLIR module into Circle FlatBuffer format and returns
  // the serialized output. Returns std::nullopt on unsupported, invalid inputs or
  // internal error.
  static std::optional<std::string> Translate(mlir::ModuleOp module,
                                              OpOrArgNameMapper *op_or_arg_name_mapper,
                                              const std::map<std::string, std::string> &metadata);

private:
  enum class OpType : char
  {
    kCircleBuiltin,
    kCustomOp
  };

  explicit Translator(mlir::ModuleOp module, OpOrArgNameMapper *op_or_arg_name_mapper,
                      const std::map<std::string, std::string> &metadata)
    : module_(module), name_mapper_(*op_or_arg_name_mapper), builder_(kInitialBufferSize),
      metadata_(metadata)
  {
    // The first buffer must be empty according to the schema definition.
    empty_buffer_ = circle::CreateBuffer(builder_);
    buffers_.push_back(empty_buffer_);

    enabled_op_types_.emplace(OpType::kCircleBuiltin);
    enabled_op_types_.emplace(OpType::kCustomOp);

    circle_dialect_ = module.getContext()->getOrLoadDialect<mlir::Circle::CIRDialect>();
  }

  std::optional<std::string> TranslateInternal();
  std::optional<std::string> FinalizeWithExtendedBuffer();

  // Returns Circle buffer populated with constant value if the operation is
  // Circle constant operation. Otherwise, returns an empty buffer. Emits error
  // and returns std::nullopt on failure.
  std::optional<BufferOffset<circle::Buffer>> BuildBuffer(mlir::Value value, const uint32_t index);

  // Builds VariantType from the given element type. Returns std::nullopt if
  // failure. Returns empty vector if the element type is not VariantType or
  // there is empty TensorType in the VariantType.
  std::optional<std::vector<BufferOffset<circle::VariantSubType>>>
  BuildVariantType(mlir::Type element_type);

  // Builds Circle tensor from the given value. `buffer_idx` is index of the
  // corresponding buffer. Emits error and returns std::nullopt on failure.
  std::optional<BufferOffset<circle::Tensor>>
  BuildTensor(mlir::Value value, const std::string &name, unsigned buffer_idx,
              const std::optional<BufferOffset<circle::QuantizationParameters>> &quant_parameters);

  BufferOffset<circle::Operator> BuildCustomOperator(Operation *inst, mlir::Circle::CustomOp op,
                                                     const std::vector<int32_t> &operands,
                                                     const std::vector<int32_t> &results);

  // TODO build If, While, CallOnce, NumericVerify, ...

  // Returns opcode index for op identified by the op_name, if already
  // available. Otherwise, creates a new OperatorCode using the given `builtin`
  // operator and associates it with `op_name`.
  uint32_t GetOpcodeIndex(const std::string &op_name, circle::BuiltinOperator builtin);

  // Builds operator for the given operation with specified operand and result
  // tensor indices. Emits an error and returns std::nullopt on failure.
  std::optional<BufferOffset<circle::Operator>>
  BuildOperator(mlir::Operation *inst, std::vector<int32_t> operands,
                const std::vector<int32_t> &results, const std::vector<int32_t> &intermediates);

  // Build a subgraph with a given name out of the region either corresponding
  // to a function's body or while op. Modifies *region by calling
  // ExtractControlEdges.
  std::optional<BufferOffset<circle::SubGraph>>
  BuildSubGraph(const std::string &name, mlir::Region *region, const int index);

  // Encodes metadata dictionary attribute of the module to the
  // metadata section in the final model.
  std::optional<VectorBufferOffset<BufferOffset<circle::Metadata>>> CreateMetadataVector();

  // Builds and returns list of tfl.SignatureDef sections in the model.
  std::optional<VectorBufferOffset<BufferOffset<circle::SignatureDef>>>
  CreateSignatureDefs(const std::vector<SignatureDefData> &signature_defs);

  // Uses the onnx.EntryPoint(?) attribute (if set) to initialize the op to name
  // mapping.
  void InitializeNamesFromAttribute(mlir::func::FuncOp fn, bool *has_input_attr);

  // check if Flatbuffer builder can no longer hold the given amount of the data
  inline bool IsModelBiggerThan2GB(const uint64_t data_size)
  {
    return FLATBUFFERS_SIZE_MAX < data_size + builder_.GetSize();
  }

  // Returns a unique name for `val`.
  std::string UniqueName(mlir::Value val);

private:
  mlir::ModuleOp module_;

  OpOrArgNameMapper &name_mapper_;

  flatbuffers::FlatBufferBuilder builder_;
  BufferOffset<circle::Buffer> empty_buffer_;

  std::vector<BufferOffset<circle::Buffer>> buffers_;
  // Maps subgraph index and tensor name in the graph to the tensor index.
  absl::flat_hash_map<int, absl::flat_hash_map<std::string, int>> tensor_index_map_;

  // Maps op name to index of the corresponding OperatorCode in opcodes_ vector.
  absl::flat_hash_map<std::string, uint32_t> opcode_index_map_;
  std::vector<BufferOffset<circle::OperatorCode>> opcodes_;

  // Maps function name to index of the corresponding subgraph in the FlatBuffer model.
  absl::flat_hash_map<std::string, int> subgraph_index_map_;
  absl::flat_hash_set<OpType> enabled_op_types_;

  // Points to Circle dialects. nullptr if the dialect is not registered.
  const mlir::Dialect *circle_dialect_ = nullptr;

  // Resource ops to provide warning messages.
  std::map<std::string, std::set<std::string>> resource_ops_;

  // Set of saved model tags, if any.
  const std::unordered_set<std::string> saved_model_tags_;
  // Map of key value pairs of metadata to export.
  const std::map<std::string, std::string> metadata_;
  // A mapping table to mlir::Operation objects for CIR subgraph and operator
  // index in a flatbuffer.
  std::vector<std::vector<mlir::Operation *>> subgraph_op_inst_map_;

  // flag to use extended Buffer mode for file size > 2G
  bool use_buffer_offset_ = false;
  // flag to indicate flatbuffer area got size > 2G
  bool require_use_buffer_offset_ = false;
  // map to store Buffer data by index as extended Buffer
  std::map<uint32_t, std::string> buffer_data_map_;
};

std::string Translator::UniqueName(mlir::Value val)
{
  return std::string(name_mapper_.GetUniqueName(val));
}

using BYTES = std::vector<uint8_t>;

template <typename T> BYTES ElementsAttrToBytes(mlir::ElementsAttr attr)
{
  BYTES data;
  for (T v : attr.getValues<T>())
  {
    uint8_t *ptr = reinterpret_cast<uint8_t *>(&v);
    for (size_t i = 0, sz = sizeof(T); i < sz; ++i)
      data.emplace_back(ptr[i]);
  }
  return data;
}

std::optional<BufferOffset<circle::Buffer>> Translator::BuildBuffer(mlir::Value value,
                                                                    const uint32_t index)
{
  auto inst = value.getDefiningOp();
  mlir::ElementsAttr attr;
  if (auto cst = llvm::dyn_cast<mlir::arith::ConstantOp>(inst))
  {
    // arith::ConstantOp have ElementAttr at this point due to validation of the
    // Circle module.
    attr = mlir::cast<mlir::ElementsAttr>(cst.getValue());
  }
  else if (auto cst = llvm::dyn_cast<mlir::Circle::ConstOp>(inst))
  {
    attr = cst.getValue();
  }
  // Support QConstOp, SparseConstOp, SparseQConstOp
  else
  {
    return empty_buffer_;
  }

  auto type = mlir::cast<mlir::TensorType>(value.getType());
  circle::TensorType circle_element_type = GetCircleType(type.getElementType());

  BYTES data;
  switch (circle_element_type)
  {
    case circle::TensorType_FLOAT32:
      data = ElementsAttrToBytes<float>(attr);
      break;
    case circle::TensorType_INT64:
      data = ElementsAttrToBytes<int64_t>(attr);
      break;
    case circle::TensorType_INT32:
      data = ElementsAttrToBytes<int32_t>(attr);
      break;
    case circle::TensorType_INT16:
      data = ElementsAttrToBytes<int16_t>(attr);
      break;
    case circle::TensorType_UINT8:
      data = ElementsAttrToBytes<uint8_t>(attr);
      break;
    case circle::TensorType_BOOL:
      data = ElementsAttrToBytes<bool>(attr);
      break;
    default:
      // TODO support other types
      throw std::runtime_error("NYI convert mlir::ElementsAttr to flatbuffers vector");
  }
  if (use_buffer_offset_)
  {
    std::string strdata;
    strdata.reserve(data.size());
    strdata.append(reinterpret_cast<char *>(data.data()), data.size());
    buffer_data_map_[index] = strdata;
    // create fake indicator Buffer with size 1, offset 1
    return circle::CreateBuffer(builder_, 0, 1, 1);
  }
  if (IsModelBiggerThan2GB(data.size()))
  {
    require_use_buffer_offset_ = true;
    return empty_buffer_;
  }
  auto buffer_data = builder_.CreateVector(data.data(), data.size());
  return circle::CreateBuffer(builder_, buffer_data);
}

std::optional<std::vector<BufferOffset<circle::VariantSubType>>>
Translator::BuildVariantType(mlir::Type element_type)
{
  // TODO implement
  std::vector<BufferOffset<circle::VariantSubType>> variant_params;
  return variant_params;
}

std::optional<BufferOffset<circle::Tensor>> Translator::BuildTensor(
  mlir::Value value, const std::string &name, unsigned buffer_idx,
  const std::optional<BufferOffset<circle::QuantizationParameters>> &quant_parameters)
{
  auto type = mlir::cast<mlir::TensorType>(value.getType());

  // Circle requires tensor shape only for the inputs and constants.
  // However, we output all known shapes for better round-tripping
  auto check_shape = [&](llvm::ArrayRef<int64_t> shape_ref) -> mlir::LogicalResult {
    auto is_out_of_range = [](int64_t dim) { return dim > std::numeric_limits<int32_t>::max(); };

    if (std::any_of(shape_ref.begin(), shape_ref.end(), is_out_of_range))
      return mlir::emitError(value.getLoc(),
                             "result shape dimensions out of 32 bit int type range");

    return mlir::success();
  };

  std::vector<int32_t> shape;
  std::vector<int32_t> shape_signature;
  auto *inst = value.getDefiningOp();
  if (type.hasStaticShape())
  {
    llvm::ArrayRef<int64_t> shape_ref = type.getShape();
    if (mlir::failed(check_shape(shape_ref)))
      return std::nullopt;

    shape = std::vector<int32_t>(shape_ref.begin(), shape_ref.end());
  }
  else if (inst && IsConst(inst))
  {
    // Const op can have a result of dynamic shaped type (e.g. due to constant
    // folding), but we can still derive the shape of a constant tensor for
    // its attribute type.
    auto tensor_attr = mlir::cast<mlir::TypedAttr>(inst->getAttr("value"));
    llvm::ArrayRef<int64_t> shape_ref =
      mlir::cast<mlir::TensorType>(tensor_attr.getType()).getShape();
    if (mlir::failed(check_shape(shape_ref)))
      return std::nullopt;

    shape = std::vector<int32_t>(shape_ref.begin(), shape_ref.end());
  }
  else if (type.hasRank())
  {
    llvm::ArrayRef<int64_t> shape_ref = type.getShape();
    if (mlir::failed(check_shape(shape_ref)))
      return std::nullopt;

    shape.reserve(shape_ref.size());
    for (auto &dim : shape_ref)
    {
      // translate dynamic shapes from mlir to circle values
      shape.push_back(dim == mlir::ShapedType::kDynamic ? 1 : static_cast<int>(dim));
      shape_signature.push_back(
        static_cast<int>(dim == mlir::ShapedType::kDynamic ? kCIRDynamicSize : dim));
    }
  }

  BufferOffset<circle::SparsityParameters> s_params = 0;
  if (auto *inst = value.getDefiningOp())
  {
    // TODO support SparseConstOp
  }

  mlir::Type element_type = type.getElementType();
  circle::TensorType circle_element_type = GetCircleType(type.getElementType());

  std::optional<std::vector<BufferOffset<circle::VariantSubType>>> variant_params =
    BuildVariantType(element_type);
  if (!variant_params.has_value())
  {
    return std::nullopt;
  }

  BufferOffset<circle::QuantizationParameters> q_params;
  q_params = circle::CreateQuantizationParameters(builder_);

  // Check if the value's uses includes an op and usage at an operand index
  // marked as a stateful. If so, set the tensor's is_variable as true
  // This is v1 ref variable semantics in the TFLite runtime.
  bool is_variable = false;
  // TODO support is_variable

  bool has_rank = type.hasRank();

  // There is a limit of 2GB for a flatbuffer.
  if (IsModelBiggerThan2GB(0))
  {
    require_use_buffer_offset_ = true;
    return std::nullopt;
  }

  if (shape_signature.empty())
  {
    return circle::CreateTensor(
      builder_, builder_.CreateVector(shape), circle_element_type, (is_variable ? 0 : buffer_idx),
      builder_.CreateString(name), q_params,
      /*is_variable=*/is_variable, s_params, /*shape_signature=*/0,
      /*has_rank=*/has_rank, variant_params->empty() ? 0 : builder_.CreateVector(*variant_params));
  }
  else
  {
    return circle::CreateTensor(
      builder_, builder_.CreateVector(shape), circle_element_type, (is_variable ? 0 : buffer_idx),
      builder_.CreateString(name), q_params,
      /*is_variable=*/is_variable, s_params,
      /*shape_signature=*/builder_.CreateVector(shape_signature),
      /*has_rank=*/has_rank, variant_params->empty() ? 0 : builder_.CreateVector(*variant_params));
  }
}

BufferOffset<circle::Operator> Translator::BuildCustomOperator(Operation *inst,
                                                               mlir::Circle::CustomOp op,
                                                               const std::vector<int32_t> &operands,
                                                               const std::vector<int32_t> &results)
{
  const std::string attrs =
    mlir::cast<mlir::Circle::ConstBytesAttr>(op.getCustomOption()).getValue().str();
  std::vector<uint8_t> custom_option_vector(attrs.size());
  memcpy(custom_option_vector.data(), attrs.data(), attrs.size());
  auto opcode_index = GetOpcodeIndex(op.getCustomCode().str(), circle::BuiltinOperator_CUSTOM);
  return circle::CreateOperator(builder_, opcode_index, builder_.CreateVector(operands),
                                builder_.CreateVector(results), circle::BuiltinOptions_NONE,
                                /*builtin_options=*/0,
                                builder_.CreateVector<uint8_t>(custom_option_vector),
                                circle::CustomOptionsFormat_FLEXBUFFERS);
}

std::optional<std::string> Translator::Translate(mlir::ModuleOp module,
                                                 OpOrArgNameMapper *op_or_arg_name_mapper,
                                                 const std::map<std::string, std::string> &metadata)
{
  OpOrArgLocNameMapper default_op_or_arg_name_mapper;
  OpOrArgNameMapper *local_op_or_arg_name_mapper = op_or_arg_name_mapper;
  if (!local_op_or_arg_name_mapper)
    local_op_or_arg_name_mapper = &default_op_or_arg_name_mapper;
  // TODO check validity

  Translator *translator = new Translator(module, local_op_or_arg_name_mapper, metadata);
  auto ret = translator->TranslateInternal();
  if (translator->require_use_buffer_offset_)
  {
    delete translator;
    translator = new Translator(module, local_op_or_arg_name_mapper, metadata);
    translator->use_buffer_offset_ = true;
    ret = translator->TranslateInternal();
  }
  delete translator;
  return ret;
}

uint32_t Translator::GetOpcodeIndex(const std::string &op_name, circle::BuiltinOperator builtin)
{
  auto it = opcode_index_map_.insert({op_name, 0});

  // If the insert succeeded, the opcode has not been created already. Create a
  // new operator code and update its index value in the map.
  if (it.second)
  {
    it.first->second = opcodes_.size();
    auto custom_code = builtin == circle::BuiltinOperator_CUSTOM
                         ? builder_.CreateString(op_name)
                         : BufferOffset<flatbuffers::String>();
    // Use version 0 for builtin op. This is a way to serialize version field to
    // flatbuffer (since 0 is non default) and it will be corrected later.
    int32_t op_version = 1; // builtin != circle::BuiltinOperator_CUSTOM ? 0 : 1;
    int8_t dep_builtin = builtin > circle::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES
                           ? circle::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES
                           : static_cast<int8_t>(builtin);
    opcodes_.push_back(CreateOperatorCode(builder_, dep_builtin, custom_code, op_version, builtin));
  }
  return it.first->second;
}

std::optional<BufferOffset<circle::Operator>>
Translator::BuildOperator(mlir::Operation *inst, std::vector<int32_t> operands,
                          const std::vector<int32_t> &results,
                          const std::vector<int32_t> &intermediates)
{
  const auto *dialect = inst->getDialect();
  if (!dialect)
  {
    inst->emitOpError("dialect is not registered");
    return std::nullopt;
  }

  // If Circle built in op, create operator as a builtin op.
  if (dialect == circle_dialect_)
  {
    auto builtin_code = GetBuiltinOpCode(inst);
    if (!builtin_code)
    {
      // TODO NumericVerifyOp / WhileOp

      if (auto custom_op = dyn_cast<mlir::Circle::CustomOp>(inst))
      {
        return BuildCustomOperator(inst, custom_op, operands, results);
      }

      inst->emitOpError("is not a supported Circle op");
      return std::nullopt;
    }

    // TODO BuiltinOperator_CALL_ONCE

    std::string op_name = inst->getName().getStringRef().str();
    uint32_t opcode_index = GetOpcodeIndex(op_name, *builtin_code);

    // If this is TransposeConv we need to do a special case of ignoring the
    // optional tensor, to allow newly created models to run on old runtimes.
    if (*builtin_code == circle::BuiltinOperator_TRANSPOSE_CONV)
    {
      if (operands.size() == 4 && operands.at(3) == -1)
      {
        operands.pop_back();
      }
    }

    auto offset =
      CreateFlatBufferOperator(inst, opcode_index, operands, results, intermediates, &builder_);
    if (!offset)
    {
      inst->emitOpError("is not a supported Circle op");
    }
    return offset;
  }

  return inst->emitOpError("is not builtin Circle op"), std::nullopt;
}

void Translator::InitializeNamesFromAttribute(mlir::func::FuncOp fn, bool *has_input_attr)
{
  mlir::ArrayAttr inputNames = fn->getAttrOfType<mlir::ArrayAttr>("input_names");
  if (inputNames)
  {
    assert(inputNames.size() == fn.getArguments().size());
    for (const auto &it : llvm::enumerate(fn.getArguments()))
    {
      auto strattr = mlir::cast<mlir::StringAttr>(inputNames[it.index()]);
      name_mapper_.InitOpName(it.value(), strattr);
    }
    *has_input_attr = true;
  }
  mlir::ArrayAttr outputNames = fn->getAttrOfType<mlir::ArrayAttr>("output_names");
  if (outputNames)
  {
    auto term = fn.back().getTerminator();
    assert(outputNames.size() == term->getOperands().size());
    for (const auto &it : llvm::enumerate(term->getOperands()))
    {
      auto strattr = mlir::cast<mlir::StringAttr>(outputNames[it.index()]);
      name_mapper_.InitOpName(it.value(), strattr);
    }
    *has_input_attr = true;
  }
}

std::optional<BufferOffset<circle::SubGraph>>
Translator::BuildSubGraph(const std::string &name, mlir::Region *region, const int index)
{
  bool has_input_attr = false;
  if (auto fn = llvm::dyn_cast<mlir::func::FuncOp>(region->getParentOp()))
  {
    InitializeNamesFromAttribute(fn, &has_input_attr);
  }

  std::vector<BufferOffset<circle::Tensor>> tensors;
  llvm::DenseMap<mlir::Value, int> tensor_index_map;

  // Builds tensor and buffer for argument or operation result. Returns false on failure.
  auto build_tensor_and_buffer = [&](mlir::Value value, const int subgraph_index,
                                     const std::string &tensor_name) {
    // NoneType represents optional and may be skipped here.
    if (mlir::isa<mlir::NoneType>(value.getType()))
    {
      return true;
    }

    tensor_index_map.insert({value, tensors.size()});
    tensor_index_map_[subgraph_index][tensor_name] = tensors.size();
    std::optional<BufferOffset<circle::QuantizationParameters>> quant_parameters;
    if (value.hasOneUse())
    {
      // TODO fill quant_parameters
    }
    LLVM_DEBUG({ llvm::dbgs() << "Export Tensor: " << tensor_name << "\n"; });
    auto tensor_or = BuildTensor(value, tensor_name, buffers_.size(), quant_parameters);
    if (!tensor_or)
      return false;
    tensors.push_back(*tensor_or);

    // TODO(ashwinm): Check if for stateful tensors, if it is also needed to
    // make the Buffer empty apart from setting the buffer_idx=0 in the
    // Tensor. This does not seem to affect runtime behavior for RNN/LSTM,
    // but would be good for reducing memory footprint.
    if (auto *inst = value.getDefiningOp())
    {
      uint32_t buff_index = buffers_.size();
      auto buffer_or = BuildBuffer(value, buff_index);
      if (require_use_buffer_offset_)
        return false;
      if (!buffer_or)
        return false;
      buffers_.push_back(*buffer_or);
    }
    else
    {
      buffers_.push_back(empty_buffer_);
    }
    return true;
  };

  std::vector<BufferOffset<circle::Operator>> operators;

  // Maps positions of operations in bb to positions in operators
  llvm::DenseMap<int, int> operation_index_to_operator_index;
  std::vector<mlir::Operation *> operators_in_mlir;
  auto &bb = region->front();

  // Main function's arguments are first passed to `input` op so they don't
  // have associated tensor and buffer. Build FlatBuffer tensor and buffer for
  // other functions.
  for (unsigned i = 0, e = bb.getNumArguments(); i < e; ++i)
  {
    mlir::BlockArgument arg = bb.getArgument(i);
    std::string tensor_name;
    if (has_input_attr)
      tensor_name = std::string(name_mapper_.GetUniqueName(arg));
    if (tensor_name.empty())
      tensor_name = absl::StrCat("arg", i);
    if (!build_tensor_and_buffer(arg, index, tensor_name))
      return std::nullopt;
  }

  bool failed_once = false;
  for (const auto &item : llvm::enumerate(bb))
  {
    mlir::Operation &inst = item.value();
    const int operation_index = item.index();
    if (inst.hasTrait<mlir::OpTrait::IsTerminator>())
      break;

    // TODO support Quantization

    std::vector<int32_t> intermediates;
    // TODO Build intermediate tensors for LSTM and insert these tensors into
    // flatbuffer.

    for (auto val : inst.getResults())
    {
      std::string tensor_name = UniqueName(val);
      // For "numeric_verify" op, the name is used to find out the original
      // activation tensor rather than its own unique name in the visualization
      // or debugging tools.
      auto builtin_code = GetBuiltinOpCode(&inst);
      // TODO NumericVerifyOp ?
      if (!build_tensor_and_buffer(val, index, tensor_name))
        return std::nullopt;
    }

    // Skip constant ops as they don't represent a Circle operator.
    if (IsConst(&inst))
      continue;

    // Fetch operand and result tensor indices.
    std::vector<int32_t> results;
    results.reserve(inst.getNumResults());
    for (auto result : inst.getResults())
    {
      results.push_back(tensor_index_map.lookup(result));
    }
    mlir::Operation *real_inst = &inst;
    std::vector<int32_t> operands;
    operands.reserve(real_inst->getNumOperands());
    for (auto operand : real_inst->getOperands())
    {
      if (mlir::isa<mlir::NoneType>(operand.getType()))
        operands.push_back(kCircleOptionalTensor);
      else
        operands.push_back(tensor_index_map.lookup(operand));
    }

    // TODO CustomOp

    if (auto cir_operator = BuildOperator(real_inst, operands, results, intermediates))
    {
      operation_index_to_operator_index.try_emplace(operation_index, operators.size());
      operators.push_back(*cir_operator);
      operators_in_mlir.push_back(real_inst);
    }
    else
    {
      failed_once = true;
    }
  }
  if (index + 1 > subgraph_op_inst_map_.size())
  {
    subgraph_op_inst_map_.resize(index + 1);
  }
  subgraph_op_inst_map_[index] = operators_in_mlir;
  if (failed_once)
    return std::nullopt;

  std::vector<int32_t> inputs, outputs;
  for (auto arg : bb.getArguments())
  {
    inputs.push_back(tensor_index_map[arg]);
  }
  for (auto result : bb.getTerminator()->getOperands())
  {
    outputs.push_back(tensor_index_map[result]);
  }

  return circle::CreateSubGraph(builder_, builder_.CreateVector(tensors),
                                builder_.CreateVector(inputs), builder_.CreateVector(outputs),
                                builder_.CreateVector(operators),
                                /*name=*/builder_.CreateString(name));
}

std::optional<VectorBufferOffset<BufferOffset<circle::Metadata>>> Translator::CreateMetadataVector()
{
  // TODO implement

  std::vector<BufferOffset<circle::Metadata>> metadata;

  return builder_.CreateVector(metadata);
}

std::vector<SignatureDefData> BuildSignaturedef(mlir::func::FuncOp main_op,
                                                const std::string &saved_model_tag,
                                                const uint32_t subgraph_index,
                                                OpOrArgNameMapper &name_mapper)
{
  // TODO implement

  // Fill the SignatureDefData container.
  // We create vector of size 1 as Circle now supports only 1 signatureDef.
  std::vector<SignatureDefData> result(1);

  // TODO fill inputs/outputs

  result[0].subgraph_index = subgraph_index;
  return result;
}

std::optional<VectorBufferOffset<BufferOffset<circle::SignatureDef>>>
Translator::CreateSignatureDefs(const std::vector<SignatureDefData> &signature_defs)
{
  std::vector<BufferOffset<circle::SignatureDef>> signature_defs_buffer;

  // TODO implement

  return builder_.CreateVector(signature_defs_buffer);
}

std::optional<std::string> Translator::TranslateInternal()
{
  // A list of named regions in the module with main function being the first in
  // the list. The main function is required as the first subgraph in the model
  // is entry point for the model.
  std::vector<std::pair<std::string, mlir::Region *>> named_regions;
  named_regions.reserve(std::distance(module_.begin(), module_.end()));

  int subgraph_idx = 0;

  // TODO check for multiple subgraphs

  // Entry functions for signature defs.
  std::vector<mlir::func::FuncOp> entry_functions;
  std::vector<mlir::func::FuncOp> non_entry_functions;
  // TODO check symbol
  mlir::func::FuncOp main_fn = module_.lookupSymbol<mlir::func::FuncOp>("main_graph");
  if (main_fn != nullptr)
  {
    entry_functions.push_back(main_fn);
  }

  // Walk over the module collection ops with functions and while ops.
  module_.walk([&](mlir::func::FuncOp fn) {
    if (main_fn == fn)
      return mlir::WalkResult::advance();

    non_entry_functions.push_back(fn);
    return mlir::WalkResult::advance();
  });

  // Assign the subgraph index. Among the given functions, it will put entry
  // functions at the beginning of the list of the subgrahs.
  for (auto fn : entry_functions)
  {
    subgraph_index_map_[fn.getName().str()] = subgraph_idx++;
    named_regions.emplace_back(fn.getName().str(), &fn.getBody());
  }
  for (auto fn : non_entry_functions)
  {
    subgraph_index_map_[fn.getName().str()] = subgraph_idx++;
    named_regions.emplace_back(fn.getName().str(), &fn.getBody());
  }

  // Build subgraph for each of the named regions.
  std::vector<BufferOffset<circle::SubGraph>> subgraphs;
  subgraphs.reserve(named_regions.size());
  // model_control_dependencies_.assign(named_regions.size(), {});
  int first_failed_func = -1;

  // When we export each function in the module op, intentionally, we export the
  // entry functions at the beginning of the subgraph list and the
  // subgraph_index is the index in entry functions and at the same, is the
  // index in the subgraph list.
  int subgraph_index = 0;
  for (const auto &it : llvm::enumerate(named_regions))
  {
    auto subgraph_or = BuildSubGraph(it.value().first, it.value().second, subgraph_index);
    if (require_use_buffer_offset_)
      return std::nullopt;
    if (!subgraph_or)
    {
      if (first_failed_func == -1)
        // Record the index of the first region that cannot be converted.
        // Keep looping through all subgraphs in the module to make sure that
        // we collect the list of missing ops from the entire module.
        first_failed_func = it.index();
    }
    else
    {
      subgraphs.push_back(*subgraph_or);
      ++subgraph_index;
    }
  }

  // TODO dump warnings with resource, flex, custom ops

  if (first_failed_func != -1)
  {
    // TODO get detailed error for flex, custom ops
    auto &failed_region = named_regions[first_failed_func];
    return failed_region.second->getParentOp()->emitError()
             << "failed while converting: '" << failed_region.first,
           std::nullopt;
  }
  std::string model_description;
  if (auto attr = module_->getAttrOfType<mlir::StringAttr>("circle.description"))
  {
    model_description = attr.getValue().str();
  }
  else
  {
    model_description = "MLIR Converted.";
  }

  // Build the model and finish the model building process.
  auto description = builder_.CreateString(model_description.data());
  VectorBufferOffset<int32_t> metadata_buffer = 0; // Deprecated
  auto metadata = CreateMetadataVector();
  if (!metadata)
    return std::nullopt;

  std::vector<SignatureDefData> signature_defs_vec;
  subgraph_index = 0;
  // Build SignatureDefs for the tf.entry_function based func ops.
  for (auto fn : entry_functions)
  {
    auto signature_defs =
      BuildSignaturedef(fn, saved_model_tags_.empty() ? "" : *saved_model_tags_.begin(),
                        subgraph_index, name_mapper_);
    for (const auto &signature_def : signature_defs)
    {
      signature_defs_vec.push_back(signature_def);
    }
    // When we export each function in the module op, intentionally, we export
    // the entry functions at the beginning of the subgraph list and the
    // subgraph_index is the index in entry functions and at the same, is the
    // index in the subgraph list.
    ++subgraph_index;
  }
  auto signature_defs = CreateSignatureDefs(signature_defs_vec);

  if (IsModelBiggerThan2GB(0))
  {
    require_use_buffer_offset_ = true;
    return std::nullopt;
  }

  auto model = circle::CreateModel(builder_, CIRCLE_SCHEMA_VERSION, builder_.CreateVector(opcodes_),
                                   builder_.CreateVector(subgraphs), description,
                                   builder_.CreateVector(buffers_), metadata_buffer, *metadata,
                                   *signature_defs);
  circle::FinishModelBuffer(builder_, model);
  // There is a limit of 2GB for a flatbuffer.
  if (IsModelBiggerThan2GB(0))
  {
    require_use_buffer_offset_ = true;
    return std::nullopt;
  }

  return FinalizeWithExtendedBuffer();
}

std::optional<std::string> Translator::FinalizeWithExtendedBuffer(void)
{
  if (!use_buffer_offset_)
  {
    // Return serialized string for the built FlatBuffer.
    return std::string(reinterpret_cast<const char *>(builder_.GetBufferPointer()),
                       builder_.GetSize());
  }

  auto align16 = [](size_t &v) {
    while (v % 16 != 0)
      v++;
  };

  // get total memory for flatbuffer + all buffer_data
  size_t result_size = builder_.GetSize();
  align16(result_size);
  for (auto &it : buffer_data_map_)
  {
    auto &buffer_data = it.second;
    result_size += buffer_data.size();
    align16(result_size);
  }
  align16(result_size);
  result_size += 16; // for safety

  std::string result;
  const char *buff_ptr = reinterpret_cast<const char *>(builder_.GetBufferPointer());

  auto padalign16 = [](std::string &str) {
    while (str.size() % 16 != 0)
      str += '\0';
  };

  result.reserve(result_size);
  result.append(buff_ptr, builder_.GetSize());

  auto mutable_model = circle::GetMutableModel(result.data());
  auto mutable_buffers = mutable_model->mutable_buffers();

  // pad to be 16 bytes aligned
  padalign16(result);
  for (auto &it : buffer_data_map_)
  {
    uint32_t buffer_index = it.first;
    auto &buffer_data = it.second;
    uint64_t offset = result.size();
    uint64_t size = buffer_data.size();

    circle::Buffer *mutable_buffer = mutable_buffers->GetMutableObject(buffer_index);
    mutable_buffer->mutate_offset(offset);
    mutable_buffer->mutate_size(size);

    result.append(buffer_data);
    padalign16(result);
  }
  padalign16(result);

  return result;
}

} // namespace

// Translates the given MLIR `module` into a FlatBuffer and stores the
// serialized flatbuffer into the string.
// Returns true on successful exporting, false otherwise.
bool MlirToFlatBufferTranslateFunction(mlir::ModuleOp module, std::string *serialized_flatbuffer)
{
  std::map<std::string, std::string> metadata;
  OpOrArgNameMapper *op_or_arg_name_mapper = nullptr;

  auto maybe_translated = Translator::Translate(module, op_or_arg_name_mapper, metadata);
  if (!maybe_translated)
    return false;
  *serialized_flatbuffer = std::move(*maybe_translated);
  return true;
}

} // namespace Circle
} // namespace mlir
