/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CircleTensorExporter.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Service/CircleTypeInference.h>
#include <luci/Service/CircleShapeInference.h>
#include <luci/Log.h>

#include <loco/IR/Algorithm.h>
#include <loco/IR/CanonicalNode.h>
#include <loco/IR/CanonicalNodeVisitor.h>
#include <loco/IR/DataTypeTraits.h>
#include <oops/InternalExn.h>

#include <string.h>

using namespace circle;
using namespace flatbuffers;

namespace
{

using namespace luci;

class CircleTensorInfo
{
public:
  CircleTensorInfo() = default;

public:
  void name(const std::string &name) { _name = name; }
  const std::string &name(void) const { return _name; }

public:
  const circle::TensorType &dtype(void) const { return _dtype; }
  void dtype(const circle::TensorType &dtype) { _dtype = dtype; }

  const ShapeDescription &shape(void) const { return _shape; }
  void shape(const ShapeDescription &shape) { _shape = shape; }

  luci::ShapeStatus shape_status(void) const { return _shape_status; }
  void shape_status(luci::ShapeStatus ss) { _shape_status = ss; }

public:
  luci::CircleConst *content(void) const { return _content; }
  void content(luci::CircleConst *c) { _content = c; }

  luci::CircleQuantParam *quantparam(void) const { return _quantparam; }
  void quantparam(luci::CircleQuantParam *qp) { _quantparam = qp; }

  luci::SparsityParam *sparsityparam(void) const { return _sparsityparam; }
  void sparsityparam(luci::SparsityParam *sp) { _sparsityparam = sp; }

  bool is_variable(void) const { return _is_variable; }
  void is_variable(bool v) { _is_variable = v; }

private:
  std::string _name;

  circle::TensorType _dtype{circle::TensorType_FLOAT32};
  ShapeDescription _shape{};
  luci::ShapeStatus _shape_status{luci::ShapeStatus::UNDEFINED};

  luci::CircleConst *_content = nullptr;
  luci::CircleQuantParam *_quantparam = nullptr;
  luci::SparsityParam *_sparsityparam = nullptr;

  bool _is_variable = false;
};

class CircleTensorContext
{
public:
  CircleTensorContext() = default;

public:
  void emplace_back(CircleTensorInfo &ti)
  {
    assert(_names.find(ti.name()) == _names.end());
    _tis.emplace_back(ti);
    _names.insert(ti.name());
  }
  size_t size(void) const { return _tis.size(); }
  std::vector<CircleTensorInfo>::iterator begin(void) { return _tis.begin(); }
  std::vector<CircleTensorInfo>::iterator end(void) { return _tis.end(); }

public:
  bool exist(const std::string &name) const { return _names.find(name) != _names.end(); }

private:
  std::vector<CircleTensorInfo> _tis;
  std::set<std::string> _names;
};

struct NoOpDetector final : public luci::CircleNodeMutableVisitor<bool>
{
  // Input is Virtual but does produce a Tensor
  // Output is Virtual that does not produce any Tensor
  bool visit(luci::CircleOutput *) final { return true; }
  bool visit(luci::CircleOutputExclude *) final { return true; }

  // Return false by default
  bool visit(luci::CircleNode *) final { return false; }
};

void allocateCircleTensorInfo(CircleNode *node, CircleTensorContext &ctx)
{
  LOGGER(l);

  auto tensor_index = static_cast<CircleTensorIndex>(ctx.size());
  // TODO Use Graph-level metadata for Input & Output
  std::string tensor_name = node->name();
  // NOTE tensor_name maybe empty. this assertion will alert when this happens.
  //      currently we require tensor should have a name.
  // TODO if this breaks, fix the cause or permit empty tensor_name.
  assert(!tensor_name.empty());
  if (ctx.exist(tensor_name))
  {
    // NOTE this should assign unique name for a Tensor.
    tensor_name = tensor_name + "_" + std::to_string(tensor_index);
    assert(!ctx.exist(tensor_name));
  }
  INFO(l) << "[luci] Tensor for " << tensor_name << ": " << tensor_index << std::endl;

  CircleTensorInfo tensor_info;

  tensor_info.name(tensor_name);
  tensor_info.dtype(to_circle_tensortype(node->dtype()));
  if (node->shape_status() == ShapeStatus::VALID)
    tensor_info.shape(to_shape_description(node));
  tensor_info.shape_status(node->shape_status());

  tensor_info.content(dynamic_cast<luci::CircleConst *>(node));
  tensor_info.quantparam(node->quantparam());
  tensor_info.sparsityparam(node->sparsityparam());

  tensor_info.is_variable(dynamic_cast<luci::CircleVariable *>(node) != nullptr);

  set_tensor_index(node, tensor_index);

  ctx.emplace_back(tensor_info);
}

class MultiOutputDetector final : public luci::CircleNodeMutableVisitor<bool>
{
public:
  MultiOutputDetector(CircleTensorContext &ctx) : _ctx(ctx) {}

private:
  void store_outputs(luci::CircleNode *node, uint32_t count)
  {
    auto outs = loco::succs(node);
    assert(outs.size() == count);
    (void)count; // for unused variable error in release build
    for (auto out : outs)
    {
      auto circle_out = loco::must_cast<luci::CircleNode *>(out);
      allocateCircleTensorInfo(circle_out, _ctx);
    }
    set_tensor_index(node, -1);
  }

public:
  bool visit(luci::CircleBidirectionalSequenceLSTMOut *) final { return true; }
  bool visit(luci::CircleCustomOut *) final { return true; }
  bool visit(luci::CircleIfOut *) final { return true; }
  bool visit(luci::CircleNonMaxSuppressionV4Out *) final { return true; }
  bool visit(luci::CircleNonMaxSuppressionV5Out *) final { return true; }
  bool visit(luci::CircleSplitOut *) final { return true; }
  bool visit(luci::CircleSplitVOut *) final { return true; }
  bool visit(luci::CircleTopKV2Out *) final { return true; }
  bool visit(luci::CircleUnpackOut *) final { return true; }
  bool visit(luci::CircleUniqueOut *) final { return true; }
  bool visit(luci::CircleWhileOut *) final { return true; }

  bool visit(luci::CircleBidirectionalSequenceLSTM *node) final
  {
    if (node->merge_outputs())
    {
      store_outputs(node, 1);
    }
    else
    {
      store_outputs(node, 2);
    }
    return true;
  }

  bool visit(luci::CircleCustom *node) final
  {
    store_outputs(node, node->numOutputs());
    return true;
  }

  bool visit(luci::CircleIf *node) final
  {
    store_outputs(node, node->output_count());
    return true;
  }

  bool visit(luci::CircleNonMaxSuppressionV4 *node) final
  {
    store_outputs(node, 2);
    return true;
  }

  bool visit(luci::CircleNonMaxSuppressionV5 *node) final
  {
    store_outputs(node, 3);
    return true;
  }

  bool visit(luci::CircleSplit *node) final
  {
    store_outputs(node, uint32_t(node->num_split()));
    return true;
  }

  bool visit(luci::CircleSplitV *node) final
  {
    store_outputs(node, uint32_t(node->num_split()));
    return true;
  }

  bool visit(luci::CircleTopKV2 *node) final
  {
    store_outputs(node, 2);
    return true;
  }

  bool visit(luci::CircleUnpack *node) final
  {
    store_outputs(node, node->num());
    return true;
  }

  bool visit(luci::CircleUnique *node) final
  {
    store_outputs(node, 2);
    return true;
  }

  bool visit(luci::CircleWhile *node) final
  {
    store_outputs(node, node->output_count());
    return true;
  }

  // Return false by default
  bool visit(luci::CircleNode *) final { return false; }

private:
  CircleTensorContext &_ctx;
};

void allocateCircleTensor(CircleNode *node, CircleTensorContext &ctx)
{
  if (node == nullptr)
    throw std::runtime_error("allocateCIrcleTensor Failed : node is nullptr");

  auto isNoOp = [](loco::Node *node) {
    if (auto circle_node = dynamic_cast<luci::CircleNode *>(node))
    {
      NoOpDetector d;
      return circle_node->accept(&d);
    }
    return false;
  };

  if (isNoOp(node))
  {
    set_tensor_index(node, -1);
    return;
  }

  // TODO revise this when loco supports multiple outputs
  // NOTE this will store all virtual output tensors and skip for the real node
  if (auto circle_node = dynamic_cast<luci::CircleNode *>(node))
  {
    MultiOutputDetector d(ctx);
    if (circle_node->accept(&d))
      return;
  }

  allocateCircleTensorInfo(node, ctx);
}

} // namespace

namespace
{

flatbuffers::Offset<Vector<int32_t>> encodeShape(FlatBufferBuilder &builder,
                                                 const ShapeDescription &shape)
{
  assert(shape._rank_known && "unknown number of dimensions is not supported");

  std::vector<int32_t> encoded_shape;
  encoded_shape.resize(shape._dims.size());
  for (uint32_t i = 0; i < shape._dims.size(); ++i)
    encoded_shape.at(i) = shape._dims.at(i) == -1 ? 1 : shape._dims.at(i);

  return builder.CreateVector(encoded_shape);
}

flatbuffers::Offset<Vector<int32_t>> encodeShapeSignature(FlatBufferBuilder &builder,
                                                          const ShapeDescription &shape)
{
  assert(shape._rank_known && "unknown number of dimensions is not supported");

  // shape_signature is set if and only if at least one of dimensions are unknown.
  for (uint32_t i = 0; i < shape._dims.size(); ++i)
    if (shape._dims.at(i) == -1)
      return builder.CreateVector(shape._dims);

  return flatbuffers::Offset<Vector<int32_t>>();
}

flatbuffers::Offset<circle::Buffer> encodeOpBuffer(FlatBufferBuilder &builder)
{
  return CreateBuffer(builder);
}

template <typename NodeT>
flatbuffers::Offset<circle::Buffer> encodeOpBuffer(FlatBufferBuilder &builder, NodeT *)
{
  return CreateBuffer(builder);
}

template <loco::DataType DT>
flatbuffers::Offset<circle::Buffer> encodeOpBufferByDType(FlatBufferBuilder &builder,
                                                          luci::CircleConst *c)
{
  using NativeType = typename loco::DataTypeImpl<DT>::Type;

  std::vector<NativeType> raw_data;
  const uint32_t size = c->size<DT>();
  raw_data.reserve(size);
  for (uint32_t i = 0; i < size; ++i)
  {
    raw_data.push_back(c->at<DT>(i));
  }
  const size_t raw_size = size * sizeof(NativeType);
  auto array_offset = builder.CreateVector(reinterpret_cast<uint8_t *>(raw_data.data()), raw_size);
  return CreateBuffer(builder, array_offset);
}

template <>
flatbuffers::Offset<circle::Buffer>
encodeOpBufferByDType<loco::DataType::STRING>(FlatBufferBuilder &builder, luci::CircleConst *c)
{
  const uint32_t count = c->size<loco::DataType::STRING>();
  uint32_t raw_size = sizeof(int32_t) * (count + 2);
  for (uint32_t i = 0; i < count; ++i)
  {
    auto &value = c->at<loco::DataType::STRING>(i);
    raw_size += value.length();
  }

  // serialize string data
  //   int32_t count
  //   int32_t offsets[count + 1]
  //   string  values[count]
  std::vector<uint8_t> raw_data;
  raw_data.reserve(raw_size);

  auto *i32d = reinterpret_cast<int32_t *>(raw_data.data());
  int32_t start = sizeof(int32_t) * (count + 2);
  int32_t offset = start;
  std::vector<int32_t> offsets;

  *i32d++ = count;
  *i32d++ = start;
  offsets.push_back(start);
  for (uint32_t i = 0; i < count; ++i)
  {
    auto &value = c->at<loco::DataType::STRING>(i);
    offset += value.length();
    *i32d++ = offset;
    offsets.push_back(offset);
  }

  auto *data = reinterpret_cast<uint8_t *>(i32d);
  for (uint32_t i = 0; i < count; ++i)
  {
    int32_t length = offsets[i + 1] - offsets[i];
    auto &value = c->at<loco::DataType::STRING>(i);
    memcpy(data, value.c_str(), length);
    data += length;
  }

  auto array_offset = builder.CreateVector(reinterpret_cast<uint8_t *>(raw_data.data()), raw_size);
  return CreateBuffer(builder, array_offset);
}

template <loco::DataType DT>
flatbuffers::Offset<circle::Buffer> encodeOpBufferPack4bit(FlatBufferBuilder &builder,
                                                           luci::CircleConst *c)
{
  const uint32_t size = c->size<DT>();
  const uint32_t raw_size = (size + 1) / 2;
  std::vector<uint8_t> raw_data(raw_size);

  for (uint32_t i = 0; i < raw_size; ++i)
  {
    uint32_t sidx = i * 2;
    uint8_t data = static_cast<uint8_t>(c->at<DT>(sidx));
    raw_data[i] = data & 0x0f;
    sidx++;
    if (sidx < size)
    {
      data = static_cast<uint8_t>(c->at<DT>(sidx));
      raw_data[i] |= data << 4;
    }
  }

  auto array_offset = builder.CreateVector(raw_data.data(), raw_size);
  return CreateBuffer(builder, array_offset);
}

template <>
flatbuffers::Offset<circle::Buffer> encodeOpBuffer(FlatBufferBuilder &builder, luci::CircleConst *c)
{
  switch (c->dtype())
  {
    case loco::DataType::FLOAT32:
      return encodeOpBufferByDType<loco::DataType::FLOAT32>(builder, c);
    case loco::DataType::S4:
      return encodeOpBufferPack4bit<loco::DataType::S4>(builder, c);
    case loco::DataType::S8:
      return encodeOpBufferByDType<loco::DataType::S8>(builder, c);
    case loco::DataType::S16:
      return encodeOpBufferByDType<loco::DataType::S16>(builder, c);
    case loco::DataType::S32:
      return encodeOpBufferByDType<loco::DataType::S32>(builder, c);
    case loco::DataType::S64:
      return encodeOpBufferByDType<loco::DataType::S64>(builder, c);
    case loco::DataType::U4:
      return encodeOpBufferPack4bit<loco::DataType::U4>(builder, c);
    case loco::DataType::U8:
      return encodeOpBufferByDType<loco::DataType::U8>(builder, c);
    case loco::DataType::BOOL:
      return encodeOpBufferByDType<loco::DataType::BOOL>(builder, c);
    case loco::DataType::STRING:
      return encodeOpBufferByDType<loco::DataType::STRING>(builder, c);
    default:
      break;
  }

  // NOTE loco::DataType::FLOAT16 is added but we do not export this type
  //      as backends currently don't support this type.
  //      currently this is supported only for "Tensor(Float16) - Dequantize"
  //      sequence so that after 'fold_dequantize' option this Tensor is
  //      converted to FLOAT32.

  INTERNAL_EXN_V("Unsupported datatype", oops::to_uint32(c->dtype()));
}

flatbuffers::Offset<circle::QuantizationParameters>
encodeQuantizationParameters(FlatBufferBuilder &builder, luci::CircleQuantParam *quantparam)
{
  if (quantparam == nullptr)
    return 0;

  flatbuffers::Offset<flatbuffers::Vector<float>> min;
  flatbuffers::Offset<flatbuffers::Vector<float>> max;
  flatbuffers::Offset<flatbuffers::Vector<float>> scale;
  flatbuffers::Offset<flatbuffers::Vector<int64_t>> zero_point;
  if (quantparam->min.size() && quantparam->max.size())
  {
    min = builder.CreateVector(quantparam->min);
    max = builder.CreateVector(quantparam->max);
  }
  if (quantparam->scale.size() && quantparam->zerop.size())
  {
    scale = builder.CreateVector(quantparam->scale);
    zero_point = builder.CreateVector(quantparam->zerop);
  }
  // Note: QuantizationDetails is not supported
  return circle::CreateQuantizationParameters(builder, min, max, scale, zero_point,
                                              circle::QuantizationDetails::QuantizationDetails_NONE,
                                              0, quantparam->quantized_dimension);
}

flatbuffers::Offset<circle::SparsityParameters>
encodeSparsityParameters(FlatBufferBuilder &builder, luci::SparsityParam *sparsityparam)
{
  if (sparsityparam == nullptr)
    return 0;

  std::vector<flatbuffers::Offset<circle::DimensionMetadata>> dim_metadata_vec;
  auto luci_dim_metadata = sparsityparam->dim_metadata;
  for (auto it : luci_dim_metadata)
  {
    // array_segments
    auto circle_array_segments = to_circle_sparse_index_vector(builder, it.array_segments());
    auto circle_array_segments_type =
      to_circle_sparse_index_vector_type(it.array_segments().type());

    // array_indices
    auto circle_array_indices = to_circle_sparse_index_vector(builder, it.array_indices());
    auto circle_array_indices_type = to_circle_sparse_index_vector_type(it.array_indices().type());
    auto dim_metadata = circle::CreateDimensionMetadata(
      builder, to_circle_dimensiontype(it.format()), it.dense_size(), circle_array_segments_type,
      circle_array_segments, circle_array_indices_type, circle_array_indices);
    dim_metadata_vec.emplace_back(dim_metadata);
  }

  return circle::CreateSparsityParametersDirect(builder, &sparsityparam->traversal_order,
                                                &sparsityparam->block_map, &dim_metadata_vec);
}

template <loco::DataType DT> bool has_same_elements(luci::CircleConst *lhs, luci::CircleConst *rhs)
{
  assert(lhs->dtype() == DT);
  assert(rhs->dtype() == DT);
  assert(lhs->size<DT>() == rhs->size<DT>());

  for (uint32_t i = 0; i < lhs->size<DT>(); ++i)
    if (lhs->at<DT>(i) != rhs->at<DT>(i))
      return false;
  return true;
}

bool has_same_values(luci::CircleConst *lhs, luci::CircleConst *rhs)
{
  if (lhs->dtype() != rhs->dtype())
    return false;

  if (lhs->rank() != rhs->rank())
    return false;

  for (uint32_t i = 0; i < lhs->rank(); ++i)
    if (!(lhs->dim(i) == rhs->dim(i)))
      return false;

  switch (lhs->dtype())
  {
    case loco::DataType::FLOAT32:
      return has_same_elements<loco::DataType::FLOAT32>(lhs, rhs);

    case loco::DataType::S4:
      return has_same_elements<loco::DataType::S4>(lhs, rhs);

    case loco::DataType::S8:
      return has_same_elements<loco::DataType::S8>(lhs, rhs);

    case loco::DataType::S16:
      return has_same_elements<loco::DataType::S16>(lhs, rhs);

    case loco::DataType::S32:
      return has_same_elements<loco::DataType::S32>(lhs, rhs);

    case loco::DataType::S64:
      return has_same_elements<loco::DataType::S64>(lhs, rhs);

    case loco::DataType::U4:
      return has_same_elements<loco::DataType::U4>(lhs, rhs);

    case loco::DataType::U8:
      return has_same_elements<loco::DataType::U8>(lhs, rhs);

    case loco::DataType::BOOL:
      return has_same_elements<loco::DataType::BOOL>(lhs, rhs);

    default:
      break;
  }

  return false;
}

uint32_t get_buffer_id(FlatBufferBuilder &builder, SerializedModelData &md, luci::CircleConst *node)
{
  if (node != nullptr)
  {
    // When buffer with same values is found, use the buffer id.
    for (auto key_value : md._cached_buffer_id)
    {
      if (has_same_values(key_value.first, node))
        return key_value.second;
    }

    // When buffer with same values is not found, generate new buffer
    auto buffer = encodeOpBuffer(builder, node);

    auto buffer_id = static_cast<uint32_t>(md._buffers.size());
    md._buffers.push_back(buffer);

    // Cache the newly generated buffer id
    md._cached_buffer_id.insert({node, buffer_id});

    return buffer_id;
  }
  else
  {
    // When there is no CircleConst, there is nothing to cache.
    // So return new buffer id.
    auto buffer = encodeOpBuffer(builder);

    auto buffer_id = static_cast<uint32_t>(md._buffers.size());
    md._buffers.push_back(buffer);

    return buffer_id;
  }
}

void exportOpDefinedTensor(const CircleTensorInfo &info, FlatBufferBuilder &builder,
                           SerializedModelData &md, SerializedGraphData &gd)
{
  // Create and register output tensor shape
  flatbuffers::Offset<Vector<int32_t>> shape_offset;
  flatbuffers::Offset<Vector<int32_t>> shape_signature_offset;
  if (info.shape_status() == ShapeStatus::VALID)
  {
    shape_offset = encodeShape(builder, info.shape());
    shape_signature_offset = encodeShapeSignature(builder, info.shape());
  }

  auto quantparam = encodeQuantizationParameters(builder, info.quantparam());

  auto sparsityparam = encodeSparsityParameters(builder, info.sparsityparam());

  auto buffer_id = get_buffer_id(builder, md, info.content());

  auto name_offset = builder.CreateString(info.name());

  auto is_variable = info.is_variable();

  auto tensor_offset = CreateTensor(builder, shape_offset, info.dtype(), buffer_id, name_offset,
                                    quantparam, is_variable, sparsityparam, shape_signature_offset);
  gd._tensors.push_back(tensor_offset);
}

} // namespace

namespace luci
{

void prepareModelData(FlatBufferBuilder &builder, SerializedModelData &md)
{
  // add one empty buffer
  //   note: this follows TFLite
  //   note: there's a comment in tflite fbs file
  //   - Note the 0th entry of this array must be an empty buffer (sentinel).
  //   - This is a convention so that tensors without a buffer can provide 0 as
  //   - their buffer.
  auto buffer = encodeOpBuffer(builder);
  md._buffers.push_back(buffer);
}

void exportOpDefinedTensors(loco::Graph *g, FlatBufferBuilder &builder, SerializedModelData &md,
                            SerializedGraphData &gd)
{
  CircleTensorContext tensor_ctx;

  // NOTE There may exist dangle CircleInput that is not visited with postorder_traversal()
  //      All dangle CircleOutput should be visited by postorder_traversal()
  auto nodes = g->nodes();
  for (uint32_t n = 0; n < nodes->size(); ++n)
  {
    auto node = dynamic_cast<luci::CircleInput *>(nodes->at(n));
    if (node != nullptr)
      allocateCircleTensor(node, tensor_ctx);
  }

  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    CircleNode *circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (dynamic_cast<const luci::CircleInput *>(circle_node) != nullptr)
      continue;
    allocateCircleTensor(circle_node, tensor_ctx);
  }

  for (const auto &tensor_info : tensor_ctx)
  {
    exportOpDefinedTensor(tensor_info, builder, md, gd);
  }
}

} // namespace luci
