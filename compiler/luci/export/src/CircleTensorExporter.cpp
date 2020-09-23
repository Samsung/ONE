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
#include "TypeBridge.h"

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

using namespace circle;
using namespace flatbuffers;

namespace
{

using namespace luci;

class CircleTensoInfo
{
public:
  CircleTensoInfo() = default;

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

private:
  std::string _name;

  circle::TensorType _dtype{circle::TensorType_FLOAT32};
  ShapeDescription _shape{};
  luci::ShapeStatus _shape_status{luci::ShapeStatus::UNDEFINED};

  luci::CircleConst *_content = nullptr;
  luci::CircleQuantParam *_quantparam = nullptr;
  luci::SparsityParam *_sparsityparam = nullptr;
};

using CircleTensorContext = std::vector<CircleTensoInfo>;

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
  // auto tensor_name = "t_" + std::to_string(tensor_index);
  std::string tensor_name = node->name();
  if (tensor_name.empty())
    tensor_name = "t_" + std::to_string(tensor_index);
  INFO(l) << "[luci] Tensor for " << tensor_name << ": " << tensor_index << std::endl;

  CircleTensoInfo tensor_info;

  tensor_info.name(tensor_name);
  tensor_info.dtype(to_circle_tensortype(luci::node_dtype(node)));
  if (node->shape_status() == ShapeStatus::VALID)
    tensor_info.shape(to_shape_description(luci::node_shape(node)));
  tensor_info.shape_status(node->shape_status());

  tensor_info.content(dynamic_cast<luci::CircleConst *>(node));
  tensor_info.quantparam(node->quantparam());
  tensor_info.sparsityparam(node->sparsityparam());

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
  bool visit(luci::CircleIfOut *) final { return true; }
  bool visit(luci::CircleSplitOut *) final { return true; }
  bool visit(luci::CircleSplitVOut *) final { return true; }
  bool visit(luci::CircleTopKV2Out *) final { return true; }
  bool visit(luci::CircleUnpackOut *) final { return true; }
  bool visit(luci::CircleWhileOut *) final { return true; }

  bool visit(luci::CircleIf *node) final
  {
    store_outputs(node, node->output_count());
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
  return builder.CreateVector(shape._dims);
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
flatbuffers::Offset<circle::Buffer> encodeOpBuffer(FlatBufferBuilder &builder, luci::CircleConst *c)
{
  switch (c->dtype())
  {
    case loco::DataType::FLOAT32:
      return encodeOpBufferByDType<loco::DataType::FLOAT32>(builder, c);
    case loco::DataType::S8:
      return encodeOpBufferByDType<loco::DataType::S8>(builder, c);
    case loco::DataType::S16:
      return encodeOpBufferByDType<loco::DataType::S16>(builder, c);
    case loco::DataType::S32:
      return encodeOpBufferByDType<loco::DataType::S32>(builder, c);
    case loco::DataType::S64:
      return encodeOpBufferByDType<loco::DataType::S64>(builder, c);
    case loco::DataType::U8:
      return encodeOpBufferByDType<loco::DataType::U8>(builder, c);
    case loco::DataType::BOOL:
      return encodeOpBufferByDType<loco::DataType::BOOL>(builder, c);
    default:
      break;
  }

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
  flatbuffers::Offset<flatbuffers::Vector<int32_t>> traversal_order =
      builder.CreateVector(sparsityparam->traversal_order);
  flatbuffers::Offset<flatbuffers::Vector<int32_t>> block_map =
      builder.CreateVector(sparsityparam->block_map);
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<DimensionMetadata>>> dim_metadata;

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

    auto circle_dim_metadata_builder = circle::DimensionMetadataBuilder{builder};

    circle_dim_metadata_builder.add_format(to_circle_dimensiontype(it.format()));
    circle_dim_metadata_builder.add_dense_size(it.dense_size());
    circle_dim_metadata_builder.add_array_segments(circle_array_segments);
    circle_dim_metadata_builder.add_array_segments_type(circle_array_segments_type);
    circle_dim_metadata_builder.add_array_indices(circle_array_indices);
    circle_dim_metadata_builder.add_array_indices_type(circle_array_indices_type);
    auto dim_metadata = circle_dim_metadata_builder.Finish();
    dim_metadata_vec.emplace_back(dim_metadata);
  }
  dim_metadata = builder.CreateVector(dim_metadata_vec);

  return circle::CreateSparsityParameters(builder, traversal_order, block_map, dim_metadata);
}

void exportOpDefinedTensor(const CircleTensoInfo &info, FlatBufferBuilder &builder,
                           SerializedModelData &md, SerializedGraphData &gd)
{
  // Create and register output tensor shape
  flatbuffers::Offset<Vector<int32_t>> shape_offset;
  if (info.shape_status() == ShapeStatus::VALID)
    shape_offset = encodeShape(builder, info.shape());

  // encode and register output tensor buffer
  auto buffer =
      info.content() == nullptr ? encodeOpBuffer(builder) : encodeOpBuffer(builder, info.content());

  auto quantparam = encodeQuantizationParameters(builder, info.quantparam());

  auto sparsityparam = encodeSparsityParameters(builder, info.sparsityparam());

  auto buffer_id = static_cast<uint32_t>(md._buffers.size());
  md._buffers.push_back(buffer);

  auto name_offset = builder.CreateString(info.name());
  auto tensor_offset = CreateTensor(builder, shape_offset, info.dtype(), buffer_id, name_offset,
                                    quantparam, /*is_variable*/ false, sparsityparam);
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
