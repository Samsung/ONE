/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TFLTensorExporter.h"
#include "TFLTypeInference.h"
#include "ShapeInference.h"

// TODO Fix include style
#include "loco/IR/Algorithm.h"
#include "loco/IR/CanonicalNode.h"
#include "loco/IR/CanonicalNodeVisitor.h"
#include "loco/IR/DataTypeTraits.h"

#include "Dialect/IR/TFLNodes.h"

#include <oops/InternalExn.h>

using namespace tflite;
using namespace flatbuffers;

namespace
{

using namespace exo;
using namespace exo::tflite_detail;

class TFLTensorInfo
{
public:
  TFLTensorInfo() = default;

public:
  void name(const std::string &name) { _name = name; }
  const std::string &name(void) const { return _name; }

public:
  const tflite::TensorType &dtype(void) const { return _dtype; }
  void dtype(const tflite::TensorType &dtype) { _dtype = dtype; }

  const ShapeDescription &shape(void) const { return _shape; }
  void shape(const ShapeDescription &shape) { _shape = shape; }

public:
  locoex::TFLConst *tfl_content(void) const { return _tfl_content; }
  void tfl_content(locoex::TFLConst *c) { _tfl_content = c; }

private:
  std::string _name;

  tflite::TensorType _dtype{TensorType_FLOAT32};
  ShapeDescription _shape{};

  // TODO Find a better design
  loco::ConstGen *_content = nullptr; // TODO deprecate
  locoex::TFLConst *_tfl_content = nullptr;
};

using TFLTensorContext = std::vector<TFLTensorInfo>;

struct NoOpDetector final : public loco::CanonicalNodeMutableVisitor<bool>
{
  bool visit(loco::BiasEncode *) final
  {
    // BiasEncode is always noop
    return true;
  }

  bool visit(loco::FilterEncode *node) final
  {
    auto encoder = loco::must_cast<loco::PermutingEncoder<loco::Domain::Filter> *>(node->encoder());
    auto perm = encoder->perm();

    return isNHWC(perm);
  }

  bool visit(loco::FeatureEncode *node) final
  {
    auto encoder =
      loco::must_cast<loco::PermutingEncoder<loco::Domain::Feature> *>(node->encoder());
    auto perm = encoder->perm();
    return isNHWC(perm);
  }

  bool visit(loco::FeatureDecode *node) final
  {
    auto decoder =
      loco::must_cast<loco::PermutingDecoder<loco::Domain::Feature> *>(node->decoder());
    auto perm = decoder->perm();
    return isNHWC(perm);
  }

  // Return false by default
  bool visit(loco::Node *) final { return false; }
};

bool isNoOp(loco::Node *node)
{
  if (auto canonical_node = dynamic_cast<loco::CanonicalNode *>(node))
  {
    NoOpDetector d;
    return canonical_node->accept(&d);
  }
  return false;
}

void allocateTFLiteTensor(loco::Node *node, TFLTensorContext &ctx)
{
  if (isNoOp(node))
  {
    assert(node->arity() == 1 && node->arg(0) != nullptr);
    set_tensor_index(node, get_tensor_index(node->arg(0)));
    return;
  }

  auto tensor_index = static_cast<TFLTensorIndex>(ctx.size());
  // TODO Use Graph-level metadata for Input & Output
  auto tensor_name = "t_" + std::to_string(tensor_index);

  TFLTensorInfo tensor_info;

  tensor_info.name(tensor_name);
  tensor_info.dtype(TypeInference::get(node));
  tensor_info.shape(ShapeInference::get(node));

  if (auto const_node = dynamic_cast<locoex::TFLConst *>(node))
    tensor_info.tfl_content(const_node);

  set_tensor_index(node, tensor_index);

  ctx.emplace_back(tensor_info);
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

flatbuffers::Offset<tflite::Buffer> encodeOpBuffer(FlatBufferBuilder &builder)
{
  return CreateBuffer(builder);
}

template <typename NodeT>
flatbuffers::Offset<tflite::Buffer> encodeOpBuffer(FlatBufferBuilder &builder, NodeT *)
{
  return CreateBuffer(builder);
}

template <loco::DataType DT>
flatbuffers::Offset<tflite::Buffer> encodeOpBufferByDType(FlatBufferBuilder &builder,
                                                          locoex::TFLConst *c)
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
flatbuffers::Offset<tflite::Buffer> encodeOpBuffer(FlatBufferBuilder &builder, locoex::TFLConst *c)
{
  if (c->dtype() == loco::DataType::FLOAT32)
  {
    return encodeOpBufferByDType<loco::DataType::FLOAT32>(builder, c);
  }
  else if (c->dtype() == loco::DataType::S32)
  {
    return encodeOpBufferByDType<loco::DataType::S32>(builder, c);
  }

  INTERNAL_EXN_V("Unsupported datatype", oops::to_uint32(c->dtype()));
}

} // namespace

namespace exo
{
namespace tflite_detail
{

void exportOpDefinedTensor(const TFLTensorInfo &info, FlatBufferBuilder &builder,
                           SerializedModelData &gd)
{
  // Create and register output tensor shape
  auto shape_offset = encodeShape(builder, info.shape());

  // encode and register output tensor buffer
  auto buffer = info.tfl_content() == nullptr ? encodeOpBuffer(builder)
                                              : encodeOpBuffer(builder, info.tfl_content());

  auto buffer_id = static_cast<uint32_t>(gd._buffers.size());
  gd._buffers.push_back(buffer);

  auto name_offset = builder.CreateString(info.name());
  auto tensor_offset = CreateTensor(builder, shape_offset, info.dtype(), buffer_id, name_offset,
                                    /*quantization*/ 0, /*is_variable*/ false);
  gd._tensors.push_back(tensor_offset);
}

void exportOpDefinedTensors(loco::Graph *g, FlatBufferBuilder &builder, SerializedModelData &gd)
{
  TFLTensorContext tensor_ctx;

  for (auto node : loco::postorder_traversal(loco::output_nodes(g)))
  {
    allocateTFLiteTensor(node, tensor_ctx);
  }

  // add one empty buffer
  //   note: there's a comment in tflite fbs file
  //   - Note the 0th entry of this array must be an empty buffer (sentinel).
  //   - This is a convention so that tensors without a buffer can provide 0 as
  //   - their buffer.
  auto buffer = encodeOpBuffer(builder);
  gd._buffers.push_back(buffer);

  for (const auto &tensor_info : tensor_ctx)
  {
    exportOpDefinedTensor(tensor_info, builder, gd);
  }
}

} // namespace tflite_detail
} // namespace exo
