/*
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __BASE_LOADER_BASE_LOADER_H__
#define __BASE_LOADER_BASE_LOADER_H__

#include "ir/Graph.h"
#include "ir/Shape.h"
#include "ir/Operations.Include.h"

#include "flatbuffers/flexbuffers.h"

#include <map>
#include <memory>
#include <fstream>
#include <limits>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <util/logging.h>

namespace onert
{
namespace base_loader
{

template <typename LoaderDomain> class BaseLoader
{
protected:
  using Verifier = typename LoaderDomain::Verifier;
  using ActivationFunctionType = typename LoaderDomain::ActivationFunctionType;
  using Buffer = typename LoaderDomain::Buffer;
  using BuiltinOperator = typename LoaderDomain::BuiltinOperator;
  using CustomOptionsFormat = typename LoaderDomain::CustomOptionsFormat;
  using Metadata = typename LoaderDomain::Metadata;
  using Model = typename LoaderDomain::Model;
  using Operator = typename LoaderDomain::Operator;
  using Padding = typename LoaderDomain::Padding;
  using Pool2DOptions = typename LoaderDomain::Pool2DOptions;
  using SubGraph = typename LoaderDomain::SubGraph;
  using Tensor = typename LoaderDomain::Tensor;
  using TensorType = typename LoaderDomain::TensorType;
  using DimensionType = typename LoaderDomain::DimensionType;
  using SparseIndexVector = typename LoaderDomain::SparseIndexVector;

protected:
  bool isOptionalInputTensor(std::int32_t idx) { return idx == -1; }
  virtual bool allowOptionalInputTensor(BuiltinOperator) = 0;

public:
  /**
   * @brief Construct a new Loader object
   *
   * @param model reference to model
   */
  explicit BaseLoader(std::unique_ptr<ir::Model> &model)
    : _base{nullptr}, _pagesize(getpagesize()), _fd(-1), _model(model), _domain_model{nullptr}
  {
    _use_mmaped_data = util::getConfigBool(util::config::USE_MMAPED_DATA);
  }

  /**
   * @brief Load a model from file
   *
   * @param file_path
   */
  void loadFromFile(const std::string &file_path);
  /**
   * @brief Load a model from a buffer
   *
   * @param buffer buffer pointer
   * @param size buffer size
   */
  void loadFromBuffer(uint8_t *buffer, size_t size);

protected:
  ~BaseLoader() = default;
  void loadModel();

  // Helper functions
  ir::Activation convertActivation(ActivationFunctionType type);
  ir::DataType tensorTypeToDataType(TensorType type);
  ir::OperandIndex tensorIdxToOperandIdx(int32_t tensorIdx);
  flexbuffers::Map getCustomOpAttrMap(const Operator *op);

  // Create operands form tflite::Tensor
  ir::OperandIndex loadOperand(const Tensor *tensor, ir::Graph &subg);
  void loadQuantization(const Tensor *tensor, ir::TypeInfo &typeInfo);
  void loadSparsity(const Tensor *tensor, ir::TypeInfo &typeInfo);
  void loadOperationIO(const Operator *op, ir::OperandIndexSequence &inputs,
                       ir::OperandIndexSequence &outputs);
  // Create operations from Operator
  void loadOperation(const Operator *op, ir::Graph &subg);
  // Load Strides and Paddings from options to param
  template <typename Param, typename OptionsType>
  void loadStridesAndPaddings(Param &param, const OptionsType *options);
  // Load Pool2D param
  template <typename Param> void loadPool2DOptions(Param &param, const Pool2DOptions *options);
  // Get BuiltinOperator
  BuiltinOperator getBuiltinOperator(const Operator *op)
  {
    auto const builtin_opcode = _domain_model->operator_codes()->Get(op->opcode_index());
    auto builtin_op = builtin_opcode->builtin_code();
    if (builtin_op < BuiltinOperator::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES)
      builtin_op = static_cast<BuiltinOperator>(builtin_opcode->deprecated_builtin_code());

    return builtin_op;
  }

private:
  std::unique_ptr<ir::Data> loadMetadata(const uint32_t buffer_idx);
  virtual std::unique_ptr<ir::Graph> loadSubgraph(const SubGraph *subg) = 0;
  // Operations
  template <typename OpIR, typename... Args>
  const OpIR *loadOperationTo(const Operator *op, ir::Graph &subg, Args &&...args);

  void loadAddV2(const Operator *op, ir::Graph &subg);
  void loadArgMinMax(const Operator *op, ir::Graph &subg, bool is_argmax);
  void loadBatchMatMul(const Operator *op, ir::Graph &subg);
  void loadBinaryArithmetic(const Operator *op, ir::Graph &subg,
                            ir::operation::BinaryArithmetic::ArithmeticType op_type);
  void loadComparison(const Operator *op, ir::Graph &subg);
  void loadConcatenation(const Operator *op, ir::Graph &subg);
  void loadConv2D(const Operator *op, ir::Graph &subg);
  void loadCustom(const Operator *op, ir::Graph &subg);
  void loadDepthToSpace(const Operator *op, ir::Graph &subg);
  void loadDepthwiseConv2D(const Operator *op, ir::Graph &subg);
  void loadEinsum(const Operator *op, ir::Graph &subg);
  void loadElementwiseActivation(const Operator *op, ir::Graph &subg,
                                 ir::operation::ElementwiseActivation::Type op_type,
                                 float alpha = 0.f, float beta = 0.f);
  void loadElementwiseBinary(const Operator *op, ir::Graph &subg,
                             ir::operation::ElementwiseBinary::ElementwiseBinaryType op_type);
  void loadElementwiseUnary(const Operator *op, ir::Graph &subg,
                            ir::operation::ElementwiseUnary::Type op_type);
  void loadFC(const Operator *op, ir::Graph &subg);
  void loadFusedBatchNorm(const Operator *op, ir::Graph &subg);
  void loadGather(const Operator *op, ir::Graph &subg);
  void loadIf(const Operator *op, ir::Graph &subg);
  void loadLeakyRelu(const Operator *op, ir::Graph &subg);
  void loadLogSoftmax(const Operator *op, ir::Graph &subg);
  void loadDetectionPostProcess(const Operator *op, ir::Graph &subg);
  void loadOneHot(const Operator *op, ir::Graph &subg);
  void loadPack(const Operator *op, ir::Graph &subg);
  void loadPool2D(const Operator *op, ir::Graph &subg, ir::operation::Pool2D::PoolType op_type);
  void loadReduce(const Operator *op, ir::Graph &subg,
                  ir::operation::Reduce::ReduceType reduce_type);
  void loadReduceAll(const Operator *op, ir::Graph &subg);
  void loadReshape(const Operator *op, ir::Graph &subg);
  void loadResizeBilinear(const Operator *op, ir::Graph &subg);
  void loadResizeNearestNeighbor(const Operator *op, ir::Graph &subg);
  void loadSoftmax(const Operator *op, ir::Graph &subg);
  void loadSpaceToDepth(const Operator *op, ir::Graph &subg);
  void loadSplit(const Operator *op, ir::Graph &subg);
  void loadSplitV(const Operator *op, ir::Graph &subg);
  void loadSqueeze(const Operator *op, ir::Graph &subg);
  void loadStridedSlice(const Operator *op, ir::Graph &subg);
  void loadTransposeConv(const Operator *op, ir::Graph &subg);
  void loadUnidirectionalSequenceLSTM(const Operator *op, ir::Graph &subg);
  void loadUnpack(const Operator *op, ir::Graph &subg);
  void loadWhile(const Operator *op, ir::Graph &subg);

  void verifySubgraphIndex(int subg_index)
  {
    const auto num_subgraphs = _domain_model->subgraphs()->size();
    if (subg_index < 0 || subg_index >= static_cast<int32_t>(num_subgraphs))
      throw std::runtime_error{std::string{"Invalid subgraph index - "} +
                               std::to_string(subg_index)};
  }

protected:
  // Base address for mapped region for loading (if needed)
  uint8_t *_base;
  // Memory page size
  int32_t _pagesize;
  // loaded file description
  int _fd;
  // Reference to ir::model (to be loaded from _domain_model)
  std::unique_ptr<ir::Model> &_model;
  const Model *_domain_model;
  // Maps Tensor indices to onert Operands.
  std::vector<ir::OperandIndex> _tensor_to_operand;
  std::unordered_map<ir::OperandIndex, std::string> _tensor_names;
  // Verifier
  std::unique_ptr<Verifier> _verifier;
  // Boolean flag to use MMAPED_DATA
  bool _use_mmaped_data = false;

  std::unordered_map<uint32_t /* Buffer Index in circle file */, std::shared_ptr<ir::Data>>
    _buf_to_data;
};

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::BaseLoader::loadFromFile(const std::string &file_path)
{
  _fd = open(file_path.c_str(), O_RDONLY);
  if (_fd < 0)
  {
    throw std::runtime_error("Failed to open file " + file_path);
  }

  struct stat file_stat;
  if (fstat(_fd, &file_stat) != 0)
  {
    throw std::runtime_error("Fstat failed or file " + file_path + " is not a regular file");
  }
  int size = file_stat.st_size;

  // Map model file into memory region
  _base = static_cast<uint8_t *>(mmap(NULL, size, PROT_READ, MAP_PRIVATE, _fd, 0));
  if (_base == MAP_FAILED)
  {
    close(_fd);
    throw std::runtime_error("mmap failed - " + std::string(strerror(errno)));
  }

  _verifier = std::make_unique<Verifier>(reinterpret_cast<const std::uint8_t *>(_base), size);

  loadModel();
  munmap(_base, size);

  close(_fd);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::BaseLoader::loadFromBuffer(uint8_t *buffer, size_t size)
{
  _base = buffer;
  _verifier = std::make_unique<Verifier>(reinterpret_cast<const std::uint8_t *>(_base), size);
  loadModel();
}

template <typename LoaderDomain>
std::unique_ptr<ir::Data>
BaseLoader<LoaderDomain>::BaseLoader::loadMetadata(const uint32_t buffer_idx)
{
  assert(_domain_model != nullptr);
  const auto *data = _domain_model->buffers()->Get(buffer_idx)->data();
  if (_fd == -1) // Model is from memory
  {
    return std::make_unique<ir::ExternalData>(data->data(), data->size());
  }
  else // Model is loaded(mmap'd) from a file
  {
    size_t data_size = data->size();
    ptrdiff_t offset_start = data->data() - _base;
    ptrdiff_t offset_end = offset_start + data_size;

    ptrdiff_t page_start = (offset_start / _pagesize) * _pagesize;
    size_t mapping_size = offset_end - page_start;

    // Since metadata is not access often in inference/training time, always use mmaped-data
    // Ref : https://github.com/Samsung/ONE/issues/3961#issuecomment-681750231
    return std::make_unique<ir::MMapedData>(_fd, page_start, mapping_size, offset_start, data_size);
  }
}

template <typename LoaderDomain>
ir::Activation
BaseLoader<LoaderDomain>::BaseLoader::convertActivation(const ActivationFunctionType type)
{
  switch (type)
  {
    case ActivationFunctionType::ActivationFunctionType_NONE:
      return ir::Activation::NONE;
    case ActivationFunctionType::ActivationFunctionType_RELU:
      return ir::Activation::RELU;
    case ActivationFunctionType::ActivationFunctionType_RELU_N1_TO_1:
      return ir::Activation::RELU1;
    case ActivationFunctionType::ActivationFunctionType_RELU6:
      return ir::Activation::RELU6;
    case ActivationFunctionType::ActivationFunctionType_TANH:
      return ir::Activation::TANH;
    default:
      throw std::runtime_error(std::string("Unsupported or invalid activation type: ") +
                               std::to_string(static_cast<int>(type)));
  }
}

template <typename LoaderDomain>
ir::DataType BaseLoader<LoaderDomain>::BaseLoader::tensorTypeToDataType(const TensorType type)
{
  switch (type)
  {
    case TensorType::TensorType_FLOAT32:
      return ir::DataType::FLOAT32;
    case TensorType::TensorType_FLOAT16:
      return ir::DataType::FLOAT16;
    case TensorType::TensorType_INT32:
      return ir::DataType::INT32;
    case TensorType::TensorType_UINT8:
      return ir::DataType::QUANT_UINT8_ASYMM;
    case TensorType::TensorType_INT64:
      return ir::DataType::INT64;
    // case TensorType::TensorType_STRING:
    case TensorType::TensorType_BOOL:
      return ir::DataType::BOOL8;
    case TensorType::TensorType_INT16:
      return ir::DataType::QUANT_INT16_ASYMM;
    // case TensorType::TensorType_COMPLEX64
    case TensorType::TensorType_INT8:
      return ir::DataType::QUANT_INT8_ASYMM;
    // case TensorType::TensorType_FLOAT64
    case TensorType::TensorType_UINT32:
      return ir::DataType::UINT32;
    default:
      throw std::runtime_error(
        std::string("Unsupported tensor type: ").append(EnumNameTensorType(type)));
  }
}

template <typename LoaderDomain>
ir::OperandIndex BaseLoader<LoaderDomain>::BaseLoader::tensorIdxToOperandIdx(int32_t tensorIdx)
{
  return isOptionalInputTensor(tensorIdx) ? ir::OperandIndex() : _tensor_to_operand[tensorIdx];
}

template <typename LoaderDomain>
flexbuffers::Map BaseLoader<LoaderDomain>::BaseLoader::getCustomOpAttrMap(const Operator *op)
{
  size_t custom_op_data_size = op->custom_options()->size();
  auto custom_op_data = op->custom_options()->Data();
  auto data_root = flexbuffers::GetRoot(custom_op_data, custom_op_data_size);
  return data_root.AsMap();
}

/* Copy is copied from tensorflow lite */
template <typename T> bool Copy(const T *data_ptr, std::vector<uint16_t> &arr)
{
  if (data_ptr->values() == nullptr)
  {
    return false;
  }

  int size = data_ptr->values()->size();
  arr.reserve(size);
  for (int i = 0; i < size; i++)
  {
    arr.emplace_back(static_cast<uint16_t>(data_ptr->values()->Get(i)));
  }
  return true;
}

template <typename LoaderDomain>
ir::OperandIndex BaseLoader<LoaderDomain>::loadOperand(const Tensor *tensor, ir::Graph &subg)
{
  ir::Shape shape;
  // Shape
  const auto *tensor_shape = tensor->shape();
  if (tensor_shape != nullptr)
  {
    for (const auto &dim : *tensor_shape)
    {
      shape.append(dim);
    }
  }

  // Note for tensor->shape_signature()
  // We don't handle shape signature
  //    How we handle:
  //       If shape_signature[k] == -1, we will use tensor->shape()[k] == 1
  //       If app wants to change the input shape, call nnfw_apply_input_tensorinfo() can
  //       be used.

  // TypeInfo
  ir::TypeInfo type_info(tensorTypeToDataType(tensor->type()));
  loadQuantization(tensor, type_info);
  loadSparsity(tensor, type_info);

  // Create operand
  const auto operand_index = subg.addOperand(shape, type_info);

  // Constant tensors are indicated by non-empty data.
  const auto *data = _domain_model->buffers()->Get(tensor->buffer())->data();
  if (data != nullptr)
  {
    using std::ptrdiff_t;
    std::shared_ptr<ir::Data> data_obj;

    if (_fd == -1) // Model is from memory
    {
      data_obj = std::make_shared<ir::ExternalData>(data->data(), data->size());
    }
    else // Model is loaded(mmap'd) from a file
    {
      size_t data_size = data->size();
      ptrdiff_t unaligned_offset_start = data->data() - _base;
      ptrdiff_t offset_end = unaligned_offset_start + data_size;

      // Calculated aligned offset from base address of mapped region
      // munmap accepts memory address which is a multiple of the pagesize
      ptrdiff_t aligned_offset_start = (unaligned_offset_start / _pagesize) * _pagesize;
      size_t mmap_size = offset_end - aligned_offset_start;

      uint32_t buf_idx = tensor->buffer();
      auto buffer_found = _buf_to_data.find(buf_idx);

      if (buffer_found != _buf_to_data.end())
      {
        // Another tensor points this buffer and its matching Data(either CachedData or MMapedData)
        // was already created. Let's reuse the Data
        data_obj = buffer_found->second;
      }
      else if (_use_mmaped_data)
      {
        data_obj = std::make_shared<ir::MMapedData>(_fd, aligned_offset_start, mmap_size,
                                                    unaligned_offset_start, data_size);
        _buf_to_data[buf_idx] = data_obj;
      }
      else
      {
        size_t offset = unaligned_offset_start - aligned_offset_start;
        uint8_t *mmap_base = static_cast<uint8_t *>(
          mmap(NULL, mmap_size, PROT_READ, MAP_PRIVATE, _fd, aligned_offset_start));

        data_obj = std::make_shared<ir::CachedData>(mmap_base + offset, data_size);
        _buf_to_data[buf_idx] = data_obj;

        munmap(mmap_base, mmap_size);
      }
    }
    subg.setOperandValue(operand_index, std::move(data_obj));
  }

  _tensor_names.emplace(operand_index, tensor->name()->str());

  // Variable
  if (tensor->is_variable())
  {
    if (data != nullptr)
      throw std::runtime_error("Variable tensor with buffer is not supported!");

    subg.operands().at(operand_index).info().setAsVariable();
  }

  return operand_index;
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadQuantization(const Tensor *tensor, ir::TypeInfo &typeInfo)
{
  auto q_params = tensor->quantization();
  if (q_params == nullptr || q_params->scale() == nullptr || q_params->scale()->size() == 0)
  {
    typeInfo.quantization(0., 0);
    return;
  }
  if (q_params->zero_point() == nullptr)
  {
    throw std::runtime_error("Quantization params: scale is not null, but zero_point is null.");
  }
  const size_t num_scales = q_params->scale()->size();
  if (num_scales != q_params->zero_point()->size())
  {
    throw std::runtime_error("Quantization params: scale size != zero_point size");
  }
  std::vector<float> scales;
  std::vector<int32_t> zero_points;
  scales.resize(num_scales);
  zero_points.resize(num_scales);
  for (size_t i = 0; i < num_scales; ++i)
  {
    scales[i] = q_params->scale()->Get(i);
    // zero_point is defined as long (i64) in schema while TypeInfo's zero_point is int32_t.
    // int64_t is used instead of long because long is 4 byte in most 32bit architecture.
    int64_t zero_point = q_params->zero_point()->Get(i);
    if (zero_point < std::numeric_limits<int32_t>::min() ||
        zero_point > std::numeric_limits<int32_t>::max())
      throw std::runtime_error("Zero_point is out of int32 range.");
    zero_points[i] = static_cast<int32_t>(zero_point);
  }
  auto details = q_params->details_as_CustomQuantization();
  if (details != nullptr)
    throw std::runtime_error("Custom Quantization is not supported");
  typeInfo.quantization(std::move(scales), std::move(zero_points));
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadSparsity(const Tensor *tensor, ir::TypeInfo &typeInfo)
{
  auto src_sparsity = tensor->sparsity();
  if (src_sparsity != nullptr)
  {
    std::vector<uint16_t> w1_segments;
    std::vector<uint16_t> w1_indices;
    // check traversal_order
    if (src_sparsity->traversal_order())
    {
      const int traversal_order_size = src_sparsity->traversal_order()->size();
      for (int i = 0; i < traversal_order_size; ++i)
      {
        if (i != src_sparsity->traversal_order()->Get(i))
          throw std::runtime_error("traversal_order [0, 1, ..., n-1] is only supported.");
      }
    }
    // check block_map
    int block_rank = 0;
    if (src_sparsity->block_map())
    {
      block_rank = src_sparsity->block_map()->size();
      for (int i = 0; i < block_rank; ++i)
      {
        if (i != src_sparsity->block_map()->Get(i))
          throw std::runtime_error("block_map [0, 1, ..., n-1] is only supported.");
      }
    }
    // load metadata
    const auto dim_metadata_size = src_sparsity->dim_metadata()->size();
    const auto dense_rank = tensor->shape() ? tensor->shape()->size() : 0;
    if (dense_rank + block_rank != dim_metadata_size)
      throw std::runtime_error("sparsity dim_metadata length is wrong.");
    bool random_sparsity = dim_metadata_size == 2 && block_rank == 0;
    bool block2D_sparsity = dim_metadata_size == 4 && block_rank == 2;
    if (dim_metadata_size != !random_sparsity && !block2D_sparsity)
      throw std::runtime_error(
        "sparsity is supported only for 2D tensor with random or 16x1 block sparsity.");

    const auto *src_metadata = src_sparsity->dim_metadata()->Get(0);
    if (src_metadata->format() != DimensionType::DimensionType_DENSE)
      throw std::runtime_error("sparse tensor dim[0] is not DENSE");
    src_metadata = src_sparsity->dim_metadata()->Get(1);
    if (src_metadata->format() != DimensionType::DimensionType_SPARSE_CSR)
      throw std::runtime_error("sparse tensor dim[0] is not SPARSE_CSR");
    auto ParseSparseIndexVector = [src_metadata, &w1_segments, &w1_indices]() {
      if (src_metadata->array_segments() == nullptr || src_metadata->array_indices() == nullptr)
        return false;
      bool status = true;
      /* `onert` inernally uses uint16 type regardless of the value of
         the array_segments_type and array_indices_type */
      switch (src_metadata->array_segments_type())
      {
        case SparseIndexVector::SparseIndexVector_Int32Vector:
          throw std::runtime_error("sparse tensor with int32 segment type is not supported");
        case SparseIndexVector::SparseIndexVector_Uint16Vector:
          status = Copy(src_metadata->array_segments_as_Uint16Vector(), w1_segments);
          break;
        case SparseIndexVector::SparseIndexVector_Uint8Vector:
          status = Copy(src_metadata->array_segments_as_Uint8Vector(), w1_segments);
          break;
        default:
          return false;
      }
      if (status != true)
        return false;
      switch (src_metadata->array_indices_type())
      {
        case SparseIndexVector::SparseIndexVector_Int32Vector:
          throw std::runtime_error("sparse tensor with int32 indices type is not supported");
        case SparseIndexVector::SparseIndexVector_Uint16Vector:
          return Copy(src_metadata->array_indices_as_Uint16Vector(), w1_indices);
        case SparseIndexVector::SparseIndexVector_Uint8Vector:
          return Copy(src_metadata->array_indices_as_Uint8Vector(), w1_indices);
        default:
          break;
      }
      return false;
    };
    if (ParseSparseIndexVector() == false)
      throw std::runtime_error("Error during parsing sparsity index information");
    // Get block size
    std::vector<int32_t> block_size;
    for (int i = 0; i < block_rank; ++i)
    {
      auto block_metadata = src_sparsity->dim_metadata()->Get(dense_rank + i);
      if (block_metadata->format() != DimensionType::DimensionType_DENSE)
        throw std::runtime_error("block dimension must be DENSE.");
      block_size.push_back(block_metadata->dense_size());
    }
    typeInfo.sparsity(std::make_shared<ir::Sparsity>(std::move(w1_segments), std::move(w1_indices),
                                                     std::move(block_size)));
  }
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadOperationIO(const Operator *op, ir::OperandIndexSequence &inputs,
                                               ir::OperandIndexSequence &outputs)
{
  for (const std::int32_t idx : *op->inputs())
  {
    // Optional tensors are not supported yet except for FULLY_CONNECTED and BCQ_FULLY_CONNECTED
    auto check_optional_input = [&]() {
      auto builtin_code = getBuiltinOperator(op);
      if (isOptionalInputTensor(idx) && !allowOptionalInputTensor(builtin_code))
        throw std::runtime_error(
          std::string("loader doesn't support optional input tensor yet for ")
            .append(EnumNameBuiltinOperator(builtin_code)));
    };
    check_optional_input();
    inputs.append(tensorIdxToOperandIdx(idx));
  }

  for (const std::int32_t idx : *op->outputs())
  {
    outputs.append(tensorIdxToOperandIdx(idx));
  }
}

template <typename LoaderDomain>
template <typename Param, typename OptionsType>
void BaseLoader<LoaderDomain>::loadStridesAndPaddings(Param &param, const OptionsType *options)
{
  // Strides
  param.stride.vertical = options->stride_h();
  param.stride.horizontal = options->stride_w();
  // Paddings
  switch (options->padding())
  {
    case Padding::Padding_SAME:
      param.padding.type = ir::PaddingType::SAME;
      break;
    case Padding::Padding_VALID:
      param.padding.type = ir::PaddingType::VALID;
      break;
    default:
      throw std::runtime_error{"Invalid padding type"};
  }
  // param paddings indexes unused
}

template <typename LoaderDomain>
template <typename Param>
void BaseLoader<LoaderDomain>::loadPool2DOptions(Param &param, const Pool2DOptions *options)
{
  // Strides and Paddings
  if (options->stride_h() <= 0 || options->stride_w() <= 0)
    throw std::runtime_error{"Invalid stride vertical or horizontal - both must be bigger than 0"};
  loadStridesAndPaddings(param, options);
  // Filter width and height
  // Strides
  if (options->filter_width() <= 0 || options->filter_height() <= 0)
    throw std::runtime_error{"Invalid filter width or height - both must be bigger than 0"};
  param.kw = options->filter_width();
  param.kh = options->filter_height();
  // Activation
  param.activation = convertActivation(options->fused_activation_function());
}

template <typename LoaderDomain>
template <typename OpIR, typename... Args>
const OpIR *BaseLoader<LoaderDomain>::loadOperationTo(const Operator *op, ir::Graph &subg,
                                                      Args &&...args)
{
  static_assert(sizeof...(args) <= 1, "You can't have more than 1 arguments!");
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  loadOperationIO(op, inputs, outputs);

  std::unique_ptr<OpIR> new_op(new OpIR(inputs, outputs, std::forward<Args>(args)...));
  auto ret = new_op.get();
  subg.addOperation(std::move(new_op));

  return ret;
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadConv2D(const Operator *op, ir::Graph &subg)
{
  ir::operation::Conv2D::Param param;
  const auto *options = op->builtin_options_as_Conv2DOptions();
  param.activation = convertActivation(options->fused_activation_function());
  loadStridesAndPaddings(param, options);
  param.dilation.width_factor = options->dilation_w_factor();
  param.dilation.height_factor = options->dilation_h_factor();

  const auto conv = loadOperationTo<ir::operation::Conv2D>(op, subg, param);

  // TFLite support old hybrid quantization (float input/output, uint8 kernel)
  // but it interprets weight type as init8 internally
  const auto &input_operand =
    subg.operands().at(conv->getInputs().at(ir::operation::Conv2D::INPUT));
  auto &weights_operand = subg.operands().at(conv->getInputs().at(ir::operation::Conv2D::KERNEL));
  if (input_operand.typeInfo().type() == ir::DataType::FLOAT32 &&
      ((weights_operand.typeInfo().type() == ir::DataType::QUANT_UINT8_ASYMM) ||
       weights_operand.typeInfo().type() == ir::DataType::QUANT_INT8_ASYMM))
  {
    weights_operand.type(ir::DataType::QUANT_INT8_SYMM);
  }
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadDepthwiseConv2D(const Operator *op, ir::Graph &subg)
{
  ir::operation::DepthwiseConv2D::Param param;
  const auto *options = op->builtin_options_as_DepthwiseConv2DOptions();
  param.activation = convertActivation(options->fused_activation_function());
  loadStridesAndPaddings(param, options);
  param.multiplier = options->depth_multiplier();
  // Dilation h/w factor unused
  param.dilation.width_factor = options->dilation_w_factor();
  param.dilation.height_factor = options->dilation_h_factor();

  const auto dconv = loadOperationTo<ir::operation::DepthwiseConv2D>(op, subg, param);

  // TFLite does not support old hybrid quantization (float input/output, uint8 kernel)
  // for depthwise convolution.
  // But for consistency with Conv2D and FC, we interpret weight type as init8 internally
  const auto &input_operand =
    subg.operands().at(dconv->getInputs().at(ir::operation::DepthwiseConv2D::INPUT));
  auto &weights_operand =
    subg.operands().at(dconv->getInputs().at(ir::operation::DepthwiseConv2D::KERNEL));
  if (input_operand.typeInfo().type() == ir::DataType::FLOAT32 &&
      ((weights_operand.typeInfo().type() == ir::DataType::QUANT_UINT8_ASYMM) ||
       weights_operand.typeInfo().type() == ir::DataType::QUANT_INT8_ASYMM))
  {
    weights_operand.type(ir::DataType::QUANT_INT8_SYMM);
  }
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadTransposeConv(const Operator *op, ir::Graph &subg)
{
  ir::operation::TransposeConv::Param param;
  const auto *options = op->builtin_options_as_TransposeConvOptions();
  loadStridesAndPaddings(param, options);

  loadOperationTo<ir::operation::TransposeConv>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadPool2D(const Operator *op, ir::Graph &subg,
                                          ir::operation::Pool2D::PoolType op_type)
{
  ir::operation::Pool2D::Param param;
  param.op_type = op_type;
  const auto *options = op->builtin_options_as_Pool2DOptions();

  loadPool2DOptions(param, options);

  loadOperationTo<ir::operation::Pool2D>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadReshape(const Operator *op, ir::Graph &subg)
{
  ir::operation::Reshape::Param param{};
  const auto *options = op->builtin_options_as_ReshapeOptions();
  if (options != nullptr)
  {
    const auto *new_shape = options->new_shape();
    if (new_shape)
    {
      for (uint i = 0; i < new_shape->size(); ++i)
      {
        param.new_shape.push_back(new_shape->Get(i));
      }
    }
  }

  loadOperationTo<ir::operation::Reshape>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadSoftmax(const Operator *op, ir::Graph &subg)
{
  ir::operation::Softmax::Param param;
  const auto *options = op->builtin_options_as_SoftmaxOptions();
  // Beta
  param.beta = options->beta();

  loadOperationTo<ir::operation::Softmax>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadConcatenation(const Operator *op, ir::Graph &subg)
{
  ir::operation::Concat::Param param;
  const auto *options = op->builtin_options_as_ConcatenationOptions();
  // Axis
  param.axis = options->axis();
  // activation unused

  loadOperationTo<ir::operation::Concat>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadFC(const Operator *op, ir::Graph &subg)
{
  ir::operation::FullyConnected::Param param;
  const auto *options = op->builtin_options_as_FullyConnectedOptions();

  param.activation = convertActivation(options->fused_activation_function());
  param.weights_format = static_cast<ir::FullyConnectedWeightsFormat>(options->weights_format());

  const auto fc = loadOperationTo<ir::operation::FullyConnected>(op, subg, param);

  // TFLite supports old hybrid quantization (float input/output, uint8 kernel)
  // but it interprets weight type as init8 internally
  const auto &input_operand =
    subg.operands().at(fc->getInputs().at(ir::operation::FullyConnected::INPUT));
  auto &weights_operand =
    subg.operands().at(fc->getInputs().at(ir::operation::FullyConnected::WEIGHT));
  if (input_operand.typeInfo().type() == ir::DataType::FLOAT32 &&
      ((weights_operand.typeInfo().type() == ir::DataType::QUANT_UINT8_ASYMM) ||
       weights_operand.typeInfo().type() == ir::DataType::QUANT_INT8_ASYMM))
  {
    weights_operand.type(ir::DataType::QUANT_INT8_SYMM);
  }
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadAddV2(const Operator *op, ir::Graph &subg)
{
  ir::operation::BinaryArithmetic::Param param;
  param.arithmetic_type = ir::operation::BinaryArithmetic::ArithmeticType::ADD;

  if (op->custom_options() == nullptr)
  {
    param.activation = ir::Activation::NONE;
  }
  else
  {
    const auto attr_map = getCustomOpAttrMap(op);
    const auto fused_activation_func = static_cast<typename LoaderDomain::ActivationFunctionType>(
      attr_map["fused_activation_function"].AsInt8());
    param.activation = convertActivation(fused_activation_func);
  }

  loadOperationTo<ir::operation::BinaryArithmetic>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadDepthToSpace(const Operator *op, ir::Graph &subg)
{
  ir::operation::DepthToSpace::Param param;
  const auto *options = op->builtin_options_as_DepthToSpaceOptions();
  param.block_size = options->block_size();

  loadOperationTo<ir::operation::DepthToSpace>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadBinaryArithmetic(
  const Operator *op, ir::Graph &subg, ir::operation::BinaryArithmetic::ArithmeticType op_type)
{
  ir::operation::BinaryArithmetic::Param param;
  param.arithmetic_type = op_type;
  switch (op_type)
  {
    case ir::operation::BinaryArithmetic::ArithmeticType::ADD:
    {
      const auto *add_options = op->builtin_options_as_AddOptions();
      param.activation = convertActivation(add_options->fused_activation_function());
      break;
    }
    case ir::operation::BinaryArithmetic::ArithmeticType::SUB:
    {
      const auto *sub_options = op->builtin_options_as_SubOptions();
      param.activation = convertActivation(sub_options->fused_activation_function());
      break;
    }
    case ir::operation::BinaryArithmetic::ArithmeticType::MUL:
    {
      const auto *mul_options = op->builtin_options_as_MulOptions();
      param.activation = convertActivation(mul_options->fused_activation_function());
      break;
    }
    case ir::operation::BinaryArithmetic::ArithmeticType::DIV:
    {
      const auto *div_options = op->builtin_options_as_DivOptions();
      param.activation = convertActivation(div_options->fused_activation_function());
      break;
    }
    default:
      assert(false &&
             "The function 'loadBinaryArithmetic' supports only BinaryArithmetic operations");
      break;
  }

  loadOperationTo<ir::operation::BinaryArithmetic>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadPack(const Operator *op, ir::Graph &subg)
{
  ir::operation::Pack::Param param;
  const auto *options = op->builtin_options_as_PackOptions();
  param.num = options->values_count();
  param.axis = options->axis();

  loadOperationTo<ir::operation::Pack>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadElementwiseActivation(
  const Operator *op, ir::Graph &subg, ir::operation::ElementwiseActivation::Type op_type,
  float alpha, float beta)
{
  ir::operation::ElementwiseActivation::Param param;
  param.op_type = op_type;
  param.alpha = alpha;
  param.beta = beta;

  loadOperationTo<ir::operation::ElementwiseActivation>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadResizeBilinear(const Operator *op, ir::Graph &subg)
{
  ir::operation::ResizeBilinear::Param param;
  param.align_corners = op->builtin_options_as_ResizeBilinearOptions()->align_corners();
  param.half_pixel_centers = op->builtin_options_as_ResizeBilinearOptions()->half_pixel_centers();

  loadOperationTo<ir::operation::ResizeBilinear>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadResizeNearestNeighbor(const Operator *op, ir::Graph &subg)
{
  ir::operation::ResizeNearestNeighbor::Param param;
  param.align_corners = op->builtin_options_as_ResizeNearestNeighborOptions()->align_corners();

  loadOperationTo<ir::operation::ResizeNearestNeighbor>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadReduce(const Operator *op, ir::Graph &subg,
                                          ir::operation::Reduce::ReduceType reduce_type)
{
  ir::operation::Reduce::Param param;
  param.reduce_type = reduce_type;
  param.keep_dims = op->builtin_options_as_ReducerOptions()->keep_dims();

  loadOperationTo<ir::operation::Reduce>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadReduceAll(const Operator *op, ir::Graph &subg)
{
  ir::operation::Reduce::Param param;
  param.reduce_type = ir::operation::Reduce::ReduceType::ALL;
  if (op->custom_options() == nullptr)
  {
    param.keep_dims = false;
  }
  else
  {
    const auto attr_map = getCustomOpAttrMap(op);
    param.keep_dims = attr_map["keep_dims"].AsBool();
  }

  loadOperationTo<ir::operation::Reduce>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadElementwiseBinary(
  const Operator *op, ir::Graph &subg,
  ir::operation::ElementwiseBinary::ElementwiseBinaryType op_type)
{
  ir::operation::ElementwiseBinary::Param param;
  param.op_type = op_type;

  loadOperationTo<ir::operation::ElementwiseBinary>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadElementwiseUnary(const Operator *op, ir::Graph &subg,
                                                    ir::operation::ElementwiseUnary::Type op_type)
{
  ir::operation::ElementwiseUnary::Param param;
  param.op_type = op_type;

  const auto eu = loadOperationTo<ir::operation::ElementwiseUnary>(op, subg, param);
  if (op_type == ir::operation::ElementwiseUnary::Type::CAST)
  {
    auto qasymm8ToUint8 = [](ir::Operand &operand) {
      if (operand.typeInfo().type() == ir::DataType::QUANT_UINT8_ASYMM)
      {
        operand.type(ir::DataType::UINT8);
      }
    };
    qasymm8ToUint8(
      subg.operands().at(eu->getInputs().at(ir::operation::ElementwiseUnary::Input::INPUT)));
    qasymm8ToUint8(subg.operands().at(eu->getOutputs().at(0)));
  }
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadGather(const Operator *op, ir::Graph &subg)
{
  ir::operation::Gather::Param param;
  param.axis = op->builtin_options_as_GatherOptions()->axis();

  loadOperationTo<ir::operation::Gather>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadDetectionPostProcess(const Operator *op, ir::Graph &subg)
{
  const auto &m = getCustomOpAttrMap(op);

  ir::operation::DetectionPostProcess::Param param;

  param.max_detections = m["max_detections"].AsInt32();

  // TODO fixme
  param.max_classes_per_detection = m["max_classes_per_detection"].AsInt32();
  if (m["detections_per_class"].IsNull())
    param.max_boxes_per_class = 100;
  else
    param.max_boxes_per_class = m["detections_per_class"].AsInt32();

  if (m["use_regular_nms"].IsNull())
    param.do_fast_eval = true;
  else
    param.do_fast_eval = !m["use_regular_nms"].AsBool();

  param.score_threshold = m["nms_score_threshold"].AsFloat();
  param.iou_threshold = m["nms_iou_threshold"].AsFloat();

  // TODO add num classes support
  param.num_classes = m["num_classes"].AsInt32();

  param.scale.y_scale = m["y_scale"].AsFloat();
  param.scale.x_scale = m["x_scale"].AsFloat();
  param.scale.h_scale = m["h_scale"].AsFloat();
  param.scale.w_scale = m["w_scale"].AsFloat();

  // TODO depends on input model framework
  param.center_size_boxes = true;

  loadOperationTo<ir::operation::DetectionPostProcess>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadBatchMatMul(const Operator *op, ir::Graph &subg)
{
  ir::operation::BatchMatMul::Param param;

  const auto builtin_op = getBuiltinOperator(op);

  switch (builtin_op)
  {
    case BuiltinOperator::BuiltinOperator_BATCH_MATMUL:
      // Handled on each loader: different option name
      //  Circle: adjoint_lhs, adjoint_rhs
      //  TFLite: adj_x, adj_y
      throw std::runtime_error(
        std::string("Cannot handle here: ").append(EnumNameBuiltinOperator(builtin_op)) + " as " +
        EnumNameBuiltinOperator(BuiltinOperator::BuiltinOperator_BATCH_MATMUL));
    case BuiltinOperator::BuiltinOperator_CUSTOM:
      if (op->custom_options() == nullptr)
      {
        param.adj_x = false;
        param.adj_y = false;
      }
      else
      {
        const auto attr_map = getCustomOpAttrMap(op);
        param.adj_x = attr_map["adj_x"].AsBool();
        param.adj_y = attr_map["adj_y"].AsBool();
      }
      break;
    default:
      throw std::runtime_error(
        std::string("Wrong loaded operation: ").append(EnumNameBuiltinOperator(builtin_op)) +
        " as " + EnumNameBuiltinOperator(BuiltinOperator::BuiltinOperator_BATCH_MATMUL));
  }

  loadOperationTo<ir::operation::BatchMatMul>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadSpaceToDepth(const Operator *op, ir::Graph &subg)
{
  ir::operation::SpaceToDepth::Param param;
  const auto *options = op->builtin_options_as_SpaceToDepthOptions();
  param.block_size = options->block_size();

  loadOperationTo<ir::operation::SpaceToDepth>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadCustom(const Operator *op, ir::Graph &subg)
{
  ir::OperandIndexSequence inputs;
  ir::OperandIndexSequence outputs;

  assert(op->custom_options_format() == CustomOptionsFormat::CustomOptionsFormat_FLEXBUFFERS &&
         "Unsupported custom operation options format");

  auto *op_code = _domain_model->operator_codes()->Get(op->opcode_index());
  auto custom_op_name = op_code->custom_code()->str();

  enum class BuiltinOP
  {
    AddV2,
    ReduceAll,
    MatrixBandPart,
    BatchMatMul,
    Einsum,
    BroadcastTo,
    FusedBatchNorm,
    StatelessRandomUniform,
    Erf,
    DetectionPostProcess
  };

  // Mapping from custom op name string to BuiltinOP enum
  std::map<std::string, BuiltinOP> builtin_map = {
    {"AddV2", BuiltinOP::AddV2},
    {"All", BuiltinOP::ReduceAll},
    {"MatrixBandPart", BuiltinOP::MatrixBandPart},
    {"BatchMatMulV2", BuiltinOP::BatchMatMul},
    {"Einsum", BuiltinOP::Einsum},
    {"FusedBatchNormV3", BuiltinOP::FusedBatchNorm},
    {"BroadcastTo", BuiltinOP::BroadcastTo},
    {"StatelessRandomUniform", BuiltinOP::StatelessRandomUniform},
    {"Erf", BuiltinOP::Erf},
    {"TFLite_Detection_PostProcess", BuiltinOP::DetectionPostProcess},
  };

  try
  {
    // Throw out_of_range if it is unknown custom op
    auto custom_op_id = builtin_map.at(custom_op_name);
    switch (custom_op_id)
    {
      case BuiltinOP::AddV2:
        loadAddV2(op, subg);
        break;
      case BuiltinOP::ReduceAll:
        loadReduceAll(op, subg);
        break;
      case BuiltinOP::MatrixBandPart:
        loadOperationTo<ir::operation::MatrixBandPart>(op, subg);
        break;
      case BuiltinOP::BatchMatMul:
        loadBatchMatMul(op, subg);
        break;
      case BuiltinOP::Einsum:
        loadEinsum(op, subg);
        break;
      case BuiltinOP::BroadcastTo:
        loadOperationTo<ir::operation::BroadcastTo>(op, subg);
        break;
      case BuiltinOP::FusedBatchNorm:
        loadFusedBatchNorm(op, subg);
        break;
      case BuiltinOP::StatelessRandomUniform:
        loadOperationTo<ir::operation::StatelessRandomUniform>(op, subg);
        break;
      case BuiltinOP::Erf:
        loadElementwiseUnary(op, subg, ir::operation::ElementwiseUnary::Type::ERF);
        break;
      case BuiltinOP::DetectionPostProcess:
        loadDetectionPostProcess(op, subg);
        break;
      default:
        throw std::runtime_error{
          "Loader: Custom OP map is defined but operation loader function is not defined"};
    }

    return;
  }
  catch (...)
  {
    loadOperationIO(op, inputs, outputs);

    auto constraint = ir::OperandConstraint::createExact(inputs.size());

    size_t custom_op_data_size = op->custom_options()->size();
    auto custom_op_data = new char[custom_op_data_size];
    std::copy(op->custom_options()->begin(), op->custom_options()->end(), custom_op_data);

    ir::operation::Custom::Userdata userdata{};
    userdata.data = custom_op_data;
    userdata.size = custom_op_data_size;

    auto new_op = std::make_unique<ir::operation::Custom>(constraint, inputs, outputs,
                                                          custom_op_name, userdata);

    subg.addOperation(std::move(new_op));
  }
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadSqueeze(const Operator *op, ir::Graph &subg)
{
  ir::operation::Squeeze::Param param;
  const auto *options = op->builtin_options_as_SqueezeOptions();
  const auto *dims = options->squeeze_dims();
  if (dims)
  {
    if (dims->size() > sizeof(param.dims) / sizeof(param.dims[0]))
      throw std::runtime_error("Squeeze: 'param.ndims' is out of range.");
    param.ndim = dims->size();
    for (int i = 0; i < param.ndim; ++i)
      param.dims[i] = dims->Get(i);
  }

  loadOperationTo<ir::operation::Squeeze>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadSplit(const Operator *op, ir::Graph &subg)
{
  ir::operation::Split::Param param;
  const auto *options = op->builtin_options_as_SplitOptions();
  param.num_splits = options->num_splits();

  loadOperationTo<ir::operation::Split>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadSplitV(const Operator *op, ir::Graph &subg)
{
  ir::operation::SplitV::Param param;
  const auto *options = op->builtin_options_as_SplitVOptions();
  param.num_splits = options->num_splits();

  loadOperationTo<ir::operation::SplitV>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadStridedSlice(const Operator *op, ir::Graph &subg)
{
  ir::operation::StridedSlice::Param param;
  const auto *options = op->builtin_options_as_StridedSliceOptions();
  param.begin_mask = options->begin_mask();
  param.end_mask = options->end_mask();
  param.shrink_axis_mask = options->shrink_axis_mask();

  loadOperationTo<ir::operation::StridedSlice>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadUnpack(const Operator *op, ir::Graph &subg)
{
  ir::operation::Unpack::Param param;
  const auto *options = op->builtin_options_as_UnpackOptions();
  param.num = options->num();
  param.axis = options->axis();

  loadOperationTo<ir::operation::Unpack>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadComparison(const Operator *op, ir::Graph &subg)
{
  ir::operation::Comparison::Param param;
  const auto builtin_op = getBuiltinOperator(op);

  switch (builtin_op)
  {
    case BuiltinOperator::BuiltinOperator_EQUAL:
      param.comparison_type = ir::operation::Comparison::ComparisonType::Equal;
      break;
    case BuiltinOperator::BuiltinOperator_NOT_EQUAL:
      param.comparison_type = ir::operation::Comparison::ComparisonType::NotEqual;
      break;
    case BuiltinOperator::BuiltinOperator_GREATER_EQUAL:
      param.comparison_type = ir::operation::Comparison::ComparisonType::GreaterEqual;
      break;
    case BuiltinOperator::BuiltinOperator_GREATER:
      param.comparison_type = ir::operation::Comparison::ComparisonType::Greater;
      break;
    case BuiltinOperator::BuiltinOperator_LESS_EQUAL:
      param.comparison_type = ir::operation::Comparison::ComparisonType::LessEqual;
      break;
    case BuiltinOperator::BuiltinOperator_LESS:
      param.comparison_type = ir::operation::Comparison::ComparisonType::Less;
      break;
    default:
      throw std::runtime_error(
        std::string("Unsupported operation: ").append(EnumNameBuiltinOperator(builtin_op)));
  }

  loadOperationTo<ir::operation::Comparison>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadEinsum(const Operator *op, ir::Graph &subg)
{
  ir::operation::Einsum::Param param;
  if (op->custom_options() == nullptr)
  {
    throw std::runtime_error{"Einsum: empty equation"};
  }
  else
  {
    const auto attr_map = getCustomOpAttrMap(op);
    param.equation = attr_map["equation"].ToString();
  }

  const auto es = loadOperationTo<ir::operation::Einsum>(op, subg, param);
  if (es->getInputs().size() != 2)
  {
    throw std::runtime_error{"Einsum: NYI input - only support two inputs"};
  }
}
template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadFusedBatchNorm(const Operator *op, ir::Graph &subg)
{
  ir::operation::FusedBatchNorm::Param param;
  if (op->custom_options() == nullptr)
  {
    throw std::runtime_error{"FusedBatchNorm: empty option"};
  }
  else
  {
    const auto attr_map = getCustomOpAttrMap(op);
    param.is_training = attr_map["is_training"].AsBool();
    param.epsilon = attr_map["epsilon"].AsFloat();
    param.data_format = attr_map["data_format"].ToString();
  }

  const auto fbn = loadOperationTo<ir::operation::FusedBatchNorm>(op, subg, param);

  if (fbn->getInputs().size() != 5)
  {
    throw std::runtime_error{"FusedBatchNorm: NYI input - only support five inputs"};
  }
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadOneHot(const Operator *op, ir::Graph &subg)
{
  if (op->inputs()->size() != 4 || op->outputs()->size() != 1)
    throw std::runtime_error("OneHot Op has wrong number of input or output tensors.");

  // Set parameter
  ir::operation::OneHot::Param param;
  param.axis = op->builtin_options_as_OneHotOptions()->axis();

  loadOperationTo<ir::operation::OneHot>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadIf(const Operator *op, ir::Graph &subg)
{
  const auto *options = op->builtin_options_as_IfOptions();
  const int32_t then_index = options->then_subgraph_index();
  const int32_t else_index = options->else_subgraph_index();

  verifySubgraphIndex(then_index);
  verifySubgraphIndex(else_index);

  ir::operation::If::Param param;
  param.then_subg_index = ir::SubgraphIndex{static_cast<uint16_t>(then_index)};
  param.else_subg_index = ir::SubgraphIndex{static_cast<uint16_t>(else_index)};

  loadOperationTo<ir::operation::If>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadWhile(const Operator *op, ir::Graph &subg)
{
  const auto *options = op->builtin_options_as_WhileOptions();
  const int32_t cond_index = options->cond_subgraph_index();
  const int32_t body_index = options->body_subgraph_index();

  verifySubgraphIndex(cond_index);
  verifySubgraphIndex(body_index);

  ir::operation::While::Param param;
  param.cond_subg_index = ir::SubgraphIndex{static_cast<uint16_t>(cond_index)};
  param.body_subg_index = ir::SubgraphIndex{static_cast<uint16_t>(body_index)};

  loadOperationTo<ir::operation::While>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadArgMinMax(const Operator *op, ir::Graph &subg, bool is_argmax)
{
  ir::operation::ArgMinMax::Param param;
  const auto output_type = is_argmax ? op->builtin_options_as_ArgMaxOptions()->output_type()
                                     : op->builtin_options_as_ArgMinOptions()->output_type();
  param.output_type = tensorTypeToDataType(output_type);
  param.is_arg_max = is_argmax;

  loadOperationTo<ir::operation::ArgMinMax>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadLogSoftmax(const Operator *op, ir::Graph &subg)
{
  ir::operation::LogSoftmax::Param param;
  // In tflite, beta is fixed to 1.0 and axis is fixed to -1.
  param.beta = 1.0f;
  param.axis = -1;

  loadOperationTo<ir::operation::LogSoftmax>(op, subg, param);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadLeakyRelu(const Operator *op, ir::Graph &subg)
{
  float alpha = op->builtin_options_as_LeakyReluOptions()->alpha();
  loadElementwiseActivation(op, subg, ir::operation::ElementwiseActivation::Type::LEAKY_RELU, alpha,
                            1.f);
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadUnidirectionalSequenceLSTM(const Operator *op, ir::Graph &subg)
{
  ir::operation::LSTM::Param param;
  const auto *options = op->builtin_options_as_UnidirectionalSequenceLSTMOptions();
  param.activation = convertActivation(options->fused_activation_function());
  param.cell_threshold = options->cell_clip();
  param.projection_threshold = options->proj_clip();
  param.time_major = options->time_major();
  // The asymmetric_quantize_inputs option is unused yet

  ir::OperandIndexSequence inputs;
  for (const std::int32_t idx : *op->inputs())
  {
    inputs.append(tensorIdxToOperandIdx(idx));
  }

  ir::OperandIndexSequence outputs;
  // loader doesn't support optional output tensor yet
  if (op->outputs()->size() != 1)
  {
    auto builtin_code = getBuiltinOperator(op);
    throw std::runtime_error(std::string("loader doesn't support optional output tensor yet for ")
                               .append(EnumNameBuiltinOperator(builtin_code)));
  }
  for (size_t i = 0; i < ir::operation::LSTM::Output::OUTPUT; ++i)
  {
    // Add optional outputs
    outputs.append(ir::OperandIndex());
  }
  outputs.append(tensorIdxToOperandIdx(op->outputs()->Get(0)));

  std::unique_ptr<ir::operation::LSTM> new_op(new ir::operation::LSTM(inputs, outputs, param));
  subg.addOperation(std::move(new_op));
}

template <typename LoaderDomain>
void BaseLoader<LoaderDomain>::loadOperation(const Operator *op, ir::Graph &subg)
{
  auto const builtin_op = getBuiltinOperator(op);

  switch (builtin_op)
  {
    case BuiltinOperator::BuiltinOperator_ADD_N:
      loadOperationTo<ir::operation::AddN>(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_CONV_2D:
      loadConv2D(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_AVERAGE_POOL_2D:
      loadPool2D(op, subg, ir::operation::Pool2D::PoolType::AVG);
      return;
    case BuiltinOperator::BuiltinOperator_DEPTHWISE_CONV_2D:
      loadDepthwiseConv2D(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_TRANSPOSE_CONV:
      loadTransposeConv(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_RESHAPE:
      loadReshape(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SOFTMAX:
      loadSoftmax(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_MAX_POOL_2D:
      loadPool2D(op, subg, ir::operation::Pool2D::PoolType::MAX);
      return;
    case BuiltinOperator::BuiltinOperator_CONCATENATION:
      loadConcatenation(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_FLOOR:
      loadElementwiseUnary(op, subg, ir::operation::ElementwiseUnary::Type::FLOOR);
      return;
    case BuiltinOperator::BuiltinOperator_FULLY_CONNECTED:
      loadFC(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_ADD:
      loadBinaryArithmetic(op, subg, ir::operation::BinaryArithmetic::ArithmeticType::ADD);
      return;
    case BuiltinOperator::BuiltinOperator_SUB:
      loadBinaryArithmetic(op, subg, ir::operation::BinaryArithmetic::ArithmeticType::SUB);
      return;
    case BuiltinOperator::BuiltinOperator_MUL:
      loadBinaryArithmetic(op, subg, ir::operation::BinaryArithmetic::ArithmeticType::MUL);
      return;
    case BuiltinOperator::BuiltinOperator_DIV:
      loadBinaryArithmetic(op, subg, ir::operation::BinaryArithmetic::ArithmeticType::DIV);
      return;
    case BuiltinOperator::BuiltinOperator_PACK:
      loadPack(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_ELU:
      loadElementwiseActivation(op, subg, ir::operation::ElementwiseActivation::Type::ELU);
      return;
    case BuiltinOperator::BuiltinOperator_RELU:
      loadElementwiseActivation(op, subg, ir::operation::ElementwiseActivation::Type::RELU,
                                ir::operation::ElementwiseActivation::infinity, 0.f);
      return;
    case BuiltinOperator::BuiltinOperator_RELU_N1_TO_1:
      loadElementwiseActivation(op, subg, ir::operation::ElementwiseActivation::Type::RELU, 1.f,
                                -1.f);
      return;
    case BuiltinOperator::BuiltinOperator_RELU6:
      loadElementwiseActivation(op, subg, ir::operation::ElementwiseActivation::Type::RELU, 6.f,
                                0.f);
      return;
    case BuiltinOperator::BuiltinOperator_RESIZE_BILINEAR:
      loadResizeBilinear(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR:
      loadResizeNearestNeighbor(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_RSQRT:
      loadElementwiseUnary(op, subg, ir::operation::ElementwiseUnary::Type::RSQRT);
      return;
    case BuiltinOperator::BuiltinOperator_SELECT:
    case BuiltinOperator::BuiltinOperator_SELECT_V2:
      loadOperationTo<ir::operation::Select>(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SQRT:
      loadElementwiseUnary(op, subg, ir::operation::ElementwiseUnary::Type::SQRT);
      return;
    case BuiltinOperator::BuiltinOperator_SQUARE:
      loadElementwiseUnary(op, subg, ir::operation::ElementwiseUnary::Type::SQUARE);
      return;
    case BuiltinOperator::BuiltinOperator_SQUARED_DIFFERENCE:
      loadOperationTo<ir::operation::SquaredDifference>(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_TANH:
      loadElementwiseActivation(op, subg, ir::operation::ElementwiseActivation::Type::TANH, 1.f,
                                1.f);
      return;
    case BuiltinOperator::BuiltinOperator_TRANSPOSE:
      loadOperationTo<ir::operation::Transpose>(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_MEAN:
      loadReduce(op, subg, ir::operation::Reduce::ReduceType::MEAN);
      return;
    case BuiltinOperator::BuiltinOperator_REDUCE_ANY:
      loadReduce(op, subg, ir::operation::Reduce::ReduceType::ANY);
      return;
    case BuiltinOperator::BuiltinOperator_REDUCE_MAX:
      loadReduce(op, subg, ir::operation::Reduce::ReduceType::MAX);
      return;
    case BuiltinOperator::BuiltinOperator_REVERSE_V2:
      loadOperationTo<ir::operation::Reverse>(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_PAD:
    case BuiltinOperator::BuiltinOperator_PADV2:
      loadOperationTo<ir::operation::Pad>(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_LOGISTIC:
      loadElementwiseActivation(op, subg, ir::operation::ElementwiseActivation::Type::LOGISTIC);
      return;
    case BuiltinOperator::BuiltinOperator_EXP:
      loadElementwiseUnary(op, subg, ir::operation::ElementwiseUnary::Type::EXP);
      return;
    case BuiltinOperator::BuiltinOperator_EXPAND_DIMS:
      loadOperationTo<ir::operation::ExpandDims>(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_GATHER:
      loadGather(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SPACE_TO_BATCH_ND:
      loadOperationTo<ir::operation::SpaceToBatchND>(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_BATCH_TO_SPACE_ND:
      loadOperationTo<ir::operation::BatchToSpaceND>(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SUM:
      loadReduce(op, subg, ir::operation::Reduce::ReduceType::SUM);
      return;
    case BuiltinOperator::BuiltinOperator_CUSTOM:
      loadCustom(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SQUEEZE:
      loadSqueeze(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_PRELU:
      loadOperationTo<ir::operation::PReLU>(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SPLIT:
      loadSplit(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SPLIT_V:
      loadSplitV(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_SLICE:
      loadOperationTo<ir::operation::Slice>(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_STRIDED_SLICE:
      loadStridedSlice(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_UNPACK:
      loadUnpack(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_FLOOR_DIV:
      loadElementwiseBinary(op, subg,
                            ir::operation::ElementwiseBinary::ElementwiseBinaryType::FLOOR_DIV);
      return;
    case BuiltinOperator::BuiltinOperator_FLOOR_MOD:
      loadElementwiseBinary(op, subg,
                            ir::operation::ElementwiseBinary::ElementwiseBinaryType::FLOOR_MOD);
      return;
    case BuiltinOperator::BuiltinOperator_MINIMUM:
      loadElementwiseBinary(op, subg, ir::operation::ElementwiseBinary::ElementwiseBinaryType::MIN);
      return;
    case BuiltinOperator::BuiltinOperator_MAXIMUM:
      loadElementwiseBinary(op, subg, ir::operation::ElementwiseBinary::ElementwiseBinaryType::MAX);
      return;
    case BuiltinOperator::BuiltinOperator_CAST:
      loadElementwiseUnary(op, subg, ir::operation::ElementwiseUnary::Type::CAST);
      return;
    case BuiltinOperator::BuiltinOperator_EQUAL:
    case BuiltinOperator::BuiltinOperator_NOT_EQUAL:
    case BuiltinOperator::BuiltinOperator_GREATER_EQUAL:
    case BuiltinOperator::BuiltinOperator_GREATER:
    case BuiltinOperator::BuiltinOperator_LESS_EQUAL:
    case BuiltinOperator::BuiltinOperator_LESS:
      loadComparison(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_ONE_HOT:
      loadOneHot(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_ABS:
      loadElementwiseUnary(op, subg, ir::operation::ElementwiseUnary::Type::ABS);
      return;
    case BuiltinOperator::BuiltinOperator_COS:
      loadElementwiseUnary(op, subg, ir::operation::ElementwiseUnary::Type::COS);
      return;
    case BuiltinOperator::BuiltinOperator_SIN:
      loadElementwiseUnary(op, subg, ir::operation::ElementwiseUnary::Type::SIN);
      return;
    case BuiltinOperator::BuiltinOperator_SHAPE:
      loadOperationTo<ir::operation::Shape>(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_REDUCE_PROD:
      loadReduce(op, subg, ir::operation::Reduce::ReduceType::PROD);
      return;
    case BuiltinOperator::BuiltinOperator_IF:
      loadIf(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_WHILE:
      loadWhile(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_NEG:
      loadElementwiseUnary(op, subg, ir::operation::ElementwiseUnary::Type::NEG);
      return;
    case BuiltinOperator::BuiltinOperator_ARG_MAX:
      loadArgMinMax(op, subg, true);
      return;
    case BuiltinOperator::BuiltinOperator_ARG_MIN:
      loadArgMinMax(op, subg, false);
      return;
    case BuiltinOperator::BuiltinOperator_LOG:
      loadElementwiseUnary(op, subg, ir::operation::ElementwiseUnary::Type::LOG);
      return;
    case BuiltinOperator::BuiltinOperator_ROUND:
      loadElementwiseUnary(op, subg, ir::operation::ElementwiseUnary::Type::ROUND);
      return;
    case BuiltinOperator::BuiltinOperator_POW:
      loadOperationTo<ir::operation::Pow>(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_LOGICAL_NOT:
      loadElementwiseUnary(op, subg, ir::operation::ElementwiseUnary::Type::LOGICAL_NOT);
      return;
    case BuiltinOperator::BuiltinOperator_LOGICAL_AND:
      loadElementwiseBinary(op, subg,
                            ir::operation::ElementwiseBinary::ElementwiseBinaryType::LOGICAL_AND);
      return;
    case BuiltinOperator::BuiltinOperator_LOGICAL_OR:
      loadElementwiseBinary(op, subg,
                            ir::operation::ElementwiseBinary::ElementwiseBinaryType::LOGICAL_OR);
      return;
    case BuiltinOperator::BuiltinOperator_FILL:
      loadOperationTo<ir::operation::Fill>(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_ZEROS_LIKE:
      loadElementwiseUnary(op, subg, ir::operation::ElementwiseUnary::Type::ZEROS_LIKE);
      return;
    case BuiltinOperator::BuiltinOperator_TILE:
      loadOperationTo<ir::operation::Tile>(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_RANGE:
      loadOperationTo<ir::operation::Range>(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_BATCH_MATMUL:
      loadBatchMatMul(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_LOG_SOFTMAX:
      loadLogSoftmax(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_QUANTIZE:
      loadElementwiseUnary(op, subg, ir::operation::ElementwiseUnary::Type::QUANTIZE);
      return;
    case BuiltinOperator::BuiltinOperator_DEQUANTIZE:
      loadElementwiseUnary(op, subg, ir::operation::ElementwiseUnary::Type::DEQUANTIZE);
      return;
    case BuiltinOperator::BuiltinOperator_SPACE_TO_DEPTH:
      loadSpaceToDepth(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_L2_NORMALIZATION:
      loadOperationTo<ir::operation::L2Normalization>(op, subg);
      break;
    case BuiltinOperator::BuiltinOperator_LEAKY_RELU:
      loadLeakyRelu(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_RANK:
      loadOperationTo<ir::operation::Rank>(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
      loadUnidirectionalSequenceLSTM(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_DEPTH_TO_SPACE:
      loadDepthToSpace(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_EMBEDDING_LOOKUP:
      loadOperationTo<ir::operation::EmbeddingLookup>(op, subg);
      return;
    case BuiltinOperator::BuiltinOperator_HASHTABLE_LOOKUP:
      loadOperationTo<ir::operation::HashtableLookup>(op, subg);
      return;
    default:
      throw std::runtime_error(
        std::string("Unsupported operation: ").append(EnumNameBuiltinOperator(builtin_op)));
  }
}

template <typename LoaderDomain> void BaseLoader<LoaderDomain>::loadModel()
{
  LoaderDomain::VerifyModelBuffer(*_verifier.get());
  _domain_model = LoaderDomain::GetModel(_base);

  auto model = std::make_unique<ir::Model>();
  // Version unused
  // const auto version = _model->version();
  // Description unused

  // Load Metadata
  auto const metadata_list = _domain_model->metadata();
  if (metadata_list != nullptr)
  {
    for (uint32_t i = 0; i < metadata_list->size(); ++i)
    {
      const auto metadata = metadata_list->Get(i);
      if (metadata->name() == nullptr)
        continue; // metadata should have name

      std::unique_ptr<const ir::Data> data = loadMetadata(metadata->buffer());
      model->add_metadata(metadata->name()->str(), std::move(data));
    }
  }

  // const auto *description = _model->description();
  // Load subgraphs and map operations on subgraph
  const auto subgraphs = _domain_model->subgraphs();
  if (subgraphs->size() - 1 > ir::SubgraphIndex::max())
    throw std::runtime_error{"The number of subgraphs cannot exceed " +
                             std::to_string(ir::SubgraphIndex::max() + 1)};
  for (uint16_t subgraph_index = 0; subgraph_index < subgraphs->size(); ++subgraph_index)
  {
    auto subg = loadSubgraph((*_domain_model->subgraphs())[subgraph_index]);
    // NOTE: Used () instead of {}, which does not check narrowing.
    // It is okay since overflow is checked the above if-statement.
    model->push(ir::SubgraphIndex(subgraph_index), std::move(subg));
  }
  _model = std::move(model);
}

} // namespace base_loader
} // namespace onert

#endif //__BASE_LOADER_BASE_LOADER_H__
