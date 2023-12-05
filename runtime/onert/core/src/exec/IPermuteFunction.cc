/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "IPermuteFunction.h"

#include <cker/operation/Quantize.h>
#include <cker/operation/Dequantize.h>
#include "backend/IPortableTensor.h"
#include "exec/IFunction.h"
#include "ir/Index.h"
#include "ir/Shape.h"
#include <memory>
#include <misc/polymorphic_downcast.h>
#include <typeinfo>
#include "util/Utils.h"
#include <vector>
#include <unordered_map>

namespace
{
using namespace onert;

inline nnfw::cker::Shape getShape(const backend::ITensor *tensor)
{
  const ir::Shape shape = tensor->getShape();

  assert(tensor->layout() == ir::Layout::NHWC || tensor->layout() == ir::Layout::UNKNOWN);

  auto rank = shape.rank();
  nnfw::cker::Shape ret(rank);
  auto data = ret.DimsData();
  for (int i = 0; i < rank; ++i)
  {
    data[i] = shape.dim(i);
  }
  return ret;
}

// Quantize per element
template <typename InputT, typename OutputT>
void elementwiseQuantize(const backend::ITensor *src_tensor, backend::ITensor *dst_tensor)
{
  const auto scale = dst_tensor->data_scale();
  const auto zero_point = dst_tensor->data_zero_point();

  int min_val = std::numeric_limits<OutputT>::min();
  int max_val = std::numeric_limits<OutputT>::max();

  auto loop_shape = src_tensor->getShape();
  const auto src_layout = src_tensor->layout();
  const auto dst_layout = dst_tensor->layout();
  const bool is_permutation = src_layout != dst_layout && loop_shape.rank() == 4;
  ShapeLoop(loop_shape, [&](const onert::ir::Coordinates &coords) {
    const InputT *input_data =
      reinterpret_cast<const InputT *>(src_tensor->buffer() + src_tensor->calcOffset(coords));
    int32_t unclamped = static_cast<int32_t>(round(*input_data / scale)) + zero_point;
    int32_t clamped = std::min(std::max(unclamped, min_val), max_val);

    ir::Coordinates dst_coords =
      is_permutation ? ir::convertCoordinates(coords, src_layout, dst_layout) : coords;
    OutputT *output_data =
      reinterpret_cast<OutputT *>(dst_tensor->buffer() + dst_tensor->calcOffset(dst_coords));
    *output_data = clamped;
  });
}

// TODO Optimize the case where tensors has the same layout
template <typename InputT, typename OutputT>
void quantize(const backend::ITensor *src_tensor, backend::ITensor *dst_tensor)
{
  if (!src_tensor->has_padding() && !dst_tensor->has_padding() &&
      src_tensor->layout() == dst_tensor->layout() && !src_tensor->is_dynamic())
  {
    assert(!dst_tensor->is_dynamic());

    // Call optimized neon kernel
    nnfw::cker::Quantize(getShape(src_tensor),
                         reinterpret_cast<const InputT *>(src_tensor->buffer()),
                         getShape(dst_tensor), reinterpret_cast<OutputT *>(dst_tensor->buffer()),
                         dst_tensor->data_scale(), dst_tensor->data_zero_point());
  }
  else
  {
    elementwiseQuantize<InputT, OutputT>(src_tensor, dst_tensor);
  }
}

// Dequantize per element
template <typename InputT, typename OutputT>
void elementwiseDequantize(const backend::ITensor *src_tensor, backend::ITensor *dst_tensor)
{
  const auto scale = src_tensor->data_scale();
  const auto zero_point = src_tensor->data_zero_point();

  auto loop_shape = src_tensor->getShape();
  const auto src_layout = src_tensor->layout();
  const auto dst_layout = dst_tensor->layout();
  const bool is_permutation = src_layout != dst_layout && loop_shape.rank() == 4;
  ShapeLoop(loop_shape, [&](const onert::ir::Coordinates &coords) {
    const InputT *input_data =
      reinterpret_cast<const InputT *>(src_tensor->buffer() + src_tensor->calcOffset(coords));
    const OutputT result = static_cast<OutputT>(scale * (*input_data - zero_point));

    ir::Coordinates dst_coords =
      is_permutation ? ir::convertCoordinates(coords, src_layout, dst_layout) : coords;
    OutputT *output_data =
      reinterpret_cast<OutputT *>(dst_tensor->buffer() + dst_tensor->calcOffset(dst_coords));
    *output_data = result;
  });
}

// TODO Optimize the case where tensors has the same layout
template <typename InputT, typename OutputT>
void dequantize(const backend::ITensor *src_tensor, backend::ITensor *dst_tensor)
{
  if (!src_tensor->has_padding() && !dst_tensor->has_padding() &&
      src_tensor->layout() == dst_tensor->layout() && !src_tensor->is_dynamic())
  {
    assert(!dst_tensor->is_dynamic());

    // Call optimized neon kernel
    nnfw::cker::Dequantize(getShape(src_tensor),
                           reinterpret_cast<const InputT *>(src_tensor->buffer()),
                           getShape(dst_tensor), reinterpret_cast<OutputT *>(dst_tensor->buffer()),
                           src_tensor->data_scale(), src_tensor->data_zero_point());
  }
  else
  {
    elementwiseDequantize<InputT, OutputT>(src_tensor, dst_tensor);
  }
}

template <typename SRC_T, typename DST_T,
          std::enable_if_t<std::is_base_of<backend::ITensor, SRC_T>::value &&
                             std::is_base_of<backend::ITensor, DST_T>::value,
                           bool> = true>
void typeAwareQuantize(const SRC_T *src_tensor, DST_T *dst_tensor)
{
  // TODO Support other types
  if (src_tensor->data_type() == ir::DataType::FLOAT32)
  {
    switch (dst_tensor->data_type())
    {
      case ir::DataType::QUANT_UINT8_ASYMM:
      {
        quantize<float, uint8_t>(src_tensor, dst_tensor);
        break;
      }
      case ir::DataType::QUANT_INT8_SYMM:
      {
        quantize<float, int8_t>(src_tensor, dst_tensor);
        break;
      }
      case ir::DataType::QUANT_INT16_SYMM:
      {
        quantize<float, int16_t>(src_tensor, dst_tensor);
        break;
      }
      default:
      {
        throw std::runtime_error("IPermuteFunction: Unsupported quantization type");
        break;
      }
    }
  }
  else if (dst_tensor->data_type() == ir::DataType::FLOAT32)
  {
    switch (src_tensor->data_type())
    {
      case ir::DataType::QUANT_UINT8_ASYMM:
      {
        dequantize<uint8_t, float>(src_tensor, dst_tensor);
        break;
      }
      case ir::DataType::QUANT_INT8_SYMM:
      {
        dequantize<int8_t, float>(src_tensor, dst_tensor);
        break;
      }
      case ir::DataType::QUANT_INT16_SYMM:
      {
        dequantize<int16_t, float>(src_tensor, dst_tensor);
        break;
      }
      default:
      {
        throw std::runtime_error("IPermuteFunction: Unsupported dequantization type");
        break;
      }
    }
  }
  else
  {
    throw std::runtime_error("IPermuteFunction: Unsupported type for type-aware quantization yet");
  }
}

} // namespace

namespace onert
{
namespace exec
{

void IPermuteFunction::IPermuteFunction::run()
{
  // TODO Optimization : Make control does not reach here? when (_src_tensors.size() == 0)
  assert(_src_tensors.size() == _dst_tensors.size());
  if (_src_tensors_offsets.size() == 0)
  {
    _src_tensors_offsets.resize(_src_tensors.size());
    _dst_tensors_offsets.resize(_dst_tensors.size());
  }
  assert(_src_tensors.size() == _src_tensors_offsets.size());
  assert(_src_tensors_offsets.size() == _dst_tensors_offsets.size());

  for (size_t i = 0; i < _src_tensors.size(); ++i)
  {
    auto src_tensor = _src_tensors.at(i);
    auto dst_tensor = _dst_tensors.at(i);
    auto &src_offsets = _src_tensors_offsets.at(i);
    auto &dst_offsets = _dst_tensors_offsets.at(i);
    if (src_tensor != dst_tensor)
    {
      const auto rank = src_tensor->getShape().rank();
      permute(src_tensor, dst_tensor, rank, src_offsets, dst_offsets);
    }
  }
}

void IPermuteFunction::permute(backend::ITensor *src_tensor, backend::ITensor *dst_tensor,
                               size_t rank, std::vector<size_t> &src_offsets,
                               std::vector<size_t> &dst_offsets)
{
  if (src_tensor->total_size() == 0)
  {
    assert(dst_tensor->total_size() == 0);
    return;
  }

  assert(src_tensor != dst_tensor);
  if (underlying_type(src_tensor->data_type()) != underlying_type(dst_tensor->data_type()))
  {
    typeAwareQuantize(src_tensor, dst_tensor);
    return;
  }

  switch (src_tensor->data_type())
  {
    case ir::DataType::FLOAT32:
      permute<float>(src_tensor, dst_tensor, rank, src_offsets, dst_offsets);
      break;
    case ir::DataType::INT32:
      permute<int32_t>(src_tensor, dst_tensor, rank, src_offsets, dst_offsets);
      break;
    case ir::DataType::UINT32:
      permute<uint32_t>(src_tensor, dst_tensor, rank, src_offsets, dst_offsets);
      break;
    case ir::DataType::BOOL8:
    case ir::DataType::QUANT_UINT8_ASYMM:
    case ir::DataType::UINT8:
      permute<uint8_t>(src_tensor, dst_tensor, rank, src_offsets, dst_offsets);
      break;
    case ir::DataType::QUANT_INT8_ASYMM:
    case ir::DataType::QUANT_INT8_SYMM:
      permute<int8_t>(src_tensor, dst_tensor, rank, src_offsets, dst_offsets);
      break;
    case ir::DataType::INT64:
      permute<int64_t>(src_tensor, dst_tensor, rank, src_offsets, dst_offsets);
      break;
    case ir::DataType::QUANT_INT16_SYMM:
      permute<int16_t>(src_tensor, dst_tensor, rank, src_offsets, dst_offsets);
      break;
    default:
      throw std::runtime_error("IPermuteFunction: Not supported data type");
      break;
  }
}

const std::type_info &IPermuteFunction::underlying_type(ir::DataType type) const
{
  switch (type)
  {
    case ir::DataType::FLOAT32:
      return typeid(float);
    case ir::DataType::INT32:
      return typeid(int32_t);
    case ir::DataType::UINT32:
      return typeid(uint32_t);
    case ir::DataType::INT64:
      return typeid(int64_t);
    case ir::DataType::BOOL8:
    case ir::DataType::QUANT_UINT8_ASYMM:
    case ir::DataType::UINT8:
      return typeid(uint8_t);
    case ir::DataType::QUANT_INT8_ASYMM:
    case ir::DataType::QUANT_INT8_SYMM:
      return typeid(int8_t);
    case ir::DataType::QUANT_INT16_SYMM:
      return typeid(int16_t);
    default:
      throw std::runtime_error("IPermuteFunction: Not supported data type");
  }
}

} // namespace exec
} // namespace onert
