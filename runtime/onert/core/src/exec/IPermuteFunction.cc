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

inline nnfw::cker::Shape getShape(const backend::IPortableTensor *tensor)
{
  const ir::Shape shape = tensor->getShape();

  assert(tensor->layout() == ir::Layout::NHWC);

  auto rank = shape.rank();
  nnfw::cker::Shape ret(rank);
  auto data = ret.DimsData();
  for (int i = 0; i < rank; ++i)
  {
    data[i] = shape.dim(i);
  }
  return ret;
}

template <typename SRC_T, typename DST_T,
          std::enable_if_t<std::is_base_of<backend::ITensor, SRC_T>::value &&
                             std::is_base_of<backend::ITensor, DST_T>::value,
                           bool> = true>
void typeAwareQuantize(const SRC_T *src_tensor, DST_T *dst_tensor)
{
  if (dynamic_cast<const backend::IPortableTensor *>(src_tensor) == nullptr ||
      src_tensor->layout() != ir::Layout::NHWC ||
      dynamic_cast<backend::IPortableTensor *>(dst_tensor) == nullptr ||
      dst_tensor->layout() != ir::Layout::NHWC)
  {
    throw std::runtime_error("Currently, type-aware quantization supports only potable tensors");
  }

  // TODO Use optimized kernels
  // TODO Support other types
  const auto src = nnfw::misc::polymorphic_downcast<const backend::IPortableTensor *>(src_tensor);
  auto dst = nnfw::misc::polymorphic_downcast<backend::IPortableTensor *>(dst_tensor);
  if (src->data_type() == ir::DataType::FLOAT32)
  {
    switch (dst->data_type())
    {
      case ir::DataType::QUANT_UINT8_ASYMM:
      {
        nnfw::cker::Quantize(getShape(src), reinterpret_cast<const float *>(src->buffer()),
                             getShape(dst), reinterpret_cast<uint8_t *>(dst->buffer()),
                             dst->data_scale(), dst->data_zero_point());
        break;
      }
      case ir::DataType::QUANT_INT8_SYMM:
      {
        nnfw::cker::Quantize(getShape(src), reinterpret_cast<const float *>(src->buffer()),
                             getShape(dst), reinterpret_cast<int8_t *>(dst->buffer()),
                             dst->data_scale(), dst->data_zero_point());
        break;
      }
      case ir::DataType::QUANT_INT16_SYMM:
      {
        nnfw::cker::Quantize(getShape(src), reinterpret_cast<const float *>(src->buffer()),
                             getShape(dst), reinterpret_cast<int16_t *>(dst->buffer()),
                             dst->data_scale(), dst->data_zero_point());
        break;
      }
      default:
      {
        throw std::runtime_error("IPermuteFunction: Unsupported quantization type");
        break;
      }
    }
  }
  else if (dst->data_type() == ir::DataType::FLOAT32)
  {
    switch (src->data_type())
    {
      case ir::DataType::QUANT_UINT8_ASYMM:
      {
        nnfw::cker::Dequantize(getShape(src), reinterpret_cast<const uint8_t *>(src), getShape(dst),
                               reinterpret_cast<float *>(dst), src->data_scale(),
                               src->data_zero_point());
        break;
      }
      case ir::DataType::QUANT_INT8_SYMM:
      {
        nnfw::cker::Dequantize(getShape(src), reinterpret_cast<const int8_t *>(src), getShape(dst),
                               reinterpret_cast<float *>(dst), src->data_scale(),
                               src->data_zero_point());
        break;
      }
      case ir::DataType::QUANT_INT16_SYMM:
      {
        nnfw::cker::Dequantize(getShape(src), reinterpret_cast<const int16_t *>(src), getShape(dst),
                               reinterpret_cast<float *>(dst), src->data_scale(),
                               src->data_zero_point());
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
