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

#ifndef __ONERT_EXEC_I_PERMUTE_FUNCTION_H__
#define __ONERT_EXEC_I_PERMUTE_FUNCTION_H__

#include "feature/IndexIterator.h"
#include "feature/nchw/Reader.h"
#include "feature/nchw/View.h"
#include "feature/nhwc/Reader.h"
#include "feature/nhwc/View.h"

#include "backend/ITensor.h"
#include "exec/IFunction.h"
#include "ir/Index.h"
#include "ir/Shape.h"
#include <memory>
#include <typeinfo>
#include "util/Utils.h"
#include <vector>
#include <unordered_map>

namespace onert
{
namespace exec
{

class IPermuteFunction : public IFunction
{
protected:
  enum class PermuteType
  {
    NHWC_TO_NCHW,
    NCHW_TO_NHWC,
    COPY
  };

public:
  virtual void run() override
  {
    // TODO Optimization : Make control does not reach here? when (_src_tensors.size() == 0)
    assert(_src_tensors.size() == _dst_tensors.size());
    auto src_it = _src_tensors.begin();
    auto dst_it = _dst_tensors.begin();
    while (src_it != _src_tensors.end())
    {
      auto src_tensor = *src_it;
      auto dst_tensor = *dst_it;
      if (src_tensor != dst_tensor)
      {
        const auto rank = src_tensor->num_dimensions();
        permute(src_tensor, dst_tensor, rank);
      }
      src_it++;
      dst_it++;
    }
  }

  virtual void prepare() override { optimize(); }

  virtual void optimize() = 0;

protected:
  void permute(backend::ITensor *src_tensor, backend::ITensor *dst_tensor, size_t rank)
  {
    if (src_tensor->total_size() == 0)
    {
      assert(dst_tensor->total_size() == 0);
      return;
    }

    assert(src_tensor != dst_tensor);
    assert(underlying_type(src_tensor->data_type()) == underlying_type(dst_tensor->data_type()));
    switch (src_tensor->data_type())
    {
      case ir::DataType::FLOAT32:
        permute<float>(src_tensor, dst_tensor, rank);
        break;
      case ir::DataType::INT32:
        permute<int32_t>(src_tensor, dst_tensor, rank);
        break;
      case ir::DataType::UINT32:
        permute<uint32_t>(src_tensor, dst_tensor, rank);
        break;
      case ir::DataType::BOOL8:
      case ir::DataType::QUANT_UINT8_ASYMM:
      case ir::DataType::UINT8:
        permute<uint8_t>(src_tensor, dst_tensor, rank);
        break;
      case ir::DataType::QUANT_INT8_ASYMM:
      case ir::DataType::QUANT_INT8_SYMM:
        permute<int8_t>(src_tensor, dst_tensor, rank);
        break;
      case ir::DataType::INT64:
        permute<int64_t>(src_tensor, dst_tensor, rank);
        break;
      default:
        throw std::runtime_error("IPermuteFunction: Not supported data type");
        break;
    }
  }

private:
  // TODO make src const by proving const access()
  template <class T> void permute(backend::ITensor *src, backend::ITensor *dst, size_t rank)
  {
    assert(src->total_size() != 0 && dst->total_size() != 0);
    // If dst is subtensor, we have to use clEnqueueMapBuffer instead of clEnqueueWirteBuffer
    if (dst->hasClBuffer() && !dst->is_subtensor())
    {
      // A assertion to check mapping without calling map()
      // Now there is no case where both src and dst have cl buffer.
      assert(!src->hasClBuffer());

      if (!src->has_padding() && !dst->has_padding() && src->layout() == dst->layout())
      {
        src->access([&](backend::ITensor &) { dst->enqueueWriteBuffer(src->buffer()); });
      }
      else
      {
        _buffers_map[dst].reserve(dst->total_size());
        auto dst_buffer = _buffers_map[dst].data();
        src->access(
            [&](backend::ITensor &) { permute<T>(src, dst, rank, dst_buffer, dst->total_size()); });
        dst->enqueueWriteBuffer(dst_buffer);
      }
    }
    else if (src->hasClBuffer() && !src->is_subtensor() && !src->has_padding() &&
             !dst->has_padding() && src->layout() == dst->layout())
    {
      assert(!dst->hasClBuffer());
      dst->access([&](backend::ITensor &) { src->enqueueReadBuffer(dst->buffer()); });
    }
    else
    {
      auto fn = [&](backend::ITensor &) {
        dst->access([&](backend::ITensor &) {
          permute<T>(src, dst, rank, dst->buffer(), dst->total_size());
        });
      };
      src->access(fn);
    }
  }

  template <class T>
  void permute(backend::ITensor *src, backend::ITensor *dst, size_t rank, uint8_t *dst_buffer,
               size_t dst_size)
  {
    assert(dst_buffer != nullptr);
    assert(dst_size == dst->total_size());
    const auto permute_type = [&]() -> PermuteType {
      if (src->layout() == ir::Layout::NHWC && dst->layout() == ir::Layout::NCHW)
      {
        return PermuteType::NHWC_TO_NCHW;
      }
      else if (src->layout() == ir::Layout::NCHW && dst->layout() == ir::Layout::NHWC)
      {
        return PermuteType::NCHW_TO_NHWC;
      }
      else
      {
        return PermuteType::COPY;
      }
    }();
    if (rank == 4 && permute_type != PermuteType::COPY)
    {
      switch (permute_type)
      {
        case PermuteType::NHWC_TO_NCHW:
        {
          ir::FeatureShape shape;
          shape.N = dst->dimension(0);
          shape.C = dst->dimension(1);
          shape.H = dst->dimension(2);
          shape.W = dst->dimension(3);

          typename feature::nchw::View<T>::Strides strides;
          const auto start_offset = dst->calcOffset({0, 0, 0, 0});
          strides.W = dst->dimension(3) == 1 ? 0 : dst->calcOffset({0, 0, 0, 1}) - start_offset;
          strides.H = dst->dimension(2) == 1 ? 0 : dst->calcOffset({0, 0, 1, 0}) - start_offset;
          strides.C = dst->dimension(1) == 1 ? 0 : dst->calcOffset({0, 1, 0, 0}) - start_offset;
          strides.N = dst->dimension(0) == 1 ? 0 : dst->calcOffset({1, 0, 0, 0}) - start_offset;

          const feature::nhwc::Reader<T> from(src);
          feature::nchw::View<T> into(shape, strides,
                                      reinterpret_cast<T *>(dst_buffer + start_offset), dst_size);
          feature::iterate(shape) << [&](uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) {
            const auto value = from.at(batch, row, col, ch);
            into.at(batch, ch, row, col) = value;
          };
          break;
        }
        case PermuteType::NCHW_TO_NHWC:
        {
          ir::FeatureShape shape;
          shape.N = dst->dimension(0);
          shape.H = dst->dimension(1);
          shape.W = dst->dimension(2);
          shape.C = dst->dimension(3);

          typename feature::nhwc::View<T>::Strides strides;
          const auto start_offset = dst->calcOffset({0, 0, 0, 0});
          strides.C = dst->dimension(3) == 1 ? 0 : dst->calcOffset({0, 0, 0, 1}) - start_offset;
          strides.W = dst->dimension(2) == 1 ? 0 : dst->calcOffset({0, 0, 1, 0}) - start_offset;
          strides.H = dst->dimension(1) == 1 ? 0 : dst->calcOffset({0, 1, 0, 0}) - start_offset;
          strides.N = dst->dimension(0) == 1 ? 0 : dst->calcOffset({1, 0, 0, 0}) - start_offset;

          const feature::nchw::Reader<T> from(src);
          feature::nhwc::View<T> into(shape, strides,
                                      reinterpret_cast<T *>(dst_buffer + start_offset), dst_size);
          feature::iterate(shape) << [&](uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) {
            const auto value = from.at(batch, ch, row, col);
            into.at(batch, row, col, ch) = value;
          };
          break;
        }
        default:
        {
          throw std::runtime_error("Unsupported Permutation");
          break;
        }
      }
    }
    else if (!src->has_padding() && !dst->has_padding())
    {
      auto src_size = src->total_size();
      assert(src_size <= dst->total_size());
      memcpy(dst_buffer, src->buffer(), src_size);
    }
    else
    {
      auto loop_shape = src->getShape();
      const auto copy_axis = loop_shape.rank() - 1;
      const auto copy_len = loop_shape.dim(copy_axis) * sizeof(T);
      loop_shape.dim(copy_axis) = 1;
      ShapeLoop(loop_shape, [&](const onert::ir::Coordinates &coords) {
        // Copy src tensor's data to dst_buffer with calculated offset of dst tensor
        memcpy(dst_buffer + dst->calcOffset(coords), src->buffer() + src->calcOffset(coords),
               copy_len);
      });
    }
  }

protected:
  // NOTE The typeid expression is lvalue expression which refers to an object with static storage
  //      duration, of the polymorphic type const std::type_info or of some type derived from it.
  //      So std::type_info is non-copyable
  const std::type_info &underlying_type(ir::DataType type) const
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
      default:
        throw std::runtime_error("IPermuteFunction: Not supported data type");
    }
  }

protected:
  std::vector<backend::ITensor *> _src_tensors;
  std::vector<backend::ITensor *> _dst_tensors;
  std::unordered_map<const backend::ITensor *, std::vector<uint8_t>> _buffers_map;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_I_PERMUTE_FUNCTION_H__
