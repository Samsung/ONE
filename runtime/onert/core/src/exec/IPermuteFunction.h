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
#include <memory>
#include <vector>
#include <unordered_map>

namespace onert
{
namespace exec
{

inline void UpdateOffsets(::onert::backend::ITensor *src, ::onert::backend::ITensor *dst,
                          const ::onert::ir::Shape &loop_shape, std::vector<size_t> &src_offsets,
                          std::vector<size_t> &dst_offsets)
{
  ShapeLoop(loop_shape, [&](const onert::ir::Coordinates &coords) {
    src_offsets.emplace_back(src->calcOffset(coords));
    dst_offsets.emplace_back(dst->calcOffset(coords));
  });
}

inline void CopyStatic(const uint8_t *src_buffer, uint8_t *dst_buffer,
                       const std::vector<size_t> &src_offsets,
                       const std::vector<size_t> &dst_offsets, size_t copy_len)
{
  assert(src_offsets.size() == dst_offsets.size());
  for (size_t i = 0; i < src_offsets.size(); ++i)
  {
    memcpy(dst_buffer + dst_offsets.at(i), src_buffer + src_offsets.at(i), copy_len);
  }
}

inline void CopyDynamic(const ::onert::backend::ITensor *src, const ::onert::backend::ITensor *dst,
                        uint8_t *dst_buffer, const ::onert::ir::Shape &loop_shape, size_t copy_len)
{
  ShapeLoop(loop_shape, [&](const onert::ir::Coordinates &coords) {
    // Copy src tensor's data to dst_buffer with calculated offset of dst tensor
    memcpy(dst_buffer + dst->calcOffset(coords), src->buffer() + src->calcOffset(coords), copy_len);
  });
}

class IPermuteFunction : public IFunction
{
public:
  virtual void run() override;

  virtual void prepare() override { optimize(); }

  virtual void optimize() = 0;

protected:
  void permute(backend::ITensor *src_tensor, backend::ITensor *dst_tensor, size_t rank,
               std::vector<size_t> &src_offsets, std::vector<size_t> &dst_offsets,
               const ir::PermuteType &permute_type);

private:
  // TODO make src const by proving const access()
  template <class T>
  void permute(backend::ITensor *src, backend::ITensor *dst, size_t rank,
               std::vector<size_t> &src_offsets, std::vector<size_t> &dst_offsets,
               const ir::PermuteType &permute_type)
  {
    assert(src->total_size() != 0 && dst->total_size() != 0);
    // If dst is subtensor, we have to use clEnqueueMapBuffer instead of clEnqueueWirteBuffer
    if (dst->needMemoryMap() && !dst->is_subtensor())
    {
      // A assertion to check mapping without calling map()
      // Now there is no case where both src and dst have cl buffer.
      assert(!src->needMemoryMap());

      if (!src->has_padding() && !dst->has_padding() && permute_type == ir::PermuteType::COPY)
      {
        src->access([&](backend::ITensor &) { dst->enqueueWriteBuffer(src->buffer(), false); });
      }
      else
      {
        // TODO Optimize this block in case of that padding size of dst is big.
        _buffers_map[dst].reserve(dst->total_size());
        auto dst_buffer = _buffers_map[dst].data();
        src->access([&](backend::ITensor &) {
          permute<T>(src, dst, rank, dst_buffer, dst->total_size(), src_offsets, dst_offsets,
                     permute_type);
        });
        dst->enqueueWriteBuffer(dst_buffer, false);
      }
    }
    else if (src->needMemoryMap() && !src->is_subtensor() && !src->has_padding() &&
             !dst->has_padding() && permute_type == ir::PermuteType::COPY)
    {
      assert(!dst->needMemoryMap());
      dst->access([&](backend::ITensor &) { src->enqueueReadBuffer(dst->buffer(), true); });
    }
    else
    {
      auto fn = [&](backend::ITensor &) {
        dst->access([&](backend::ITensor &) {
          permute<T>(src, dst, rank, dst->buffer(), dst->total_size(), src_offsets, dst_offsets,
                     permute_type);
        });
      };
      src->access(fn);
    }
  }

  template <class T>
  void permute(backend::ITensor *src, backend::ITensor *dst, size_t rank, uint8_t *dst_buffer,
               size_t dst_size, std::vector<size_t> &src_offsets, std::vector<size_t> &dst_offsets,
               const ir::PermuteType &permute_type)
  {
    assert(dst_buffer != nullptr);
    assert(dst_size == dst->total_size());

    if (rank == 4 && permute_type != ir::PermuteType::COPY)
    {
      switch (permute_type)
      {
        case ir::PermuteType::NHWC_TO_NCHW:
        {
          ir::FeatureShape shape;
          auto dst_shape = dst->getShape();
          shape.N = dst_shape.dim(0);
          shape.C = dst_shape.dim(1);
          shape.H = dst_shape.dim(2);
          shape.W = dst_shape.dim(3);

          typename feature::nchw::View<T>::Strides strides;
          const auto start_offset = dst->calcOffset({0, 0, 0, 0});
          strides.W = dst_shape.dim(3) == 1 ? 0 : dst->calcOffset({0, 0, 0, 1}) - start_offset;
          strides.H = dst_shape.dim(2) == 1 ? 0 : dst->calcOffset({0, 0, 1, 0}) - start_offset;
          strides.C = dst_shape.dim(1) == 1 ? 0 : dst->calcOffset({0, 1, 0, 0}) - start_offset;
          strides.N = dst_shape.dim(0) == 1 ? 0 : dst->calcOffset({1, 0, 0, 0}) - start_offset;

          const feature::nhwc::Reader<T> from(src);
          feature::nchw::View<T> into(shape, strides,
                                      reinterpret_cast<T *>(dst_buffer + start_offset), dst_size);
          feature::iterate(shape) << [&](uint32_t batch, uint32_t ch, uint32_t row, uint32_t col) {
            const auto value = from.at(batch, row, col, ch);
            into.at(batch, ch, row, col) = value;
          };
          break;
        }
        case ir::PermuteType::NCHW_TO_NHWC:
        {
          ir::FeatureShape shape;
          auto dst_shape = dst->getShape();
          shape.N = dst_shape.dim(0);
          shape.H = dst_shape.dim(1);
          shape.W = dst_shape.dim(2);
          shape.C = dst_shape.dim(3);

          typename feature::nhwc::View<T>::Strides strides;
          const auto start_offset = dst->calcOffset({0, 0, 0, 0});
          strides.C = dst_shape.dim(3) == 1 ? 0 : dst->calcOffset({0, 0, 0, 1}) - start_offset;
          strides.W = dst_shape.dim(2) == 1 ? 0 : dst->calcOffset({0, 0, 1, 0}) - start_offset;
          strides.H = dst_shape.dim(1) == 1 ? 0 : dst->calcOffset({0, 1, 0, 0}) - start_offset;
          strides.N = dst_shape.dim(0) == 1 ? 0 : dst->calcOffset({1, 0, 0, 0}) - start_offset;

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

      if (src->is_dynamic())
      {
        assert(dst->is_dynamic());
        CopyDynamic(src, dst, dst_buffer, loop_shape, copy_len);
      }
      else
      {
        // TODO Uncomment the assertion below
        // assert(!dst->is_dynamic() || dst is output of graph);
        if (src_offsets.size() == 0)
        {
          assert(dst_offsets.size() == 0);

          auto loop_shape = src->getShape();
          const auto copy_axis = loop_shape.rank() - 1;
          loop_shape.dim(copy_axis) = 1;
          UpdateOffsets(src, dst, loop_shape, src_offsets, dst_offsets);
        }
        CopyStatic(src->buffer(), dst_buffer, src_offsets, dst_offsets, copy_len);
      }
    }
  }

protected:
  // NOTE The typeid expression is lvalue expression which refers to an object with static storage
  //      duration, of the polymorphic type const std::type_info or of some type derived from it.
  //      So std::type_info is non-copyable
  const std::type_info &underlying_type(ir::DataType type) const;

protected:
  std::vector<backend::ITensor *> _src_tensors;
  std::vector<backend::ITensor *> _dst_tensors;
  std::vector<std::vector<size_t>> _src_tensors_offsets;
  std::vector<std::vector<size_t>> _dst_tensors_offsets;
  std::vector<ir::PermuteType> _permute_types;
  std::unordered_map<const backend::ITensor *, std::vector<uint8_t>> _buffers_map;
};

// Simple PermuteLayer
class PermuteLayer : public onert::exec::IPermuteFunction
{
public:
  PermuteLayer(const std::vector<onert::backend::ITensor *> &inputs,
               const std::vector<onert::backend::ITensor *> &outputs,
               const std::vector<ir::PermuteType> &types)
  {
    assert(inputs.size() == outputs.size());
    assert(inputs.size() == types.size());
    _src_tensors = inputs;
    _dst_tensors = outputs;
    _permute_types = types;
  }
  virtual ~PermuteLayer() {}
  void optimize() override {}
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_I_PERMUTE_FUNCTION_H__
