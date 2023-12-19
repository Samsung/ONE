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

#ifndef __ONERT_BACKEND_BUILTIN_KERNEL_PERMUTELAYER_H__
#define __ONERT_BACKEND_BUILTIN_KERNEL_PERMUTELAYER_H__

#include "../ExternalContext.h"
#include "../../../exec/IPermuteFunction.h"

#include <ruy/thread_pool.h> // from @ruy

namespace onert
{
namespace backend
{
namespace builtin
{
namespace kernel
{

class PermuteLayer : public onert::exec::IPermuteFunction
{
public:
  PermuteLayer(const std::vector<ITensor *> &src_tensors, const std::vector<ITensor *> &dst_tensors,
               const std::shared_ptr<ExternalContext> &external_context);

  void optimize() override;

  void run() override;

private:
  std::shared_ptr<ExternalContext> _external_context;

private:
  void appendPermuteTasks(const ITensor *src_tensor, ITensor *dst_tensor,
                          const ir::Shape &loop_shape, size_t size);

  void runPermuteTasks(backend::ITensor *src, uint8_t *dst_buffer);

  struct PermuteWorkerTask : ruy::Task
  {
    using Strides = ir::Coordinates;

    PermuteWorkerTask(const ITensor &src_tensor, ITensor &dst_tensor,
                      const ir::Coordinates &start_coords, const ir::Shape &loop_shape, size_t size)
      : _src_buffer{src_tensor.buffer()}, _dst_buffer{dst_tensor.buffer()},
        _src_start_offset{src_tensor.calcOffset(start_coords)},
        _dst_start_offset{dst_tensor.calcOffset(start_coords)}, _src_strides{}, _dst_strides{},
        _loop_shape{loop_shape}, _size{size}, _src_layout{src_tensor.layout()},
        _dst_layout{dst_tensor.layout()}, _is_permutation{true}
    {
      // Set strides
      setStrides(src_tensor, &_src_strides);
      setStrides(dst_tensor, &_dst_strides);

      _is_permutation = (_src_layout != _dst_layout && loop_shape.rank() == 4);
    }
    // Constructor for a copy
    PermuteWorkerTask(const uint8_t *src_buffer, uint8_t *dst_buffer, uint32_t src_start_offset,
                      uint32_t dst_start_offset, size_t size)
      : _src_buffer{src_buffer}, _dst_buffer{dst_buffer}, _src_start_offset{src_start_offset},
        _dst_start_offset{dst_start_offset}, _src_strides{0}, _dst_strides{0}, _loop_shape{1},
        _size{size}, _src_layout{}, _dst_layout{}, _is_permutation{false}
    {
      // DO NOTHING
    }
    void setBuffers(const uint8_t *src_buffer, uint8_t *dst_buffer)
    {
      _src_buffer = src_buffer;
      _dst_buffer = dst_buffer;
    }
    void Run() override
    {
      ShapeLoop(_loop_shape, [&](const onert::ir::Coordinates &coords) {
        size_t src_offset = _src_start_offset;
        size_t dst_offset = _dst_start_offset;
        assert(static_cast<size_t>(_loop_shape.rank()) == coords.size());
        ir::Coordinates dst_coords = coords;
        if (_is_permutation)
        {
          dst_coords = ir::convertCoordinates(coords, _src_layout, _dst_layout);
        }
        for (auto i = 0; i < _loop_shape.rank(); ++i)
        {
          assert(coords[i] >= 0 && dst_coords[i] >= 0);
          src_offset += coords[i] * _src_strides[i];
          dst_offset += dst_coords[i] * _dst_strides[i];
        }
        memcpy(_dst_buffer + dst_offset, _src_buffer + src_offset, _size);
      });
    }

  private:
    void setStrides(const ITensor &tensor, Strides *strides)
    {
      auto shape = tensor.getShape();
      const size_t rank = shape.rank();
      for (size_t i = 0; i < rank; ++i)
      {
        ir::Coordinates no_step(rank), one_step(rank);
        one_step.set(i, 1);
        if (shape.dim(i) > 1)
        {
          strides->set(i, tensor.calcOffset(one_step) - tensor.calcOffset(no_step));
        }
        else
        {
          // If dimension value is 0 or 1, the stride of the dimension will be not used
          // Do not call calcOffset() with coordinate value that is greater than dimension value
          strides->set(i, 0);
        }
        assert((*strides)[i] >= 0);
      }
    }

  private:
    const uint8_t *_src_buffer;
    uint8_t *_dst_buffer;
    size_t _src_start_offset;
    size_t _dst_start_offset;
    Strides _src_strides;
    Strides _dst_strides;
    const ir::Shape _loop_shape;
    const size_t _size;
    const ir::Layout _src_layout;
    const ir::Layout _dst_layout;
    bool _is_permutation;
  };
  std::unordered_map<const ITensor *, std::vector<PermuteWorkerTask>> _tasks_map;
};

} // namespace kernel
} // namespace builtin
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BUILTIN_KERNEL_PERMUTELAYER_H__
