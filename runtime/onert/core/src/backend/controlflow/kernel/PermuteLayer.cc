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

#include "PermuteLayer.h"

#include "exec/ShapeConverter.h"

#include "ruy/context.h" // from @ruy

namespace onert
{
namespace backend
{
namespace controlflow
{
namespace kernel
{

PermuteLayer::PermuteLayer(const std::vector<ITensor *> &src_tensors,
                           const std::vector<ITensor *> &dst_tensors,
                           const std::shared_ptr<ExternalContext> &external_context)
    : _external_context{external_context}, _tasks_map{}
{
  assert(src_tensors.size() == dst_tensors.size());
  _src_tensors = src_tensors;
  _dst_tensors = dst_tensors;
  _src_tensors_offsets.resize(src_tensors.size());
  _dst_tensors_offsets.resize(dst_tensors.size());
}

void PermuteLayer::optimize()
{
  // Remove copying of tensor as nullptr
  auto src_it = _src_tensors.begin();
  auto dst_it = _dst_tensors.begin();
  auto src_offsets_it = _src_tensors_offsets.begin();
  auto dst_offsets_it = _dst_tensors_offsets.begin();
  while (src_it != _src_tensors.end())
  {
    if ((*src_it == *dst_it) || (*src_it == nullptr || *dst_it == nullptr))
    {
      src_it = _src_tensors.erase(src_it);
      dst_it = _dst_tensors.erase(dst_it);
      src_offsets_it = _src_tensors_offsets.erase(src_offsets_it);
      dst_offsets_it = _dst_tensors_offsets.erase(dst_offsets_it);
    }
    else
    {
      auto src = *src_it;
      auto dst = *dst_it;
      src_offsets_it->resize(0);
      dst_offsets_it->resize(0);
      if (underlying_type(src->data_type()) != underlying_type(dst->data_type()))
        throw std::runtime_error("data type does not match");
      const auto permute_type = [&]() -> PermuteType {
        if (src->num_dimensions() == 4 && src->layout() == ir::Layout::NHWC &&
            dst->layout() == ir::Layout::NCHW)
        {
          return PermuteType::NHWC_TO_NCHW;
        }
        else if (src->num_dimensions() == 4 && src->layout() == ir::Layout::NCHW &&
                 dst->layout() == ir::Layout::NHWC)
        {
          return PermuteType::NCHW_TO_NHWC;
        }
        else
        {
          return PermuteType::COPY;
        }
      }();
      auto fn = [&](backend::ITensor &src_tensor) {
        dst->access([&](backend::ITensor &dst_tensor) {
          // NOTE The buffer of both tensor can be nullptr in this step
          const auto data_size = ir::sizeOfDataType(src_tensor.data_type());

          if (permute_type == PermuteType::COPY)
          {
            if ((!src_tensor.has_padding() && !dst_tensor.has_padding()))
            {
              const auto num_elements = src_tensor.getShape().num_elements();
              const int thread_count = _external_context->ruy_context()->max_num_threads() <
                                               static_cast<int>(num_elements)
                                           ? _external_context->ruy_context()->max_num_threads()
                                           : num_elements;

              std::vector<PermuteWorkerTask> tasks;
              auto start = 0;
              for (auto i = 0; i < thread_count; ++i)
              {
                int end = start + (num_elements - start) / (thread_count - i);
                tasks.emplace_back(src_tensor.buffer(), dst_tensor.buffer(), start * data_size,
                                   start * data_size, (end - start) * data_size);
                start = end;
              }
              assert(tasks.size() >= 1);
              _tasks_map[src] = std::move(tasks);
            }
            else
            {
              auto loop_shape = src_tensor.getShape();

              auto copy_axis = loop_shape.rank() - 1;
              copy_axis = copy_axis < 0 ? 1 : copy_axis;
              const auto copy_len = loop_shape.dim(copy_axis) * data_size;
              loop_shape.dim(copy_axis) = 1;

              appendPermuteTasks(src, dst, loop_shape, copy_len);
            }
          }
          else
          {
            assert(src_tensor.num_dimensions() == 4 && (permute_type == PermuteType::NHWC_TO_NCHW ||
                                                        permute_type == PermuteType::NCHW_TO_NHWC));
            const auto loop_shape = src_tensor.getShape();
            const auto copy_len = data_size;

            appendPermuteTasks(src, dst, loop_shape, copy_len);
          }
        });
      };
      src->access(fn);
      src_it++;
      dst_it++;
      src_offsets_it++;
      dst_offsets_it++;
    }
  }
}

void PermuteLayer::appendPermuteTasks(const ITensor *src_tensor, ITensor *dst_tensor,
                                      const ir::Shape &loop_shape, size_t size)
{
  size_t distributed_dim = 0;
  if (src_tensor->layout() == dst_tensor->layout())
  {
    for (size_t i = 1; i < src_tensor->num_dimensions() - 1; ++i)
    {
      distributed_dim =
          src_tensor->dimension(distributed_dim) < src_tensor->dimension(i) ? i : distributed_dim;
    }
  }
  const auto distributed_dim_val = src_tensor->dimension(distributed_dim);
  const int thread_count =
      _external_context->ruy_context()->max_num_threads() < static_cast<int>(distributed_dim_val)
          ? _external_context->ruy_context()->max_num_threads()
          : distributed_dim_val;
  // NOTE Do not remove this assertion. It would cause performance degradation by new threads to be
  // created in the context's thread pool
  assert(thread_count <= _external_context->ruy_context()->max_num_threads());

  std::vector<PermuteWorkerTask> tasks;
  int start = 0;
  auto one_thread_loop_shape = loop_shape;
  for (auto i = 0; i < thread_count; ++i)
  {
    ir::Coordinates start_coords(one_thread_loop_shape.rank());
    start_coords.set(distributed_dim, start);
    int end = start + (distributed_dim_val - start) / (thread_count - i);
    one_thread_loop_shape.dim(distributed_dim) = end - start;
    tasks.emplace_back(*src_tensor, *dst_tensor, start_coords, one_thread_loop_shape, size);
    start = end;
  }
  assert(tasks.size() >= 1);
  _tasks_map[src_tensor] = std::move(tasks);
}

void PermuteLayer::runPermuteTasks(backend::ITensor *src, uint8_t *dst_buffer)
{
  assert(src->getShape().num_elements() * ir::sizeOfDataType(src->data_type()) <=
         src->total_size());
  std::vector<PermuteWorkerTask> &tasks = _tasks_map.at(src);
  for (size_t i = 0; i < tasks.size(); ++i)
  {
    tasks.at(i).setBuffers(src->buffer(), dst_buffer);
  }
  assert(tasks.size() >= 1);
  _external_context->ruy_context()->mutable_thread_pool()->Execute(tasks.size(), tasks.data());
}

void PermuteLayer::run()
{
  assert(_src_tensors.size() == _dst_tensors.size());
  // PermuteLayer infers dynamic shape inside itself whenever run is called for the following
  // reasons:
  // 1. PermuteLayer has to access dynamic tensor manager for input/output tensors of other backends
  // 2. Other controlflow operation(If/While) uses this layout for copying tensors of other
  // subgraphs(with other backends)
  // 3. This infering code is placed here to avoid duplicated code that can be caused by above 2
  // reasons

  // check if output is not dynamic
  for (size_t i = 0; i < _src_tensors.size(); ++i)
  {
    auto dst_tensor = _dst_tensors.at(i);
    auto src_tensor = _src_tensors.at(i);
    if (src_tensor->is_dynamic() || dst_tensor->is_dynamic())
    {
      // getting output shape
      auto src_shape = src_tensor->getShape();

      // set output shape and output buffer
      ir::Shape new_shape =
          exec::convertShape(src_shape, src_tensor->layout(), dst_tensor->layout());

      try
      {
        if (!dst_tensor->applyShape(new_shape))
          throw std::runtime_error{
              "Error: PermuteLayer: output's TensorManager does not support dynamic tensor"};
        assert(dst_tensor->buffer() != nullptr);
      }
      catch (const std::out_of_range &e)
      {
        std::cerr << "Error: out_of_range in PermuteLayer: output's TensorManager does not support "
                     "dynamic tensor"
                  << '\n';
        throw;
      }
    }
    assert(exec::convertShape(src_tensor->getShape(), src_tensor->layout(), dst_tensor->layout()) ==
           dst_tensor->getShape());
  }
  assert(_src_tensors.size() == _dst_tensors.size());
  assert(_src_tensors.size() == _src_tensors_offsets.size());
  assert(_dst_tensors.size() == _dst_tensors_offsets.size());
  auto src_it = _src_tensors.begin();
  auto dst_it = _dst_tensors.begin();
  auto src_offsets_it = _src_tensors_offsets.begin();
  auto dst_offsets_it = _dst_tensors_offsets.begin();
  while (src_it != _src_tensors.end())
  {
    auto src = *src_it;
    auto dst = *dst_it;
    auto &src_offsets = *src_offsets_it;
    auto &dst_offsets = *dst_offsets_it;

    if (src->total_size() == 0)
    {
      assert(dst->total_size() == 0);
    }
    else
    {
      if (src != dst)
      {
        // Conditions to run permutation with multithreading
        // 1. The tasks for multithreathing was created
        // 2. The tasks's size > 1
        // 3. Both tensors are not dynamic
        if (_tasks_map.find(src) == _tasks_map.end() || _tasks_map.at(src).size() == 1 ||
            src->is_dynamic() || dst->is_dynamic())
        {
          permute(src, dst, src->num_dimensions(), src_offsets, dst_offsets);
        }
        // If dst is subtensor, we have to use clEnqueueMapBuffer instead of clEnqueueWirteBuffer
        else if (dst->needMemoryMap() && !dst->is_subtensor())
        {
          if (!src->has_padding() && !dst->has_padding() && src->layout() == dst->layout())
          {
            // This is more effective than multi-threading
            src->access([&](backend::ITensor &) { dst->enqueueWriteBuffer(src->buffer(), false); });
          }
          else
          {
            // TODO Optimize this block in case of that padding size of dst is big.
            _buffers_map[dst].reserve(dst->total_size());
            auto dst_buffer = _buffers_map[dst].data();

            src->access([&](backend::ITensor &) { runPermuteTasks(src, dst_buffer); });
            dst->enqueueWriteBuffer(dst_buffer, false);
          }
        }
        else if (src->needMemoryMap() && !src->is_subtensor() && !src->has_padding() &&
                 !dst->has_padding() && src->layout() == dst->layout())
        {
          // This is more effective than multi-threading
          assert(!dst->needMemoryMap());
          dst->access([&](backend::ITensor &) { src->enqueueReadBuffer(dst->buffer(), true); });
        }
        else
        {
          auto fn = [&](backend::ITensor &) {
            dst->access([&](backend::ITensor &) { runPermuteTasks(src, dst->buffer()); });
          };
          src->access(fn);
        }
      }
    }
    src_it++;
    dst_it++;
    src_offsets_it++;
    dst_offsets_it++;
  }
}

} // namespace kernel
} // namespace controlflow
} // namespace backend
} // namespace onert
