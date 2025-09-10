/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_TOPK_V2_H__
#define __NNFW_CKER_TOPK_V2_H__

#include "cker/Shape.h"

namespace nnfw::cker
{

template <typename T, typename Tidx> class TopContainer
{
public:
  TopContainer() = delete;
  TopContainer(uint32_t k, uint32_t row_size) : k_(k)
  {
    container_.reserve(std::min(k, row_size) + 1);
  }

  void start_collecting(const T *values)
  {
    values_ = values;
    container_.clear();
    is_heap_ = false;
  }

  void push(Tidx a)
  {
    auto comparator = [this](Tidx a_, Tidx b_) { return compare_fun(a_, b_); };
    if (!is_heap_)
    {
      container_.push_back(a);
      if (container_.size() == k_ + 1)
      {
        std::make_heap(container_.begin(), container_.end(), comparator);
        std::pop_heap(container_.begin(), container_.end(), comparator);
        container_.pop_back();
        is_heap_ = true;
      }
    }
    else if (comparator(a, container_.front()))
    {
      // Due to how we defined comparator / compare_fun, container_.front()
      // contains the index of the smallest of the top-k elements seen so far.
      //
      // If control reaches this point, we know that the current index a
      // corresponds to an element which is bigger than the smallest of the
      // top-k elements seen so far.  Hence, we have to update the indices of
      // the top-k elements, by removing the index of the smallest top-k
      // element, adding a, and making sure container_[0:k] is still a heap.
      std::pop_heap(container_.begin(), container_.end(), comparator);
      container_.back() = a;
      std::push_heap(container_.begin(), container_.end(), comparator);
    }
  }

  const std::vector<Tidx> &sorted_result()
  {
    auto comparator = [this](Tidx a, Tidx b) { return compare_fun(a, b); };
    if (!is_heap_)
    {
      // Note: due to the way we defined compare_fun (see comments for that
      // function) std::sort puts the indices from container_ in decreasing
      // order of the corresponding elements.
      std::sort(container_.begin(), container_.end(), comparator);
    }
    else
    {
      std::sort_heap(container_.begin(), container_.end(), comparator);
    }
    return container_;
  }

private:
  const uint32_t k_;

  // container_[0,k) holds the indices of the largest k elements from values_
  // seen so far.  If more than k elements are pushed, then elements are
  // maintained in a min-heap order: container_.front() is
  // the index of the smallest of the top-k elements see so far.
  std::vector<Tidx> container_;

  // Once more than k elements are pushed, the container becomes a min heap,
  // and is_heap_ becomes true.
  bool is_heap_ = false;

  const T *values_ = nullptr;

  // Compares indices a and b based on the corresponding elements from values_.
  //
  // Intuitively, compare_fun(a, b) returns true iff values_[b] < values_[a]
  // (notice the inversion of direction, not a typo); ties (==) are broken in
  // favor of earlier elements (i.e., a < b).
  bool compare_fun(Tidx a, Tidx b) const
  {
    if (values_[b] < values_[a])
    {
      return true;
    }
    else if (values_[b] > values_[a])
    {
      return false;
    }
    else
    {
      return a < b;
    }
  }
};

template <typename T, typename Tidx = int32_t>
inline void TopKV2(const Shape &input_shape, const T *input_data, const uint32_t k,
                   T *output_value_data, Tidx *output_indices_data)
{
  const int32_t row_size = input_shape.Dims(input_shape.DimensionsCount() - 1);
  int32_t num_rows = 1;
  for (int32_t i = 0; i < input_shape.DimensionsCount() - 1; ++i)
  {
    num_rows *= input_shape.Dims(i);
  }

  TopContainer<T, Tidx> topc(k, row_size);
  for (int32_t row = 0; row < num_rows; ++row)
  {
    const T *values_row = input_data + row * row_size;
    topc.start_collecting(values_row);
    for (int32_t c = 0; c < row_size; ++c)
    {
      topc.push(c);
    }

    // Prepare output buffers.
    Tidx *indexes_row = output_indices_data + row * k;
    T *output_row = output_value_data + row * k;
    // We always assume that the output is sorted.
    const auto &top_k = topc.sorted_result();
    std::copy(top_k.begin(), top_k.end(), indexes_row);
    std::transform(top_k.begin(), top_k.end(), output_row,
                   [values_row](const int32_t loc) { return values_row[loc]; });
  }
}

} // namespace nnfw::cker

#endif // __NNFW_CKER_TOPK_V2_H__
