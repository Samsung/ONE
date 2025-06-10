/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

/**
 * @file topk_v2.h
 * @brief This file contains TopK method and TopContainer class for TopK operation
 * @ingroup COM_AI_RUNTIME
 */

#ifndef __NNFW_RT_OPTIMIZED_OPS_TOPK_V2_H__
#define __NNFW_RT_OPTIMIZED_OPS_TOPK_V2_H__

typedef int32_t int32;

namespace nnfw
{
namespace rt
{
namespace optimized_ops
{
/**
 * @brief class to define TopK operation
 * @note The follwing codes are impemented and modified while referring to TFLite topk_v2.cc file.
 * TopK_v2 of NN Runtime supports TENSOR_FLOAT32, TENSOR_QUANT8_ASYMM, TENSOR_INT32 other than
 * TFLite.
 * (TFLite additionaly supports kTfLiteInt64.)
 *
 * The class that collects top indexes of k values. Based on template
 * tensorflow::gtl::TopN<> but, for optimization,
 * it re-uses the same container.
 */
template <typename T> class TopContainer
{
public:
  /**
   * @brief Prevent default constructor of of this class
   */
  TopContainer() = delete;
  /**
   * @brief Constructor with params
   * @param [in] row_size Size of row in data
   * @param [in] k The top k predictions
   */
  TopContainer(int32 k, int32 row_size) : k_(k), container_(), values_(nullptr)
  {
    container_.reserve(std::min(k, row_size) + 1);
  }

  /**
   * @brief Prevent instances of this class from being copied (As this class contains pointers)
   * @param [in] topContainer To copy
   */
  TopContainer(const TopContainer &) = delete;
  /*
   * @brief Prevent instances of this class from being copied (As this class contains pointers)
   * @param [in] topContainer To copy
   * @return Reference of TopContainer
   */
  TopContainer &operator=(const TopContainer &) = delete;

  /**
   * @brief Start collecting
   * @param [in] values To set as values
   * @return N/A
   */
  void start_collecting(const T *values)
  {
    values_ = values;
    container_.clear();
  }

  /**
   * @brief Push a value to be compared for topk
   * @param [in] a A value to compare
   * @return N/A
   */
  void push(int32 a)
  {
    auto comparator = [this](int32 a, int32 b) { return compare_fun(a, b); };
    if (container_.size() <= (size_t)k_)
    {
      container_.push_back(a);
      if (container_.size() == (size_t)(k_ + 1))
      {
        std::make_heap(container_.begin(), container_.end(), comparator);
        std::pop_heap(container_.begin(), container_.end(), comparator);
      }
    }
    else if (comparator(a, container_.front()))
    {
      container_.back() = a;
      std::push_heap(container_.begin(), container_.end(), comparator);
      std::pop_heap(container_.begin(), container_.end(), comparator);
    }
  }

  /**
   * @brief Get sorted result from pushed values
   * @return Reference of vector with sorted values
   */
  const std::vector<int32> &sorted_result()
  {
    auto comparator = [this](int32 a, int32 b) { return compare_fun(a, b); };
    if (container_.size() <= (size_t)(k_))
    {
      std::sort(container_.begin(), container_.end(), comparator);
    }
    else
    {
      std::sort_heap(container_.begin(), container_.end() - 1, comparator);
      container_.resize(k_);
    }
    return container_;
  }

private:
  int32 k_;
  std::vector<int32> container_;
  const T *values_ = nullptr;

  bool compare_fun(int32 a, int32 b) const
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

/**
 * @brief Operates TopK operation with params
 * @param [in] row_size Size of row in data
 * @param [in] num_rows The number of rows in data
 * @param [in] data To be operated in
 * @param [in] k The top k predictions
 * @param [out] output_indexes Indexes of targets in the top k predictions
 * @param [out] output_values Values of targets in the top k predictions
 * @return N/A
 */
template <typename T>
void TopK(int32 row_size, int32 num_rows, const T *data, int32 k, int32 *output_indexes,
          T *output_values)
{
  TopContainer<T> topc(k, row_size);
  for (int row = 0; row < num_rows; ++row)
  {
    const T *values_row = data + row * row_size;
    topc.start_collecting(values_row);
    for (int32 c = 0; c < row_size; ++c)
    {
      topc.push(c);
    }

    // Prepare output buffers.
    int32 *indexes_row = output_indexes + row * k;
    T *output_row = output_values + row * k;
    // We always assume that the output is sorted.
    const auto &top_k = topc.sorted_result();
    std::copy(top_k.begin(), top_k.end(), indexes_row);
    std::transform(top_k.begin(), top_k.end(), output_row,
                   [values_row](const int32 loc) { return values_row[loc]; });
  }
}

} // namespace optimized_ops
} // namespace rt
} // namespace nnfw

#endif // __NNFW_RT_OPTIMIZED_OPS_TOPK_V2_H__
