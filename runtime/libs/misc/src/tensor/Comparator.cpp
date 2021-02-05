/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "misc/tensor/Comparator.h"
#include "misc/tensor/Zipper.h"

#include "misc/fp32.h"

namespace nnfw
{
namespace misc
{
namespace tensor
{

template <typename T>
std::vector<Diff<T>> Comparator<T>::compare(const Shape &shape, const Reader<T> &expected,
                                            const Reader<T> &obtained, Observer *observer) const
{
  std::vector<Diff<T>> res;

  zip(shape, expected, obtained) << [&](const Index &index, T expected_value, T obtained_value) {
    if (!_compare_fn(expected_value, obtained_value))
    {
      res.emplace_back(index, expected_value, obtained_value);
    }

    // Update max_diff_index, if necessary
    if (observer != nullptr)
    {
      observer->notify(index, expected_value, obtained_value);
    }
  };

  return res;
}

template class Comparator<float>;

} // namespace tensor
} // namespace misc
} // namespace nnfw
