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

#ifndef __ONERT_BACKEND_CONTROLFLOW_KERNEL_PERMUTELAYER_H__
#define __ONERT_BACKEND_CONTROLFLOW_KERNEL_PERMUTELAYER_H__

#include <exec/IPermuteFunction.h>

namespace onert
{
namespace backend
{
namespace controlflow
{
namespace kernel
{

class PermuteLayer : public onert::exec::IPermuteFunction
{
public:
  PermuteLayer(const std::vector<std::shared_ptr<onert::backend::ITensor>> &src_tensors,
               const std::vector<std::shared_ptr<onert::backend::ITensor>> &dst_tensors,
               std::vector<size_t> ranks)
  {
    assert(src_tensors.size() == dst_tensors.size());
    assert(src_tensors.size() == ranks.size());
    _src_tensors = src_tensors;
    _dst_tensors = dst_tensors;
    _ranks = ranks;
  }

  void optimize() override
  {
    // Remove copying of tensor as nullptr
    auto src_it = _src_tensors.begin();
    auto dst_it = _dst_tensors.begin();
    auto rank_it = _ranks.begin();
    while (src_it != _src_tensors.end())
    {
      if ((*src_it == *dst_it) || (*src_it == nullptr || *dst_it == nullptr))
      {
        src_it = _src_tensors.erase(src_it);
        dst_it = _dst_tensors.erase(dst_it);
        rank_it = _ranks.erase(rank_it);
      }
      else
      {
        ++src_it;
        ++dst_it;
        ++rank_it;
      }
    }
  }
};

} // namespace kernel
} // namespace controlflow
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CONTROLFLOW_KERNEL_PERMUTELAYER_H__
