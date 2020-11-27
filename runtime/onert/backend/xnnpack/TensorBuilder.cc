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

#include "TensorBuilder.h"

#include <util/logging.h>

#include <cassert>

namespace onert
{
namespace backend
{
namespace xnnpack
{

TensorBuilder::TensorBuilder(const std::shared_ptr<cpu_common::TensorRegistry> &tensor_reg)
  : _tensor_reg{tensor_reg}, _static_tensor_mgr{
                               new cpu_common::StaticTensorManager(_tensor_reg, nullptr)}
{
  /* empty */
}

void TensorBuilder::registerTensorInfo(const ir::OperandIndex &ind, const ir::OperandInfo &info,
                                       ir::Layout layout)
{
  assert(!info.isDynamic());
  _tensor_info_map.emplace(ind, info);

  // TODO: supports not only one layout as NHWC
  assert(layout == ir::Layout::NHWC);
  _tensor_layout_map.insert({ind, layout});
}

void TensorBuilder::notifyFirstUse(const ir::OperandIndex &ind)
{
  _lifetime_seq.emplace_back(UsesType::FIRST, ind);
}

void TensorBuilder::notifyLastUse(const ir::OperandIndex &ind)
{
  _lifetime_seq.emplace_back(UsesType::LAST, ind);
}

bool TensorBuilder::isRegistered(const ir::OperandIndex &ind) const
{
  return _tensor_info_map.find(ind) != _tensor_info_map.end();
}

void TensorBuilder::prepare()
{
  for (auto &entry : _tensor_info_map)
  {
    auto ind = entry.first;
    const auto &info = entry.second;
    const auto &layout = _tensor_layout_map[ind];
    _static_tensor_mgr->buildTensor(ind, info, layout, info.isConstant());
  }
}

void TensorBuilder::allocate()
{
  _static_tensor_mgr->allocateConsts();

  for (const auto &entry : _lifetime_seq)
  {
    const auto &ind = entry.second;
    if (entry.first == UsesType::FIRST)
    {
      assert(_tensor_info_map.find(ind) != _tensor_info_map.end());
      const auto tensor_info = _tensor_info_map.at(ind);

      assert(!_tensor_reg->getNativeTensor(ind)->is_dynamic());
      const auto size = tensor_info.total_size();
      _static_tensor_mgr->claimPlan(ind, size);
    }
    else
    {
      assert(!_tensor_reg->getNativeTensor(ind)->is_dynamic());
      _static_tensor_mgr->releasePlan(ind);
    }
  }
  _static_tensor_mgr->allocateNonconsts();
}

void TensorBuilder::postFunctionPrepare()
{
  // TODO: Release weights
}

} // namespace xnnpack
} // namespace backend
} // namespace onert
