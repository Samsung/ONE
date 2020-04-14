/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_COMPILER_OPERAND_CONTEXT_H__
#define __ONERT_COMPILER_OPERAND_CONTEXT_H__

#include "backend/ITensor.h"
#include "ir/OperandIndexMap.h"
#include <unordered_map>
#include <memory>

namespace onert
{
namespace compiler
{

class OperandContext
{
public:
  OperandContext &set(const ir::OperandIndex &ind, const std::shared_ptr<backend::ITensor> &tensor);

public:
  bool exist(const ir::OperandIndex &ind) const { return _tensors.find(ind) != _tensors.end(); }

public:
  std::shared_ptr<backend::ITensor> at(const ir::OperandIndex &ind) const
  {
    return _tensors.at(ind);
  }

  std::shared_ptr<backend::ITensor> &at(const ir::OperandIndex &ind) { return _tensors.at(ind); }

  void iterate(const std::function<void(const ir::OperandIndex &, backend::ITensor &)> &fn);

private:
  ir::OperandIndexMap<std::shared_ptr<backend::ITensor>> _tensors;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_OPERAND_CONTEXT_H__
