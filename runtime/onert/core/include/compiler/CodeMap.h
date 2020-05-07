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

#ifndef __ONERT_COMPILER_CODE_MAP_H__
#define __ONERT_COMPILER_CODE_MAP_H__

#include "backend/IDynamicTensorManager.h"
#include "backend/ITensorRegistry.h"
#include "ir/OperandInfo.h"
#include <memory>
#include <unordered_map>

namespace onert
{
namespace compiler
{

struct CodeAndInfo
{
  const ir::OpSequence *op_seq;
  const ir::operation::LowerInfo *lower_info;
  std::unique_ptr<exec::FunctionSequence> fn_seq;

  CodeAndInfo(const ir::OpSequence *op_seq, const ir::operation::LowerInfo *lower_info,
              std::unique_ptr<exec::FunctionSequence> &&fn_seq)
      : op_seq{op_seq}, lower_info{lower_info}, fn_seq{std::move(fn_seq)}
  {
  }

  virtual ~CodeAndInfo() = default;
};

struct CodeAndInfoForStaticTensor : public CodeAndInfo
{
  CodeAndInfoForStaticTensor(const ir::OpSequence *op_seq,
                             const ir::operation::LowerInfo *lower_info,
                             std::unique_ptr<exec::FunctionSequence> &&fn_seq)
      : CodeAndInfo(op_seq, lower_info, std::move(fn_seq))
  {
  }
};

struct CodeAndInfoForDynamicTensor : public CodeAndInfo
{
  /// @brief dynamic tensor_manager is used to allocate memory during shape inference at execution
  backend::IDynamicTensorManager *dynamic_tensor_manager;
  /// @brief tensor_registry is used to access tensor to infer shape and allocate memory
  std::shared_ptr<backend::ITensorRegistry> tensor_registry;

  CodeAndInfoForDynamicTensor(const ir::OpSequence *op_seq,
                              const ir::operation::LowerInfo *lower_info,
                              std::unique_ptr<exec::FunctionSequence> &&fn_seq,
                              backend::IDynamicTensorManager *dynamic_tensor_manager,
                              std::shared_ptr<backend::ITensorRegistry> &tensor_registry)
      : CodeAndInfo(op_seq, lower_info, std::move(fn_seq)),
        dynamic_tensor_manager(dynamic_tensor_manager), tensor_registry(tensor_registry)
  {
  }
};

using CodeMap = std::unordered_map<ir::OpSequenceIndex, std::unique_ptr<CodeAndInfo>>;

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_CODE_MAP_H__
