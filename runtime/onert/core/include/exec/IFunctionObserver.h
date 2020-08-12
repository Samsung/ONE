/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_EXEC_IFUNCTION_OBSERVER_H__
#define __ONERT_EXEC_IFUNCTION_OBSERVER_H__

#include "ir/Index.h"
#include "ir/OpSequence.h"
#include "exec/IFunction.h"

namespace onert
{
namespace exec
{

class IFunctionObserver
{
public:
  /// @brief Invoked after IFunction::run()
  virtual void handleEnd(const ir::Operation *op, const exec::IFunction &func) = 0;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_IFUNCTION_OBSERVER_H__
