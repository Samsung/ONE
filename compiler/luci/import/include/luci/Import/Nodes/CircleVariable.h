/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_IMPORT_OP_CIRCLE_VARIABLE_H__
#define __LUCI_IMPORT_OP_CIRCLE_VARIABLE_H__

#include "luci/Import/GraphBuilderContext.h"

#include <luci/IR/Nodes/CircleVariable.h>

/*
 * @note  Circle does not have node for variable tensor
 *        Methods here provide helper that creates CircleVariable from
 *        Tensor having is_variable true value.
 */

namespace luci
{

CircleVariable *create_circlevariable(GraphBuilderContext *context, int32_t tensor_index);

} // namespace luci

#endif // __LUCI_IMPORT_OP_CIRCLE_VARIABLE_H__
