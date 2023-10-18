/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRAIN_OPS_OPERATION_UTILS_H__
#define __ONERT_BACKEND_TRAIN_OPS_OPERATION_UTILS_H__

#include <ops/OperationUtils.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

using OperandType = onert::ir::DataType;
using cpu::ops::getBuffer;
using cpu::ops::getShape;
using cpu::ops::getNumberOfDimensions;
using cpu::ops::getNumberOfElements;
using cpu::ops::getSizeOfDimension;

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_OPS_OPERATION_UTILS_H__
