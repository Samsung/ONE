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
using cpu::ops::getPaddingType;
using cpu::ops::getShape;
using cpu::ops::getNumberOfDimensions;
using cpu::ops::getNumberOfElements;
using cpu::ops::getSizeOfDimension;

/**
 * @brief backpropagate acitvation
 *
 *             -- forward direction -->
 *
 *   [ current layer ]   ----   [ next layer ]
 *   [ op    |  act  ]
 *
 *             <-- backward direction --
 *
 * @param activation      activation of current layer
 * @param output          forward direction's output of current layer
 * @param input_backprop  backward direction's output of next layer
 *                        In other words, incoming gradient to current layer
 * @param output_backprop backward direction's output of activation,
 *                        In other words, outcoming gradient of current layer's acitvation
 *                        If activation is NONE, this param can be nullptr
 * @return tensor that holds backpropagate result of activation
 *         If activation is NONE, just return input_backprop
 */
const IPortableTensor *backpropActivation(const ir::Activation &activation,
                                          const IPortableTensor *output,
                                          const IPortableTensor *input_backprop,
                                          IPortableTensor *output_backprop);

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_OPS_OPERATION_UTILS_H__
