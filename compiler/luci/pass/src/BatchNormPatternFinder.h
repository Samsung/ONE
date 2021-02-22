/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_PASS_BATCH_NORM_PATTERN_FINDER_H__
#define __LUCI_PASS_BATCH_NORM_PATTERN_FINDER_H__

#include <luci/IR/CircleNodes.h>

namespace luci
{

/**
 * @brief Find Mul-Add pattern and return Mul and beta as BatchNorm
 */
bool is_batchnorm_add(const luci::CircleAdd *add, luci::CircleMul *&mul, luci::CircleConst *&beta);

/**
 * @brief Find Mul-Add pattern
 */
bool is_batchnorm_add(const luci::CircleAdd *add);

/**
 * @brief Find Const-Mul pattern and return Node and gamma as BatchNorm
 */
bool is_batchnorm_mul(const luci::CircleMul *mul, luci::CircleNode *&pred_node,
                      luci::CircleConst *&gamma);

} // namespace luci

#endif // __LUCI_PASS_BATCH_NORM_PATTERN_FINDER_H__
