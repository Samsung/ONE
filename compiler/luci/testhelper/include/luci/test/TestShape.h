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

#ifndef __LUCI_TESTHELPER_TEST_SHAPE_H__
#define __LUCI_TESTHELPER_TEST_SHAPE_H__

#include <luci/IR/CircleNode.h>

#include <initializer_list>

namespace luci
{
namespace test
{

using ShapeU32 = std::initializer_list<uint32_t>;
using ShapeI32 = std::initializer_list<int32_t>;

void set_shape_vector(loco::TensorShape *shape, const ShapeU32 &values);
void set_shape_vector(luci::CircleConst *const_node, const ShapeI32 &values);

uint32_t num_elements(const ShapeU32 shape);

} // namespace test
} // namespace luci

#endif // __LUCI_TESTHELPER_TEST_SHAPE_H__
