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

#ifndef __ONERT_EXEC_SHAPE_CONVERTER_H__
#define __ONERT_EXEC_SHAPE_CONVERTER_H__

#include <ir/Layout.h>
#include <ir/Shape.h>

namespace onert
{
namespace exec
{

/**
 * @brief Converts shape when its rank is 4
 *
 * @return ir::Shape Return a shape based on dst_layout. If rank is not 4, input shape is
 *         returned without conversion.
 */
ir::Shape convertShape(const ir::Shape &shape, ir::Layout src_layout, ir::Layout dst_layout);

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_SHAPE_CONVERTER_H__
