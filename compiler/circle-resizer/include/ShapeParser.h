/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_RESIZER_SHAPE_PARSER_H__
#define __CIRCLE_RESIZER_SHAPE_PARSER_H__

#include "Shape.h"

#include <string>
#include <vector>

namespace circle_resizer
{

/**
 * @brief Parse shapes from string representation to Shapes object.
 *
 * The single shape is represented by comma-separated integers inside squared brackets.
 * If there is more than one shape, they are separated by commas.
 * An example for single shape: [1,2,3], an example for many shapes: [1,2,3],[4,5].
 *
 * Exceptions:
 * std::invalid_argument if the parsing failed.
 */
std::vector<Shape> parse_shapes(const std::string &shapes);

} // namespace circle_resizer

#endif // __CIRCLE_RESIZER_SHAPE_PARSER_H__
