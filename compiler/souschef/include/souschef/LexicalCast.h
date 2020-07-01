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

/**
 * @brief This file provides string <-> number cast helpers
 */
#ifndef __SOUSCHEF_LEXICAL_CAST_H__
#define __SOUSCHEF_LEXICAL_CAST_H__

#include <string>

namespace souschef
{

/**
 * @brief Return a numeric value that corresponds to a given string
 *
 * @note This function will throw an exception on casting failure
 */
template <typename Number> Number to_number(const std::string &s);

} // namespace souschef

#endif // __SOUSCHEF_LEXICAL_CAST_H__
