/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __COMMON_H__
#define __COMMON_H__

#include <cstdint>
#include <string>

/**
 * @brief Construct HDF5-compatible dataset name from a given string
 *
 * When someone attempts to access 'A/B/C' dataset, HDF5 tries to open
 * dataset C in group B in top-level group A, which menas that dataset
 * names SHOULD NOT contain '/' in it.
 *
 * This mangle function replaces all the occurence of '/' in a given
 * string with '_' to construct HDF5-compatible dataset name.
 */
std::string mangle(const std::string &);

#if 0
Let us assume that a tensor context includes N + 1 tensors.

Then, HDF5 export will generate a HDF5 file whose structure is given as follows:
[value group]/
  [file 0] <- A dataset that contains the value of 1st (=0) tensor
  [file 1]
  ...
  [file N]
[name group]/
  [file 0] <- An attribute that contains the name of 1st (=0) tensor
  [file 1]
  ...
  [file N]
#endif

/// @brief Return the name of "value group"
std::string value_grpname(void);
/// @brief Return the name of n-th tensor dataset
std::string value_filename(uint32_t n);

/// @brief Return the name of "name group"
std::string name_grpname(void);
/// @brief Return the name of n-th tensor attribute
std::string name_filename(uint32_t n);

#endif // __COMMON_H__
