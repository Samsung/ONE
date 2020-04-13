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

#ifndef __MERGE_COMMAND_H__
#define __MERGE_COMMAND_H__

#include <cli/Command.h>

/**
 * @brief Takes .prototxt and .caffemodel filenames from ARGV
 * and fills the model with trained weights.
 * The resulting binary model with weights to be consumed by nnc is printed to StdOut
 * @return error code
 */
struct MergeCommand final : public cli::Command
{
  int run(int argc, const char *const *argv) const override;
};

#endif //__MERGE_COMMAND_H__
