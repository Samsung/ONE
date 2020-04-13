/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ENCODE_COMMAND_H__
#define __ENCODE_COMMAND_H__

#include <cli/Command.h>

namespace tfkit
{

struct EncodeCommand final : public cli::Command
{
  int run(int argc, const char *const *argv) const override;
};

} // namespace tfkit

#endif // __ENCODE_COMMAND_H__
