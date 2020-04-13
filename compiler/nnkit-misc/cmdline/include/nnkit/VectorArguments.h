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

#ifndef __NNKIT_VECTOR_ARGUMENTS_H__
#define __NNKIT_VECTOR_ARGUMENTS_H__

#include <nnkit/CmdlineArguments.h>

#include <vector>
#include <string>

namespace nnkit
{

class VectorArguments final : public CmdlineArguments
{
public:
  uint32_t size(void) const override { return _args.size(); }
  const char *at(uint32_t nth) const override { return _args.at(nth).c_str(); }

public:
  VectorArguments &append(const std::string &arg);

private:
  std::vector<std::string> _args;
};

} // namespace nnkit

#endif // __NNKIT_VECTOR_ARGUMENTS_H__
