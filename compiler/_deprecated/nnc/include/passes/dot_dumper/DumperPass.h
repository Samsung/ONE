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

#ifndef NNCC_DUMPERPASS_H
#define NNCC_DUMPERPASS_H

#include "pass/Pass.h"

namespace nnc
{

/**
 * @brief Dumps the graph to a dot file named %number%.dot
 * where %number% is how many times the graph was dumped.
 */
class DumperPass : public Pass
{
public:
  explicit DumperPass(std::string s) : _file_name(std::move(s)) {}

  PassData run(PassData data) override;

private:
  std::string _file_name;
  static int _counter;
};

} // namespace nnc
#endif // NNCC_DUMPERPASS_H
