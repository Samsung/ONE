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

#ifndef NNCC_INTERPRETERPASS_H
#define NNCC_INTERPRETERPASS_H

#include "mir/Graph.h"

#include <string>

namespace nnc
{

class InterpreterBackend final
{
public:
  InterpreterBackend(std::string input_dir, std::string output_dir);

  void run(mir::Graph *data);

private:
  std::string _input_dir;
  std::string _output_dir;
};

} // namespace nnc

#endif // NNCC_INTERPRETERPASS_H
