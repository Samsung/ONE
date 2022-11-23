/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_RUN_FORMATTER_H__
#define __ONERT_RUN_FORMATTER_H__

#include <string>
#include <vector>

#include "types.h"
#include "allocation.h"

struct nnfw_session;

namespace onert_run
{
class Formatter
{
public:
  virtual ~Formatter() = default;
  Formatter(nnfw_session *sess) : session_(sess) {}
  virtual void loadInputs(const std::string &filename, std::vector<Allocation> &inputs) = 0;
  virtual void dumpOutputs(const std::string &filename, std::vector<Allocation> &outputs) = 0;
  virtual std::vector<TensorShape> readTensorShapes(const std::string &filename)
  {
    return std::vector<TensorShape>();
  };

protected:
  nnfw_session *session_;
};
} // namespace onert_run

#endif // __ONERT_RUN_FORMATTER_H__
