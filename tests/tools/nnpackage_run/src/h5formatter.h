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

#ifndef __NNPACKAGE_RUN_H5FORMATTER_H__
#define __NNPACKAGE_RUN_H5FORMATTER_H__

#include <string>
#include <vector>

#include "types.h"
#include "allocation.h"

struct nnfw_session;

namespace nnpkg_run
{
class H5Formatter
{
public:
  H5Formatter(nnfw_session *sess) : session_(sess) {}
  std::vector<TensorShape> readTensorShapes(const std::string &filename);
  void loadInputs(const std::string &filename, std::vector<Allocation> &inputs);
  void dumpOutputs(const std::string &filename, std::vector<Allocation> &outputs);

private:
  nnfw_session *session_;
};
} // namespace nnpkg_run

#endif // __NNPACKAGE_RUN_H5FORMATTER_H__
