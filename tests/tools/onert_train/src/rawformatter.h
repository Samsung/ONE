/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_TRAIN_RAWFORMATTER_H__
#define __ONERT_TRAIN_RAWFORMATTER_H__

#include "allocation.h"
#include "formatter.h"
#include "types.h"

#include <string>
#include <vector>

struct nnfw_session;

namespace onert_train
{
class RawFormatter : public Formatter
{
public:
  RawFormatter(nnfw_session *sess) : Formatter(sess) {}
  void loadInputs(const std::string &filename, std::vector<Allocation> &inputs) override;
  void dumpOutputs(const std::string &filename, std::vector<Allocation> &outputs) override;
};
} // namespace onert_train

#endif // __ONERT_TRAIN_RAWFORMATTER_H__
