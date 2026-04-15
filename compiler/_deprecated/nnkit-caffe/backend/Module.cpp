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

#include "nnkit/support/caffe/Backend.h"

#include <nnkit/CmdlineArguments.h>

#include <memory>

extern "C" std::unique_ptr<nnkit::Backend> make_backend(const nnkit::CmdlineArguments &args)
{
  using std::make_unique;

  auto net = make_unique<::caffe::Net<float>>(args.at(0), caffe::TEST);

  if (args.size() > 1)
  {
    net->CopyTrainedLayersFrom(args.at(1));
  }

  return make_unique<::nnkit::support::caffe::Backend<float>>(std::move(net));
}
