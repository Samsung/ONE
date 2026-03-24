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

#ifndef __NNKIT_SUPPORT_CAFFE_BACKEND_H__
#define __NNKIT_SUPPORT_CAFFE_BACKEND_H__

#include "nnkit/support/caffe/InputBlobContext.h"
#include "nnkit/support/caffe/OutputBlobContext.h"
#include "nnkit/support/caffe/TensorContext.h"

#include <nnkit/Backend.h>

#include <caffe/net.hpp>

#include <memory>
#include <functional>

namespace nnkit
{
namespace support
{
namespace caffe
{

template <typename DType> class Backend final : public nnkit::Backend
{
public:
  Backend(std::unique_ptr<::caffe::Net<DType>> &&net) : _net{std::move(net)}
  {
    // DO NOTHING
  }

public:
  void prepare(const std::function<void(nnkit::TensorContext &)> &f) override
  {
    InputBlobContext<DType> blobs(*_net);
    TensorContext<DType> tensors(blobs);
    f(tensors);
  }

public:
  void run(void) override { _net->Forward(); }

public:
  void teardown(const std::function<void(nnkit::TensorContext &)> &f) override
  {
    OutputBlobContext<DType> blobs(*_net);
    TensorContext<DType> tensors(blobs);
    f(tensors);
  }

private:
  std::unique_ptr<::caffe::Net<DType>> _net;
};

} // namespace caffe
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_CAFFE_BACKEND_H__
