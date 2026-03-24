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

#ifndef __NNKIT_SUPPORT_CAFFE_OUTPUT_BLOB_CONTEXT_H__
#define __NNKIT_SUPPORT_CAFFE_OUTPUT_BLOB_CONTEXT_H__

#include "nnkit/support/caffe/BlobContext.h"

#include <caffe/net.hpp>

namespace nnkit
{
namespace support
{
namespace caffe
{

template <typename DType> class OutputBlobContext final : public BlobContext<DType>
{
public:
  OutputBlobContext(::caffe::Net<DType> &net) : _net(net)
  {
    // DO NOTHING
  }

public:
  uint32_t size(void) const override { return _net.num_outputs(); }

  std::string name(uint32_t n) const override
  {
    return _net.blob_names().at(_net.output_blob_indices().at(n));
  }

  ::caffe::Blob<DType> *blob(uint32_t n) override { return _net.output_blobs().at(n); }

private:
  ::caffe::Net<DType> &_net;
};

} // namespace caffe
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_CAFFE_OUTPUT_BLOB_CONTEXT_H__
