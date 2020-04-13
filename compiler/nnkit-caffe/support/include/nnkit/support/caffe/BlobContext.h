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

#ifndef __NNKIT_SUPPORT_CAFFE_BLOB_CONTEXT_H__
#define __NNKIT_SUPPORT_CAFFE_BLOB_CONTEXT_H__

#include <caffe/blob.hpp>

namespace nnkit
{
namespace support
{
namespace caffe
{

template <typename DType> struct BlobContext
{
  virtual ~BlobContext() = default;

  virtual uint32_t size(void) const = 0;

  virtual std::string name(uint32_t n) const = 0;
  virtual ::caffe::Blob<DType> *blob(uint32_t n) = 0;

  DType *region(uint32_t n) { return blob(n)->mutable_cpu_data(); }
};

} // namespace caffe
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_CAFFE_BLOB_CONTEXT_H__
