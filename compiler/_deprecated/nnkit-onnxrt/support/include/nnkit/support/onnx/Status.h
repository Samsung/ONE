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

#ifndef __NNKIT_SUPPORT_ONNX_STATUS_H__
#define __NNKIT_SUPPORT_ONNX_STATUS_H__

#include <onnxruntime_c_api.h>

#include <string>
#include <stdexcept>

namespace nnkit
{
namespace support
{
namespace onnx
{

class Status
{
public:
  Status() : _status(nullptr)
  {
    // DO NOTHING
  }

  Status(OrtStatus *status) : _status(status)
  {
    // DO NOTHING
  }

  ~Status()
  {
    if (_status)
    {
      OrtReleaseStatus(_status);
    }
  }

  Status &operator=(OrtStatus *status)
  {
    if (_status)
    {
      OrtReleaseStatus(_status);
    }
    _status = status;
    return *this;
  }

  bool isError(void) { return (_status != nullptr); }

  void throwOnError(void)
  {
    if (_status)
    {
      const char *msg = OrtGetErrorMessage(_status);
      std::string err{msg};
      OrtReleaseStatus(_status);
      throw std::runtime_error{err};
    }
  }

private:
  // NOTE nullptr for OrtStatus* indicates success
  OrtStatus *_status;
};

} // namespace onnx
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_ONNX_STATUS_H__
