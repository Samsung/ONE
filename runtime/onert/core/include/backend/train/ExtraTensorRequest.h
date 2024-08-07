/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_BACKEND_EXTRA_TENSOR_REQUEST_H__
#define __ONERT_BACKEND_EXTRA_TENSOR_REQUEST_H__

#include "backend/train/ExtraTensor.h"

namespace onert
{
namespace backend
{
namespace train
{

enum class ExtraTensorLifeTime
{
  BACKWARD,            // alive during backward()
  FORWARD_TO_BACKWARD, // alive from forward to backward()
};
class ExtraTensorRequest
{

public:
  ExtraTensorRequest(ir::OperandInfo info, ExtraTensorLifeTime lt, ExtraTensor **addr)
    : _info(info), _lifetime(lt), _address(addr)
  {
  }

  static ExtraTensorRequest createLike(const IPortableTensor *origin, ExtraTensor **addr)
  {
    assert(origin != nullptr);
    assert(addr != nullptr);

    return ExtraTensorRequest(origin->get_info(), ExtraTensorLifeTime::BACKWARD, addr);
  }

public:
  const ir::OperandInfo &info() const { return _info; }
  ExtraTensorLifeTime lifetime() const { return _lifetime; }

  void update_address(ExtraTensor *tensor) { *_address = tensor; }

private:
  ir::OperandInfo _info;
  ExtraTensorLifeTime _lifetime;
  backend::train::ExtraTensor **const _address;
};

using ExtraTensorRequests = std::vector<ExtraTensorRequest>;

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_EXTRA_TENSOR_REQUEST_H__
