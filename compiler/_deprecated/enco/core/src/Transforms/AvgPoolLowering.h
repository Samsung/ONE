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

#ifndef __REWRITE_H__
#define __REWRITE_H__

#include "Code.h"
#include "Pass.h"

namespace enco
{

/**
 * @brief Rewrite NN API-incompatible average pooling
 */
void lower_avgpool(enco::Code *);

struct AvgPoolLoweringPass final : public Pass
{
  PASS_CTOR(AvgPoolLoweringPass)
  {
    // DO NOTHING
  }

  void run(const SessionID &sess) const override { lower_avgpool(code(sess)); }
};

} // namespace enco

#endif // __REWRITE_H__
