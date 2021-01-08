/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_CONVERT_NCHW_TO_NHWC_PASS_H__
#define __LUCI_CONVERT_NCHW_TO_NHWC_PASS_H__

#include <logo/Pass.h>

namespace luci
{

/**
 * @brief   Class to convert NCHW Ops to NHWC
 *
 * @details Find operators that use NCHW layout and make them use NHWC.
 *          Strictly speaking, it is impossible to distinguish whether
 *          an operator is using NCHW or NHWC without programmers' annotations.
 *          But we guess the data layout of each operator as much as possible
 *          based on the assumptions described in the comments.
 *          Note that this Pass does not change the execution result even
 *          for the false-positive cases.
 */
struct ConvertNCHWToNHWCPass final : public logo::Pass
{
public:
  ConvertNCHWToNHWCPass(bool preserve_input = false, bool preserve_output = false)
    : _preserve_input(preserve_input), _preserve_output(preserve_output)
  {
    // Do nothing
  }

  virtual ~ConvertNCHWToNHWCPass() = default;

  const char *name(void) const final { return "luci::ConvertNCHWToNHWCPass"; }

  bool run(loco::Graph *g) final;

private:
  bool _preserve_input = false;
  bool _preserve_output = false;
};

} // namespace luci

#endif // __LUCI_CONVERT_NCHW_TO_NHWC_PASS_H__
