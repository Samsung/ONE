/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef CIRCLE_EXECUTION_PLAN_ISRCRATCHPAD_HELPER_H
#define CIRCLE_EXECUTION_PLAN_ISRCRATCHPAD_HELPER_H

namespace circle_planner
{

class IScratchpadHelper
{
public:
  virtual uint32_t ComputeScratchpadSizeConv2d(const luci::CircleConv2D *conv) = 0;

  virtual ~IScratchpadHelper() = default;
};

} // namespace circle_planner

#endif // CIRCLE_EXECUTION_PLAN_ISRCRATCHPAD_HELPER_H
