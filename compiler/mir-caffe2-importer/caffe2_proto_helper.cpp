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

#include "caffe2_proto_helper.h"

namespace mir_caffe2
{

const ::caffe2::Argument &findArgumentByName(RepArgument args, const std::string &name)
{
  for (auto &arg : args)
    if (arg.name() == name)
      return arg;
  throw std::runtime_error("Can't find argument with name: " + name);
}

const bool hasArgument(RepArgument args, const std::string &name)
{
  for (auto &arg : args)
    if (arg.name() == name)
      return true;
  return false;
}

int getSingleArgument(const ::caffe2::OperatorDef &op, const std::string &argument_name,
                      const int default_value)
{
  if (hasArgument(op.arg(), argument_name))
    return static_cast<int>(findArgumentByName(op.arg(), argument_name).i());
  return default_value;
}

float getSingleArgument(const ::caffe2::OperatorDef &op, const std::string &argument_name,
                        const float default_value)
{
  if (hasArgument(op.arg(), argument_name))
    return findArgumentByName(op.arg(), argument_name).f();
  return default_value;
}

std::string getSingleArgument(const ::caffe2::OperatorDef &op, const std::string &argument_name,
                              const std::string &default_value)
{
  if (hasArgument(op.arg(), argument_name))
    return findArgumentByName(op.arg(), argument_name).s();
  return default_value;
}

} // namespace mir_caffe2
