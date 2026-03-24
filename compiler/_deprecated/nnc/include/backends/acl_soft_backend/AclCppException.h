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

#ifndef _NNC_ACLCPPEXCEPTION_H_
#define _NNC_ACLCPPEXCEPTION_H_

#include <stdexcept>

namespace nnc
{

/**
 * @brief objects of this class are to be thrown from ACL C++ soft backend if errors are occurred.
 */
class AclCppException : public std::runtime_error
{
public:
  explicit AclCppException(const std::string &msg) : runtime_error(_prefix + msg) {}

private:
  static constexpr const char *_prefix = "ACL C++ soft backend error: ";
};

} // namespace nnc

#endif //_NNC_ACLCPPEXCEPTION_H_
