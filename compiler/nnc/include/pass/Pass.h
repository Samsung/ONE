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

#ifndef NNCC_PASS_H
#define NNCC_PASS_H

#include <string>

#include "pass/PassData.h"

namespace nnc
{

/**
 * @brief this class represent an interface for all compiler passes like that frontend, backend etc
 */
class Pass
{
public:
  /**
   * @brief run compiler pass
   * @param data - data that pass is taken
   * @return data that can be passed to the next pass
   * @throw PassException object if errors occured
   */
  virtual PassData run(PassData data) = 0;

  /**
   * @brief clean compiler pass data
   */
  virtual void cleanup(){};

  virtual ~Pass() = default;

  virtual std::string getName() { return "pass"; }
};

} // namespace nnc

#endif // NNCC_PASS_H
