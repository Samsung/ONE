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

#ifndef NNCC_COMBINE_TRANSPOSES_H
#define NNCC_COMBINE_TRANSPOSES_H

#include "pass/Pass.h"
#include "pass/PassData.h"

namespace nnc
{

/**
 * @brief This pass combines sequential transposes and removes identity transposes if
 * the combination results in an identity permutation.
 */
class CombineTransposes : public Pass
{
public:
  PassData run(PassData data) override;

  std::string getName() override { return "opt_combine_transposes"; };

private:
};

} // namespace nnc

#endif // NNCC_COMBINE_TRANSPOSES_H
