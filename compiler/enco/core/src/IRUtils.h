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

#ifndef __ENCO_IR_UTILS_H__
#define __ENCO_IR_UTILS_H__

#include <coco/IR.h>

#include <vector>

namespace enco
{

/**
 * @brief Replace all the "USE" of 'from' with 'into'
 *
 * NOTE subst(from, into) WILL NOT update 'DEF'
 */
void subst(coco::Object *from, coco::Object *into);

/**
 * @brief Return instructions in execution order
 */
std::vector<coco::Instr *> instr_sequence(coco::Module *m);

} // namespace enco

#endif // __ENCO_IR_UTILS_H__
