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

#ifndef __ENCO_SESSION_H__
#define __ENCO_SESSION_H__

#include "Code.h"

namespace enco
{

// TODO Rewrite this definition
using SessionID = uint32_t;

SessionID make_session(coco::Module *m, coco::Data *d);

SessionID session(const coco::Module *m);
SessionID session(const coco::Data *d);

coco::Module *module(const SessionID &);
coco::Data *data(const SessionID &);

static inline coco::Module *module(const coco::Data *d) { return module(session(d)); }
static inline coco::Data *data(const coco::Module *m) { return data(session(m)); }

// WARN This API is introduced just for backward compatibility
//      Do NOT use this anymore as it will be removed
Code *code(const SessionID &);

} // namespace enco

#endif // __ENCO_SESSION_H__
