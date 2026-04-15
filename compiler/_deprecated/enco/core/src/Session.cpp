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

#include "Session.h"

#include <map>
#include <memory>

using std::make_unique;

namespace
{

std::map<enco::SessionID, std::unique_ptr<enco::Code>> sess_to_code;
std::map<const coco::Module *, enco::SessionID> module_to_sess;
std::map<const coco::Data *, enco::SessionID> data_to_sess;

} // namespace

namespace enco
{

SessionID make_session(coco::Module *m, coco::Data *d)
{
  static uint32_t sess = 0;
  SessionID curr{sess++};

  sess_to_code[curr] = make_unique<Code>(m, d);
  module_to_sess[m] = curr;
  data_to_sess[d] = curr;

  return curr;
}

SessionID session(const coco::Module *m) { return module_to_sess.at(m); }
SessionID session(const coco::Data *d) { return data_to_sess.at(d); }

coco::Module *module(const SessionID &sess) { return sess_to_code.at(sess)->module(); }
coco::Data *data(const SessionID &sess) { return sess_to_code.at(sess)->data(); }

Code *code(const SessionID &sess) { return sess_to_code.at(sess).get(); }

} // namespace enco
