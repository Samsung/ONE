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

#ifndef __LOCO_IR_CANONICAL_DIALECT_H__
#define __LOCO_IR_CANONICAL_DIALECT_H__

#include "loco/IR/Dialect.h"

namespace loco
{

/**
 * @brief A singleton for Canonical Dialect
 *
 * CanonicalDialect serves as an in-memory unique identifier.
 */
class CanonicalDialect final : public Dialect
{
private:
  CanonicalDialect();

public:
  CanonicalDialect(const CanonicalDialect &) = delete;
  CanonicalDialect(CanonicalDialect &&) = delete;

public:
  static Dialect *get(void);
};

} // namespace loco

#endif // __LOCO_IR_CANONICAL_DIALECT_H__
