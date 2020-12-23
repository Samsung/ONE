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

#ifndef __LOCO_IR_DIALECT_H__
#define __LOCO_IR_DIALECT_H__

#include "loco/include/loco/IR/DialectService.h"

#include <map>
#include <memory>
#include <typeindex>
#include <typeinfo>

namespace loco
{

  /**
 * @brief Dialect interface
 *
 * Each dialect implementation is expected to have static "get" method
 * which returns "const Dialect *" value.
 */
  class Dialect
  {
  public:
    virtual ~Dialect() = default;

  protected:
    template <typename ConcreteService>
    void service(std::unique_ptr<ConcreteService> &&s)
    {
      _services[typeid(ConcreteService)] = std::move(s);
    }

  public:
    template <typename ConcreteService>
    ConcreteService *service(void) const
    {
      auto it = _services.find(typeid(ConcreteService));

      if (it == _services.end())
      {
        return nullptr;
      }

      return dynamic_cast<ConcreteService *>(it->second.get());
    }

  private:
    std::map<std::type_index, std::unique_ptr<DialectService>> _services;
  };

} // namespace loco

#endif // __LOCO_IR_DIALECT_H__
