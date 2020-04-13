/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __PEPPER_ENV_H__
#define __PEPPER_ENV_H__

#include <string>

//
// KVStore: Key-Value Store Interface
//
namespace pepper // TODO Extract this section if necessary
{

enum class KVStoreTrait
{
  Queryable,
};

template <KVStoreTrait Trait> class KVStoreInterface;

template <> class KVStoreInterface<KVStoreTrait::Queryable>
{
public:
  KVStoreInterface() = default;

public:
  virtual ~KVStoreInterface() = default;

public: // Core interface (PLEASE PREFER TO THE BELOW HELPERS)
  //
  //  "query(k)" SHOULD
  //  - return a valid C-string if the key "k" exists in the store, or
  //  - return nullptr otherwise.
  //
  // DESIGN NOTE - Why "query" instead of "get"?
  //
  //  Let us consider the following class declarations as an example:
  //
  //  struct Base {
  //    virtual const char *get(const char *) const = 0;
  //    const char *get(const std::string &s) const { return nullptr; }
  //  };
  //
  //  struct Derived : public Base {
  //    const char *get(const char *) const final { return nullptr; }
  //  };
  //
  //  It is impossible to write the code of the following form:
  //
  //  Derived obj;
  //
  //  std::string s = ...;
  //  obj.get(s);
  //  ^^^^^^^^^^^
  //  error: no viable conversion from 'std::string' (aka 'basic_string<char>') to 'const char *'
  //
  //  Please refer to the concept of name hiding in C++ for more details.
  virtual const char *query(const char *k) const = 0;

public: // Derived helper methods
  const char *get(const std::string &k) const { return query(k.c_str()); }

  /**
   * NOTE
   *
   *  get(k, v) same as query(k) if the key "k" exists in the store.
   *  get(k, v) returns "v" otherwise
   */
  std::string get(const std::string &key, const std::string &default_value) const;
};

} // namespace pepper

//
// ProcessEnvironment
//
namespace pepper
{

struct ProcessEnvironment final : public KVStoreInterface<KVStoreTrait::Queryable>
{
  const char *query(const char *k) const final;
};

} // namespace pepper

#endif // __PEPPER_ENV_H__
