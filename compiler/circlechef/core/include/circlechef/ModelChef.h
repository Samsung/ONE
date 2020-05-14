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

#ifndef __MODEL_CHEF_H__
#define __MODEL_CHEF_H__

#include <circlechef.pb.h>

#include <memory>

namespace circlechef
{

class GeneratedModel final
{
public:
  struct Impl
  {
    virtual ~Impl() = default;

    virtual const char *base(void) const = 0;
    virtual size_t size(void) const = 0;
  };

public:
  GeneratedModel(std::unique_ptr<Impl> &&impl) : _impl{std::move(impl)}
  {
    // DO NOTHING
  }

public:
  const char *base(void) const { return _impl->base(); }
  size_t size(void) const { return _impl->size(); }

private:
  std::unique_ptr<Impl> _impl;
};

GeneratedModel cook(const ModelRecipe &model_recipe);

} // namespace circlechef

#endif // __MODEL_CHEF_H__
