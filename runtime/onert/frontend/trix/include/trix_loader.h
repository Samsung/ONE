/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __TRIX_TRIX_LOADER_H__
#define __TRIX_TRIX_LOADER_H__

#include "trix_loader_base.h"

namespace onert
{
namespace trix_loader
{
class TrixLoader final : public TrixLoaderBase
{
public:
  explicit TrixLoader(std::unique_ptr<ir::Subgraphs> &subgs) : TrixLoaderBase(subgs) {}

protected:
  bool loadModel() override;
};

std::unique_ptr<ir::Subgraphs> loadModel(const std::string &filename);
} // namespace trix_loader
} // namespace onert

#endif // __TRIX_TRIX_LOADER_H__
