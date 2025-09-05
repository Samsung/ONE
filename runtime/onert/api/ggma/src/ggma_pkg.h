/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __API_GGMA_PKG_H__
#define __API_GGMA_PKG_H__

#include "ggma_config.h"

#include <string>

struct ggma_pkg
{
public:
  ggma_pkg(const char *path);
  ~ggma_pkg() = default;
  std::string path() const { return _path; }
  ggma::GGMAConfig load_config() const;

private:
  std::string _path;
};

#endif // __API_GGMA_PKG_H__
