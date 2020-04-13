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

#ifndef _NNCC_ARTIFACT_INDENT_H_
#define _NNCC_ARTIFACT_INDENT_H_

#include <string>
#include <ostream>

namespace nnc
{

/**
 * @brief Used by code and declaration generators to indent generated text.
 */
class ArtifactIndent
{
public:
  ArtifactIndent() : _level(0), _step(2) {}

  ArtifactIndent &operator++()
  {
    _level += _step;
    return *this;
  }

  ArtifactIndent &operator--()
  {
    _level -= _step;
    return *this;
  }

  int level() const { return _level; }

private:
  int _level;
  int _step;
};

inline std::ostream &operator<<(std::ostream &out, const ArtifactIndent &ind)
{
  if (ind.level() > 0)
    out << std::string(ind.level(), ' ');

  return out;
}

} // namespace nnc

#endif //_NNCC_ARTIFACT_INDENT_H_
