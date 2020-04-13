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

#ifndef __PP_ENCLOSED_DOCUMENT_H__
#define __PP_ENCLOSED_DOCUMENT_H__

#include "pp/LinearDocument.h"
#include "pp/MultiLineText.h"

namespace pp
{

class EnclosedDocument final : public MultiLineText
{
public:
  EnclosedDocument() : _front{}, _back{LinearDocument::Direction::Reverse}
  {
    // DO NOTHING
  }

public:
  LinearDocument &front(void) { return _front; }
  const LinearDocument &front(void) const { return _front; }

public:
  LinearDocument &back(void) { return _back; }
  const LinearDocument &back(void) const { return _back; }

public:
  uint32_t lines(void) const override;
  const std::string &line(uint32_t n) const override;

private:
  LinearDocument _front;
  LinearDocument _back;
};

} // namespace pp

#endif // __PP_ENCLOSED_DOCUMENT_H__
