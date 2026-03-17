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

#include "AsmCode.h"

namespace enco
{

void AsmCode::dump(std::ostream &os) const
{
  os << ".section .rodata" << std::endl;
  os << ".global " << _varname << std::endl;
  // Please refer to https://www.sourceware.org/binutils/docs/as/Type.html#Type for details
  os << ".type " << _varname << ", STT_OBJECT" << std::endl;
  os << ".align " << 4 << std::endl;
  os << _varname << ":" << std::endl;
  os << ".incbin " << '"' << _filename << '"' << std::endl;
}

} // namespace enco
