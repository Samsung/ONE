/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef _FME_APPLY_FM_EQUALIZER_H__
#define _FME_APPLY_FM_EQUALIZER_H__

#include "EqualizePattern.h"

#include <loco.h>

namespace fme_apply
{

class FMEqualizer final
{
public:
  void equalize(loco::Graph *g, std::vector<EqualizePattern> &p);
};

} // namespace fme_apply

#endif // _FME_APPLY_FM_EQUALIZER_H__
