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

#ifndef __CONVERSION_TRANSPOSEDCONV2D_CONVERTER__
#define __CONVERSION_TRANSPOSEDCONV2D_CONVERTER__

#include "CanonicalNodeConverter.h"

#include <loco.h>

namespace exo
{

/**
 * @brief  Convert loco::TransposedConv2D to locoex::TFLTransposeConv and auxiliary
 *
 *
 * <BEFORE>
 *
 * IFM ------- TransposedConv2D --- OFM
 * (Feature)  / (Feature)
 *           /
 * KER ------
 * (Filter)
 *
 *
 * <AFTER>
 *
 * out_backprop :   IFM ------- FeatureDecode --- TFLTransposeConv --- FeatureEncode --- OFM
 *                  [Feature]   [Tensor]         / / [Tensor]            [Feature]
 *                                              / /
 *        filter:   KER ------- FilterDecode --- /
 *                  [Filter]    [Tensor]        /
 *                                             /
 *  input_sizes :   TFLConst (new) ------------
 *                  [Tensor]
 */
class TransposedConv2DConverter : public CanonicalNodeConverter<loco::TransposedConv2D>
{
public:
  const char *name(void) const final { return "exo::TransposedConv2DConverter"; }

public:
  bool convert(loco::TransposedConv2D *origin) final;
};

} // namespace exo

#endif // __CONVERSION_TRANSPOSEDCONV2D_CONVERTER__
