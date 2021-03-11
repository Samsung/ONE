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

#ifndef __LOGO_SIMPLIFY_DOMAIN_CONVERSION_H__
#define __LOGO_SIMPLIFY_DOMAIN_CONVERSION_H__

#include <logo/Pass.h>

namespace logo
{

/**
 * @brief Simplify redundant domain conversion
 *
 * SimplifyDomainConversionPass recognizes the following patterns:
 * - FeatureDecode followed by FeatureEncode (Feature -> Tensor -> Feature)
 * - FeatureEncode followed by FeatureDecode (Tensor -> Feature -> Tensor)
 * - FilterEncode followed by FilterDecode (Tensor -> Filter -> Tensor)
 * - BiasEncode followed by BiasDecode (Tensor -> Bias -> Tensor)
 * - DepthwiseFilterEncode followed by DepthwiseFilterDecode (Tensor -> DepthwiseFilter -> Tensor)
 * - MatrixDecode followed by MatrixEncode (Matrix -> Tensor -> Matrix)
 * - MatrixEncode followed by MatrixDecode (Tensor -> Matrix -> Tensor)
 * - (TO BE ADDED)
 */
struct SimplifyDomainConversionPass final : public Pass
{
  const char *name(void) const final { return "SimplifyDomainConversionPass"; }

  bool run(loco::Graph *g) final;
};

} // namespace logo

#endif // __LOGO_SIMPLIFY_DOMAIN_CONVERSION_H__
