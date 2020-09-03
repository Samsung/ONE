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

#include "NNPackages.h"
#include "fixtures.h"

using ValidationTestFourAddModelsSetInput = ValidationTestFourModelsSetInput<NNPackages::ADD>;

TEST_F(ValidationTestFourAddModelsSetInput, run_001)
{
  NNFW_ENSURE_SUCCESS(nnfw_run(_objects[0].session));
  NNFW_ENSURE_SUCCESS(nnfw_run(_objects[1].session));

  SUCCEED();
}

TEST_F(ValidationTestFourAddModelsSetInput, run_002)
{
  int rep = 3;
  while (rep--)
  {
    for (auto obj : _objects)
      NNFW_ENSURE_SUCCESS(nnfw_run(obj.session));
  }

  SUCCEED();
}

TEST_F(ValidationTestFourAddModelsSetInput, run_async)
{
  for (auto obj : _objects)
    NNFW_ENSURE_SUCCESS(nnfw_run_async(obj.session));
  for (auto obj : _objects)
    NNFW_ENSURE_SUCCESS(nnfw_await(obj.session));

  SUCCEED();
}
