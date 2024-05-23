/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Backend.h"

#include <util/logging.h>

extern "C" {
onert::backend::Backend *onert_backend_create()
{
  VERBOSE(onert_backend_create) << "'gpu_cl' loaded\n";
  return new onert::backend::gpu_cl::Backend;
}

void onert_backend_destroy(onert::backend::Backend *backend)
{
  VERBOSE(onert_backend_destroy) << "'gpu_cl' unloaded\n";
  delete backend;
}
}
