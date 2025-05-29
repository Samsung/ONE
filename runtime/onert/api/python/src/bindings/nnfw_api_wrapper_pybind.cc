/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <pybind11/pybind11.h>

#include "nnfw_exception_bindings.h"
#include "nnfw_session_bindings.h"
#include "nnfw_tensorinfo_bindings.h"
#include "nnfw_traininfo_bindings.h"

using namespace onert::api::python;
namespace py = pybind11;

PYBIND11_MODULE(libnnfw_api_pybind, m)
{
  m.doc() = "Main module that contains infer and experimental submodules";

  // Bind common `NNFW_SESSION` class
  bind_nnfw_session(m);

  // Bind `NNFW_SESSION` class for inference
  // Currently, the `infer` session is the same as common.
  auto infer = m.def_submodule("infer", "Inference submodule");
  infer.attr("nnfw_session") = m.attr("nnfw_session");

  // Bind our NNFW-status exceptions
  auto ex = m.def_submodule("exception", "NNFW-status Exception");
  bind_nnfw_exceptions(ex);

  // Bind experimental `NNFW_SESSION` class
  auto experimental = m.def_submodule("experimental", "Experimental submodule");
  experimental.attr("nnfw_session") = m.attr("nnfw_session");
  bind_experimental_nnfw_session(experimental);

  // Bind common `tensorinfo` class
  bind_tensorinfo(m);

  // Bind enums
  bind_nnfw_enums(m);

  m.doc() = "NNFW Python Bindings for Training";

  // Bind training enums
  bind_nnfw_train_enums(m);

  // Bind training nnfw_loss_info
  bind_nnfw_loss_info(m);

  // Bind_train_info
  bind_nnfw_train_info(m);
}
