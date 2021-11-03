///*
// * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *    http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//
//#include "luci/Log.h"
//
//#include <luci/UserSettings.h>
//
//#include <cassert>
//#include <cstdlib>
//#include <iostream>
//
//// TODO Extract these lexical conversion routines as a library
//namespace
//{
//
///**
// * @brief Convert C-string as a value of type T
// *
// * safecast(s, v) returns v if s is nullptr.
// */
//template <typename T> T safecast(const char *, const T &);
//
//template <> bool safecast<bool>(const char *s, const bool &value)
//{
//  return (s == nullptr) ? value : (std::stoi(s) != 0);
//}
//
//template <> int safecast<int>(const char *s, const int &value)
//{
//  return (s == nullptr) ? value : std::stoi(s);
//}
//
//} // namespace
//
////
//// Logger
////
//namespace luci
//{
//
//Logger::Logger(hermes::Context *ctx) { activate(ctx->sources(), ctx->bus()); }
//Logger::~Logger() { deactivate(); }
//
//} // namespace luci
//
////
//// LoggerConfig
////
//namespace luci
//{
//
//LoggerConfig::LoggerConfig()
//{
//  auto settings = luci::UserSettings::settings();
//
//  _show_warn = !settings->get(luci::UserSettings::Key::MuteWarnings);
//
//  // Turn on info logging if LUCI_LOG is set as non-zero value
//  _show_info = safecast<bool>(std::getenv("LUCI_LOG"), false);
//
//  // Turn on verbose logging if LUCI_LOG is set to some level
//  // VERBOSE(l, 1) will be visible with LUCI_LOG=2 and VERBOSE(l, 2) with LUCI_LOG=3 and so on
//  _show_verbose = safecast<int>(std::getenv("LUCI_LOG"), 0);
//}
//
//void LoggerConfig::configure(const hermes::Source *source, hermes::Source::Setting &setting) const
//{
//  // Let's ignore hermes::Sources if that is not a moco logger
//  if (auto logger = dynamic_cast<const Logger *>(source))
//  {
//    configure(logger, setting);
//  }
//}
//
//void LoggerConfig::configure(const Logger *, hermes::Source::Setting &setting) const
//{
//  setting.filter(hermes::SeverityCategory::FATAL).reject_all();
//  setting.filter(hermes::SeverityCategory::ERROR).reject_all();
//  setting.filter(hermes::SeverityCategory::WARN).reject_all();
//  setting.filter(hermes::SeverityCategory::INFO).reject_all();
//  setting.filter(hermes::SeverityCategory::VERBOSE).reject_all();
//
//  // TODO enable FATAL and ERROR
//  if (_show_warn)
//  {
//    setting.filter(hermes::SeverityCategory::WARN).accept_all();
//  }
//  if (_show_info)
//  {
//    setting.filter(hermes::SeverityCategory::INFO).accept_all();
//  }
//  if (_show_verbose)
//  {
//    setting.filter(hermes::SeverityCategory::VERBOSE).accept_upto(_show_verbose);
//  }
//}
//
//} // namespace luci
