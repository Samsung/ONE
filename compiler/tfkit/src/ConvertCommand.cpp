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

#include "ConvertCommand.hpp"
#include "Support.hpp"

#include <tensorflow/core/framework/graph.pb.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>

#include <memory>
#include <cassert>
#include <map>
#include <string>

// TODO Extract this as a library
namespace
{

enum class DataFormat
{
  PBBIN,
  PBTXT,
  JSON,
};

struct Importer
{
  virtual ~Importer() = default;

  virtual bool run(std::istream *, tensorflow::GraphDef &) const = 0;
};

struct Exporter
{
  virtual ~Exporter() = default;

  virtual bool run(const tensorflow::GraphDef &, std::ostream *) const = 0;
};

template <DataFormat F> class ImporterImpl;

template <> class ImporterImpl<DataFormat::PBTXT> final : public Importer
{
public:
  bool run(std::istream *is, tensorflow::GraphDef &graph_def) const final
  {
    google::protobuf::io::IstreamInputStream iis{is};
    return google::protobuf::TextFormat::Parse(&iis, &graph_def);
  }
};

template <> class ImporterImpl<DataFormat::PBBIN> final : public Importer
{
public:
  bool run(std::istream *is, tensorflow::GraphDef &graph_def) const final
  {
    google::protobuf::io::IstreamInputStream iis{is};
    google::protobuf::io::CodedInputStream cis{&iis};
    return graph_def.ParseFromCodedStream(&cis);
  }
};

template <DataFormat F> class ExporterImpl;

template <> class ExporterImpl<DataFormat::JSON> final : public Exporter
{
public:
  bool run(const tensorflow::GraphDef &graph_def, std::ostream *os) const final
  {
    std::string str;
    google::protobuf::util::MessageToJsonString(graph_def, &str);
    *os << str << std::endl;
    return true;
  }
};

} // namespace

namespace tfkit
{

int ConvertCommand::run(int argc, const char *const *argv) const
{
  tensorflow::GraphDef graph_def;

  // NOTE The current implementation accepts only command-line for the following form:
  //
  //   tfkit convert --input-format (pb or pbtxt) --output-format json ...
  //
  // TODO Support more options
  assert(argc >= 4);
  assert(std::string(argv[0]) == "--input-format");
  const std::string input_format{argv[1]};
  assert(std::string(argv[2]) == "--output-format");
  const std::string output_format{argv[3]};

  std::map<std::string, std::unique_ptr<Importer>> importers;

  importers["pb"] = std::make_unique<ImporterImpl<DataFormat::PBBIN>>();
  importers["pbtxt"] = std::make_unique<ImporterImpl<DataFormat::PBTXT>>();

  std::map<std::string, std::unique_ptr<Exporter>> exporters;

  exporters["json"] = std::make_unique<ExporterImpl<DataFormat::JSON>>();

  auto importer = importers.at(input_format).get();
  auto exporter = exporters.at(output_format).get();

  CmdArguments cmdargs(argc - 4, argv + 4);

  auto ioconfig = make_ioconfig(cmdargs);

  if (!importer->run(ioconfig->in(), graph_def))
  {
    std::cerr << "ERROR: Failed to import" << std::endl;
    return 255;
  }

  if (!exporter->run(graph_def, ioconfig->out()))
  {
    std::cerr << "ERROR: Failed to export" << std::endl;
    return 255;
  }

  return 0;
}

} // namespace tfkit
