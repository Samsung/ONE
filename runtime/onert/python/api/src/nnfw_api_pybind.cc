#include "nnfw.h"
#include "nnfw_api_pybind.h"

namespace py = pybind11;

void ensure_status(NNFW_STATUS status)
{
  switch (status)
  {
    case NNFW_STATUS::NNFW_STATUS_NO_ERROR:
      break;
    case NNFW_STATUS::NNFW_STATUS_ERROR:
      std::cout << "[NNFW_STATUS]\tNNFW_STATUS_ERROR\n";
      exit(1);
      break;
    case NNFW_STATUS::NNFW_STATUS_UNEXPECTED_NULL:
      std::cout << "[NNFW_STATUS]\tNNFW_STATUS_UNEXPECTED_NULL\n";
      exit(1);
      break;
    case NNFW_STATUS::NNFW_STATUS_INVALID_STATE:
      std::cout << "[NNFW_STATUS]\tNNFW_STATUS_INVALID_STATE\n";
      exit(1);
      break;
    case NNFW_STATUS::NNFW_STATUS_OUT_OF_MEMORY:
      std::cout << "[NNFW_STATUS]\tNNFW_STATUS_OUT_OF_MEMORY\n";
      exit(1);
      break;
    case NNFW_STATUS::NNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE:
      std::cout << "[NNFW_STATUS]\tNNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE\n";
      exit(1);
      break;
  }
}

NNFW_LAYOUT layout_status(const char *layout = "")
{
  if (!strcmp(layout, "NCHW"))
  {
    return NNFW_LAYOUT::NNFW_LAYOUT_CHANNELS_FIRST;
  }

  else if (!strcmp(layout, "NHWC"))
  {
    return NNFW_LAYOUT::NNFW_LAYOUT_CHANNELS_LAST;
  }

  else if (!strcmp(layout, "NONE"))
  {
    return NNFW_LAYOUT::NNFW_LAYOUT_NONE;
  }

  else
  {
    std::cout << "layout type error\n";
    exit(-1);
  }
}

uint64_t num_elems(const nnfw_tensorinfo *tensor_info)
{
  uint64_t n = 1;
  for (uint32_t i = 0; i < tensor_info->rank; ++i)
  {
    n *= tensor_info->dims[i];
  }
  return n;
}

py::list get_dims(const nnfw_tensorinfo &tensor_info)
{
  py::list dims_list;
  for (int32_t i = 0; i < tensor_info.rank; ++i)
  {
    dims_list.append(tensor_info.dims[i]);
  }
  return dims_list;
}

void set_dims(nnfw_tensorinfo &tensor_info, const py::list &array)
{
  tensor_info.rank = py::len(array);
  for (int32_t i = 0; i < tensor_info.rank; ++i)
  {
    tensor_info.dims[i] = py::cast<int32_t>(array[i]);
  }
}

NNFW_SESSION::NNFW_SESSION() : session(nullptr) {}
NNFW_SESSION::~NNFW_SESSION()
{
  if (session)
  {
    close_session();
  }
}

void NNFW_SESSION::create_session() { ensure_status(nnfw_create_session(&session)); }
void NNFW_SESSION::close_session()
{
  ensure_status(nnfw_close_session(session));
  session = nullptr;
}
void NNFW_SESSION::load_model_from_file(const char *package_file_path)
{
  ensure_status(nnfw_load_model_from_file(session, package_file_path));
}
void NNFW_SESSION::apply_tensorinfo(uint32_t index, nnfw_tensorinfo tensor_info)
{
  ensure_status(nnfw_apply_tensorinfo(session, index, tensor_info));
}
void NNFW_SESSION::set_input_tensorinfo(uint32_t index, const nnfw_tensorinfo *tensor_info)
{
  ensure_status(nnfw_set_input_tensorinfo(session, index, tensor_info));
}
void NNFW_SESSION::prepare() { ensure_status(nnfw_prepare(session)); }
void NNFW_SESSION::run() { ensure_status(nnfw_run(session)); }
void NNFW_SESSION::run_async() { ensure_status(nnfw_run_async(session)); }
void NNFW_SESSION::await() { ensure_status(nnfw_await(session)); }
template <typename T>
void NNFW_SESSION::set_input(uint32_t index, nnfw_tensorinfo *tensor_info, py::array_t<T> &buffer)
{
  NNFW_TYPE type = tensor_info->dtype;
  uint32_t input_elements = num_elems(tensor_info);
  size_t length = sizeof(T) * input_elements;

  ensure_status(nnfw_set_input(session, index, type, buffer.request().ptr, length));
}
template <typename T>
void NNFW_SESSION::set_output(uint32_t index, nnfw_tensorinfo *tensor_info, py::array_t<T> &buffer)
{
  NNFW_TYPE type = tensor_info->dtype;
  uint32_t output_elements = num_elems(tensor_info);
  size_t length = sizeof(T) * output_elements;

  ensure_status(nnfw_set_output(session, index, type, buffer.request().ptr, length));
}
uint32_t NNFW_SESSION::input_size()
{
  uint32_t number;
  NNFW_STATUS status = nnfw_input_size(session, &number);
  ensure_status(status);
  return number;
}
uint32_t NNFW_SESSION::output_size()
{
  uint32_t number;
  NNFW_STATUS status = nnfw_output_size(session, &number);
  ensure_status(status);
  return number;
}
void NNFW_SESSION::set_input_layout(uint32_t index, const char *layout)
{
  NNFW_LAYOUT nnfw_layout = layout_status(layout);
  ensure_status(nnfw_set_input_layout(session, index, nnfw_layout));
}
void NNFW_SESSION::set_output_layout(uint32_t index, const char *layout)
{
  NNFW_LAYOUT nnfw_layout = layout_status(layout);
  ensure_status(nnfw_set_output_layout(session, index, nnfw_layout));
}
nnfw_tensorinfo NNFW_SESSION::input_tensorinfo(uint32_t index)
{
  nnfw_tensorinfo tensor_info = nnfw_tensorinfo();
  ensure_status(nnfw_input_tensorinfo(session, index, &tensor_info));
  return tensor_info;
}
nnfw_tensorinfo NNFW_SESSION::output_tensorinfo(uint32_t index)
{
  nnfw_tensorinfo tensor_info = nnfw_tensorinfo();
  ensure_status(nnfw_output_tensorinfo(session, index, &tensor_info));
  return tensor_info;
}
void NNFW_SESSION::set_available_backends(const char *backends)
{
  ensure_status(nnfw_set_available_backends(session, backends));
}
void NNFW_SESSION::set_op_backend(const char *op, const char *backend)
{
  ensure_status(nnfw_set_op_backend(session, op, backend));
}
uint32_t NNFW_SESSION::query_info_u32(NNFW_INFO_ID id)
{
  uint32_t val;
  ensure_status(nnfw_query_info_u32(session, id, &val));
  return val;
}

PYBIND11_MODULE(nnfw_api_pybind, m)
{
  m.doc() = "python nnfw plugin";

  py::enum_<NNFW_TYPE>(m, "NNFW_TYPE")
    .value("NNFW_TYPE_TENSOR_FLOAT32", NNFW_TYPE::NNFW_TYPE_TENSOR_FLOAT32)
    .value("NNFW_TYPE_TENSOR_INT32", NNFW_TYPE::NNFW_TYPE_TENSOR_INT32)
    .value("NNFW_TYPE_TENSOR_QUANT8_ASYMM", NNFW_TYPE::NNFW_TYPE_TENSOR_QUANT8_ASYMM)
    .value("NNFW_TYPE_TENSOR_BOOL", NNFW_TYPE::NNFW_TYPE_TENSOR_BOOL)
    .value("NNFW_TYPE_TENSOR_UINT8", NNFW_TYPE::NNFW_TYPE_TENSOR_UINT8)
    .value("NNFW_TYPE_TENSOR_INT64", NNFW_TYPE::NNFW_TYPE_TENSOR_INT64)
    .value("NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED", NNFW_TYPE::NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED)
    .value("NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED", NNFW_TYPE::NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED)
    .export_values();

  py::enum_<NNFW_LAYOUT>(m, "NNFW_LAYOUT")
    .value("NNFW_LAYOUT_NONE", NNFW_LAYOUT::NNFW_LAYOUT_NONE)
    .value("NNFW_LAYOUT_CHANNELS_LAST", NNFW_LAYOUT::NNFW_LAYOUT_CHANNELS_LAST)
    .value("NNFW_LAYOUT_CHANNELS_FIRST", NNFW_LAYOUT::NNFW_LAYOUT_CHANNELS_FIRST)
    .export_values();

  py::enum_<NNFW_INFO_ID>(m, "NNFW_INFO_ID")
    .value("NNFW_INFO_ID_VERSION", NNFW_INFO_ID::NNFW_INFO_ID_VERSION)
    .export_values();

  py::class_<nnfw_tensorinfo>(m, "nnfw_tensorinfo")
    .def(py::init<>())
    .def_readwrite("dtype", &nnfw_tensorinfo::dtype)
    .def_readwrite("rank", &nnfw_tensorinfo::rank)
    .def_property(
      "dims", [](const nnfw_tensorinfo &tensorinfo) { return get_dims(tensorinfo); },
      [](nnfw_tensorinfo &tensorinfo, const py::list &dims_list) {
        set_dims(tensorinfo, dims_list);
      });

  py::class_<NNFW_SESSION>(m, "nnfw_session")
    .def(py::init<>())
    .def("create_session", &NNFW_SESSION::create_session)
    .def("close_session", &NNFW_SESSION::close_session)
    .def("load_model_from_file", &NNFW_SESSION::load_model_from_file)
    .def("apply_tensorinfo", &NNFW_SESSION::apply_tensorinfo)
    .def("set_input_tensorinfo", &NNFW_SESSION::set_input_tensorinfo)
    .def("prepare", &NNFW_SESSION::prepare)
    .def("run", &NNFW_SESSION::run)
    .def("run_async", &NNFW_SESSION::run_async)
    .def("await", &NNFW_SESSION::await)
    .def("set_input",
         [](NNFW_SESSION &session, uint32_t index, nnfw_tensorinfo *tensorinfo,
            py::array_t<float> &buffer) { session.set_input<float>(index, tensorinfo, buffer); })
    .def("set_input",
         [](NNFW_SESSION &session, uint32_t index, nnfw_tensorinfo *tensorinfo,
            py::array_t<int> &buffer) { session.set_input<int>(index, tensorinfo, buffer); })
    .def(
      "set_input",
      [](NNFW_SESSION &session, uint32_t index, nnfw_tensorinfo *tensorinfo,
         py::array_t<uint8_t> &buffer) { session.set_input<uint8_t>(index, tensorinfo, buffer); })
    .def("set_input",
         [](NNFW_SESSION &session, uint32_t index, nnfw_tensorinfo *tensorinfo,
            py::array_t<bool> &buffer) { session.set_input<bool>(index, tensorinfo, buffer); })
    .def(
      "set_input",
      [](NNFW_SESSION &session, uint32_t index, nnfw_tensorinfo *tensorinfo,
         py::array_t<int64_t> &buffer) { session.set_input<int64_t>(index, tensorinfo, buffer); })
    .def("set_input",
         [](NNFW_SESSION &session, uint32_t index, nnfw_tensorinfo *tensorinfo,
            py::array_t<int8_t> &buffer) { session.set_input<int8_t>(index, tensorinfo, buffer); })
    .def(
      "set_input",
      [](NNFW_SESSION &session, uint32_t index, nnfw_tensorinfo *tensorinfo,
         py::array_t<int16_t> &buffer) { session.set_input<int16_t>(index, tensorinfo, buffer); })
    .def("set_output",
         [](NNFW_SESSION &session, uint32_t index, nnfw_tensorinfo *tensorinfo,
            py::array_t<float> &buffer) { session.set_output<float>(index, tensorinfo, buffer); })
    .def("set_output",
         [](NNFW_SESSION &session, uint32_t index, nnfw_tensorinfo *tensorinfo,
            py::array_t<int> &buffer) { session.set_output<int>(index, tensorinfo, buffer); })
    .def(
      "set_output",
      [](NNFW_SESSION &session, uint32_t index, nnfw_tensorinfo *tensorinfo,
         py::array_t<uint8_t> &buffer) { session.set_output<uint8_t>(index, tensorinfo, buffer); })
    .def("set_output",
         [](NNFW_SESSION &session, uint32_t index, nnfw_tensorinfo *tensorinfo,
            py::array_t<bool> &buffer) { session.set_output<bool>(index, tensorinfo, buffer); })
    .def(
      "set_output",
      [](NNFW_SESSION &session, uint32_t index, nnfw_tensorinfo *tensorinfo,
         py::array_t<int64_t> &buffer) { session.set_output<int64_t>(index, tensorinfo, buffer); })
    .def("set_output",
         [](NNFW_SESSION &session, uint32_t index, nnfw_tensorinfo *tensorinfo,
            py::array_t<int8_t> &buffer) { session.set_output<int8_t>(index, tensorinfo, buffer); })
    .def(
      "set_output",
      [](NNFW_SESSION &session, uint32_t index, nnfw_tensorinfo *tensorinfo,
         py::array_t<int16_t> &buffer) { session.set_output<int16_t>(index, tensorinfo, buffer); })
    .def("input_size", &NNFW_SESSION::input_size)
    .def("output_size", &NNFW_SESSION::output_size)
    .def("set_input_layout", &NNFW_SESSION::set_input_layout, py::arg("index"),
         py::arg("layout") = "NONE")
    .def("set_output_layout", &NNFW_SESSION::set_output_layout, py::arg("index"),
         py::arg("layout") = "NONE")
    .def("input_tensorinfo", &NNFW_SESSION::input_tensorinfo)
    .def("output_tensorinfo", &NNFW_SESSION::output_tensorinfo)
    .def("set_available_backends", &NNFW_SESSION::set_available_backends)
    .def("set_op_backend", &NNFW_SESSION::set_op_backend)
    .def("query_info_u32", &NNFW_SESSION::query_info_u32);
}