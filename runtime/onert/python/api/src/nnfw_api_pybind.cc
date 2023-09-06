#include <pybind11.h>
#include <stl_bind.h>
#include <stl.h>
#include <cstdlib>
#include <cstdint>
#include "nnfw.h"
#include "nnfw_api_internal.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<float>)
PYBIND11_MAKE_OPAQUE(std::vector<uint32_t>)

uint64_t num_elems(const nnfw_tensorinfo *ti)
{
    uint64_t n = 1;
    for (uint32_t i = 0; i < ti->rank; ++i)
    {
        n *= ti->dims[i];
    }
    return n;
}

// define nnfw_session as a class
class NNFW_SESSION
{

private:
    // define nnfw_session when class is declared
    nnfw_session *session;

public:
    NNFW_STATUS create()
    {
        return nnfw_create_session(&session);
    }
    NNFW_STATUS close()
    {
        NNFW_STATUS state = nnfw_close_session(session);
        session = nullptr;
        return state;
    }
    NNFW_STATUS load_model_from_file(const char *package_file_path)
    {
        return nnfw_load_model_from_file(session, package_file_path);
    }
    NNFW_STATUS apply_tensorinfo(uint32_t index, const nnfw_tensorinfo &tensor_info)
    {
        return nnfw_apply_tensorinfo(session, index, tensor_info);
    }
    NNFW_STATUS set_input_tensorinfo(uint32_t index, nnfw_tensorinfo &tensor_info)
    {
        return nnfw_set_input_tensorinfo(session, index, &tensor_info);
    }
    NNFW_STATUS prepare()
    {
        return nnfw_prepare(session);
    }
    NNFW_STATUS run()
    {
        return nnfw_run(session);
    }
    NNFW_STATUS run_async()
    {
        return nnfw_run_async(session);
    }
    NNFW_STATUS await()
    {
        return nnfw_await(session);
    }
    NNFW_STATUS set_input(uint32_t index, std::vector<float> &input, nnfw_tensorinfo &tensor_info, size_t length)
    {
        return nnfw_set_input(session, index, tensor_info.dtype, input.data(), sizeof(float) * length);
    }
    NNFW_STATUS set_output(uint32_t index, std::vector<float> &output, nnfw_tensorinfo &tensor_info, size_t length)
    {
        return nnfw_set_output(session, index, tensor_info.dtype, output.data(), sizeof(float) * length);
    }
    NNFW_STATUS input_size(std::vector<uint32_t> &number)
    {
        return nnfw_input_size(session, number.data());
    }
    NNFW_STATUS output_size(std::vector<uint32_t> &number)
    {
        return nnfw_output_size(session, number.data());
    }
    NNFW_STATUS set_input_layout(uint32_t index, NNFW_LAYOUT layout)
    {
        return nnfw_set_input_layout(session, index, layout);
    }
    NNFW_STATUS set_output_layout(uint32_t index, NNFW_LAYOUT layout)
    {
        return nnfw_set_output_layout(session, index, layout);
    }
    NNFW_STATUS input_tensorinfo(uint32_t index, nnfw_tensorinfo &tensor_info)
    {
        return nnfw_input_tensorinfo(session, index, &tensor_info);
    }
    NNFW_STATUS output_tensorinfo(uint32_t index, nnfw_tensorinfo &tensor_info)
    {
        return nnfw_output_tensorinfo(session, index, &tensor_info);
    }
    NNFW_STATUS set_available_backends(const char *backends)
    {
        return nnfw_set_available_backends(session, backends);
    }
    NNFW_STATUS set_op_backend(const char *op, const char *backend)
    {
        return nnfw_set_op_backend(session, op, backend);
    }
    NNFW_STATUS query_info_u32(NNFW_INFO_ID id, uint32_t *val)
    {
        return nnfw_query_info_u32(session, id, val);
    }
};

// Function to get the values from nnfw_tensorinfo array and convert to a Python list
py::list get_dims(const nnfw_tensorinfo &tensorinfo)
{
    py::list dims_list;
    for (int i = 0; i < tensorinfo.rank; ++i)
    {
        dims_list.append(tensorinfo.dims[i]);
    }
    return dims_list;
}

// Function to set the values of nnfw_tensorinfo array from a Python list
void set_dims(nnfw_tensorinfo &tensorinfo, const py::list &dims_list)
{
    tensorinfo.rank = py::len(dims_list);
    for (int i = 0; i < tensorinfo.rank; ++i)
    {
        tensorinfo.dims[i] = py::cast<int>(dims_list[i]);
    }
}

PYBIND11_MODULE(nnfw_api_pybind, m)
{
    m.doc() = "Python binding for nnfw";

    py::enum_<NNFW_TYPE>(m, "NNFW_TYPE")
        .value("NNFW_TYPE_TENSOR_FLOAT32", NNFW_TYPE_TENSOR_FLOAT32)
        .value("NNFW_TYPE_TENSOR_INT32", NNFW_TYPE_TENSOR_INT32)
        .value("NNFW_TYPE_TENSOR_QUANT8_ASYMM", NNFW_TYPE_TENSOR_QUANT8_ASYMM)
        .value("NNFW_TYPE_TENSOR_BOOL", NNFW_TYPE_TENSOR_BOOL)
        .value("NNFW_TYPE_TENSOR_UINT8", NNFW_TYPE_TENSOR_UINT8)
        .value("NNFW_TYPE_TENSOR_INT64", NNFW_TYPE_TENSOR_INT64)
        .value("NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED", NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED)
        .value("NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED", NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED)
        .export_values();

    py::enum_<NNFW_STATUS>(m, "NNFW_STATUS")
        .value("NNFW_STATUS_NO_ERROR", NNFW_STATUS_NO_ERROR)
        .value("NNFW_STATUS_ERROR", NNFW_STATUS_ERROR)
        .value("NNFW_STATUS_UNEXPECTED_NULL", NNFW_STATUS_UNEXPECTED_NULL)
        .value("NNFW_STATUS_INVALID_STATE", NNFW_STATUS_INVALID_STATE)
        .value("NNFW_STATUS_OUT_OF_MEMORY", NNFW_STATUS_OUT_OF_MEMORY)
        .value("NNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE", NNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE)
        .export_values();

    py::enum_<NNFW_LAYOUT>(m, "NNFW_LAYOUT")
        .value("NNFW_LAYOUT_NONE", NNFW_LAYOUT_NONE)
        .value("NNFW_LAYOUT_CHANNELS_LAST", NNFW_LAYOUT_CHANNELS_LAST)
        .value("NNFW_LAYOUT_CHANNELS_FIRST", NNFW_LAYOUT_CHANNELS_FIRST)
        .export_values();

    py::enum_<NNFW_INFO_ID>(m, "NNFW_INFO_ID")
        .value("NNFW_INFO_ID_VERSION", NNFW_INFO_ID_VERSION)
        .export_values();

    py::class_<nnfw_tensorinfo>(m, "nnfw_tensorinfo")
        .def(py::init<>())
        .def_readwrite("dtype", &nnfw_tensorinfo::dtype)
        .def_readwrite("rank", &nnfw_tensorinfo::rank)
        .def_property(
            "dims",
            [](const nnfw_tensorinfo &tensorinfo)
            {
                return get_dims(tensorinfo);
            },
            [](nnfw_tensorinfo &tensorinfo, const py::list &dims_list)
            {
                set_dims(tensorinfo, dims_list);
            });

    py::bind_vector<std::vector<float>>(m, "float_vector");
    py::bind_vector<std::vector<uint32_t>>(m, "uint32_t_vector");

    py::class_<NNFW_SESSION>(m, "nnfw_session")
        .def(py::init<>())
        .def("create", &NNFW_SESSION::create)
        .def("close", &NNFW_SESSION::close)
        .def("load_model_from_file", &NNFW_SESSION::load_model_from_file)
        .def("apply_tensorinfo", &NNFW_SESSION::apply_tensorinfo)
        .def("set_input_tensorinfo", &NNFW_SESSION::set_input_tensorinfo)
        .def("prepare", &NNFW_SESSION::prepare)
        .def("run", &NNFW_SESSION::run)
        .def("run_async", &NNFW_SESSION::run_async)
        .def("await", &NNFW_SESSION::await)
        .def("set_input", &NNFW_SESSION::set_input)
        .def("set_output", &NNFW_SESSION::set_output)
        .def("input_size", &NNFW_SESSION::input_size)
        .def("output_size", &NNFW_SESSION::output_size)
        .def("set_input_layout", &NNFW_SESSION::set_input_layout)
        .def("set_output_layout", &NNFW_SESSION::set_output_layout)
        .def("input_tensorinfo", &NNFW_SESSION::input_tensorinfo)
        .def("output_tensorinfo", &NNFW_SESSION::output_tensorinfo)
        .def("set_available_backends", &NNFW_SESSION::set_available_backends)
        .def("set_op_backend", &NNFW_SESSION::set_op_backend)
        .def("query_info_u32", &NNFW_SESSION::query_info_u32);
}