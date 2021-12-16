def get_string_from(string, str_content):
    add_definitions_start = str_content.find(string) + len(string) + 1
    add_definitions_end = str_content[add_definitions_start:].find(")") + add_definitions_start
    return str_content[add_definitions_start:add_definitions_end]

str_content = ''

with open('CMakeListsMbed.txt', 'r') as f:
    str_content = f.read()

include_directories = get_string_from("INCLUDE_DIRECTORIES(", str_content)
add_definitions = get_string_from("ADD_DEFINITIONS(", str_content)
add_executable = get_string_from("ADD_EXECUTABLE(mbed-os-example-blinky", str_content)

add_executable = add_executable[add_executable.find("main.cpp") + len("main.cpp") + 1:]
add_executable = add_executable.replace("mbed-os", "${ARGV0}")

include_directories = include_directories[:include_directories.find("mbed-os\n") + len("mbed-os\n")]

mbed_sources_content = "macro(set_sources_mbed)\n" + "set(SOURCES \n" + add_executable + ")\n" + "endmacro()\n" + \
                       "macro(target_include_directories_mbed)\n" + "target_include_directories(${ARGV0} PRIVATE\n" + \
                       include_directories.replace("mbed-os", "${ARGV1}") + ")\n" + "endmacro()\n" + \
                       "macro(add_definitions_mbed)\n" + "ADD_DEFINITIONS(\n" + add_definitions + ")\n" + "endmacro()\n"

with open('mbed-sources.cmake', 'w') as f:
    f.write(mbed_sources_content)
