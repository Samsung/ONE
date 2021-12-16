mbed import https://github.com/ARMmbed/mbed-os-example-blinky
cd mbed-os-example-blinky
cd mbed-os
pip install -r requirements.txt
cd ..
mbed export -i CMAKE_GCC_ARM --source . -m $1
cp CMakeLists.txt ../CMakeListsMbed.txt
cd ..
rm -rf mbed-os-example-blinky