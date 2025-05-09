mkdir 3rd
cd 3rd
git clone https://github.com/ekg/glia.git
cp -r glia/json ../include
cp glia/json-forwards.h ../include
cp glia/jsoncpp.cpp ../src/lib
cd ..
rm -rf 3rd
