# Origin of source code

This library is based on Json-cpp amalgated header and cpp files(https://github.com/open-source-parsers/jsoncpp/wiki/Amalgamated-(Possibly-outdated))

Manual fix `jsoncpp.cpp`
- Remove unused `fixNumericLocaleInput()` function
- Remove unused `getDecimalPoint()` function (used by `fixNumericLocaleInput()` only, but removed)

# Background

Since jsoncpp on tizen does not support static jsoncpp library, nnfw project will link this local library.

# Version

- 1.9.6 : https://github.com/open-source-parsers/jsoncpp/archive/1.9.6.tar.gz
