#!/bin/bash
set -e

echo
echo
echo RNN
./legalize_onnx.py RNN.onnx mod_RNN.onnx
./compare.py RNN.onnx mod_RNN.onnx

echo
echo
echo RNN-nobias
./legalize_onnx.py RNN-nobias.onnx mod_RNN-nobias.onnx
./compare.py RNN-nobias.onnx mod_RNN-nobias.onnx

echo
echo
echo RNN-noinit
./legalize_onnx.py RNN-noinit.onnx mod_RNN-noinit.onnx
./compare.py RNN-noinit.onnx mod_RNN-noinit.onnx

echo
echo
echo RNN-relu
./legalize_onnx.py RNN-relu.onnx mod_RNN-relu.onnx
./compare.py RNN-relu.onnx mod_RNN-relu.onnx

echo
echo
echo RNN-bi
./legalize_onnx.py RNN-bi.onnx mod_RNN-bi.onnx
./compare.py RNN-bi.onnx mod_RNN-bi.onnx

echo
echo
echo LSTM
./legalize_onnx.py LSTM.onnx mod_LSTM.onnx
./compare.py LSTM.onnx mod_LSTM.onnx

echo
echo
echo LSTM-noinit
./legalize_onnx.py LSTM-noinit.onnx mod_LSTM-noinit.onnx
./compare.py LSTM-noinit.onnx mod_LSTM-noinit.onnx

echo
echo
echo LSTM-nobias
./legalize_onnx.py LSTM-nobias.onnx mod_LSTM-nobias.onnx
./compare.py LSTM-nobias.onnx mod_LSTM-nobias.onnx

echo
echo
echo LSTM-bi
./legalize_onnx.py LSTM-bi.onnx mod_LSTM-bi.onnx
./compare.py LSTM-bi.onnx mod_LSTM-bi.onnx

echo SUCCESS
