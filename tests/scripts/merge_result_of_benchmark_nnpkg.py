#!/usr/bin/env python
#
# Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys, glob, os
import csv
import copy
import argparse

g_header = []
g_new_header = []
g_backends = []
g_backend_indice = {}


def global_init():
    global g_header
    global g_new_header
    global g_backends

    # TODO How to maintain easily csv header
    g_header = [
        "Model",
        "Backend",
        "ModelLoad_Time",
        "Prepare_Time",
        "Execute_Time_Min",
        "Execute_Time_Max",
        "Execute_Time_Mean",
        "ModelLoad_RSS",
        "Prepare_RSS",
        "Execute_RSS",
        "Peak_RSS",
        "ModelLoad_HWM",
        "Prepare_HWM",
        "Execute_HWM",
        "Peak_HWM",
        "ModelLoad_PSS",
        "Prepare_PSS",
        "Execute_PSS",
        "Peak_PSS",
    ]

    g_new_header = g_header + [
        'Execute_Time_Mean_Diff_Vs_Tflite_Cpu',
        'Execute_Time_Mean_Ratio_Vs_Tflite_Cpu',
        'Peak_RSS_Diff_Vs_Tflite_Cpu',
        'Peak_RSS_Ratio_Vs_Tflite_Cpu',
        'Peak_HWM_Diff_Vs_Tflite_Cpu',
        'Peak_HWM_Ratio_Vs_Tflite_Cpu',
        'Peak_PSS_Diff_Vs_Tflite_Cpu',
        'Peak_PSS_Ratio_Vs_Tflite_Cpu',
    ]

    # if new backend comes from csv, it will be stored in g_backends
    g_backends = [
        'tflite_cpu',
        'acl_cl',
        'acl_neon',
        'cpu',
    ]
    for i in range(len(g_backends)):
        b = g_backends[i]
        g_backend_indice[b] = i


class Data(object):
    def __init__(self, csv_reader=None, empty=False):
        # header
        self.Model = ""
        self.Backend = ""
        self.ModelLoad_Time = 0.0
        self.Prepare_Time = 0.0
        self.Execute_Time_Min = 0.0
        self.Execute_Time_Max = 0.0
        self.Execute_Time_Mean = 0.0  # will be compared to
        self.ModelLoad_RSS = 0
        self.Prepare_RSS = 0
        self.Execute_RSS = 0
        self.Peak_RSS = 0  # too
        self.ModelLoad_HWM = 0
        self.Prepare_HWM = 0
        self.Execute_HWM = 0
        self.Peak_HWM = 0  # too
        self.ModelLoad_PSS = 0
        self.Prepare_PSS = 0
        self.Execute_PSS = 0
        self.Peak_PSS = 0  # too
        self.Empty = empty

        if csv_reader is not None:
            self.Validate(csv_reader)
            self.Read(csv_reader)

    def Validate(self, csv_reader):
        global g_header

        # validate only the first row
        for row in csv_reader:
            assert (len(row) == len(g_header))
            for i in range(len(row)):
                assert (row[i] == g_header[i])
            break

    def Read(self, csv_reader):
        global g_header
        global g_backends

        for row in csv_reader:
            if row == g_header:
                continue
            self.Model = row[0]
            self.Backend = row[1]
            self.ModelLoad_Time = float(row[2])
            self.Prepare_Time = float(row[3])
            self.Execute_Time_Min = float(row[4])
            self.Execute_Time_Max = float(row[5])
            self.Execute_Time_Mean = float(row[6])
            self.ModelLoad_RSS = int(row[7])
            self.Prepare_RSS = int(row[8])
            self.Execute_RSS = int(row[9])
            self.Peak_RSS = int(row[10])
            self.ModelLoad_HWM = int(row[11])
            self.Prepare_HWM = int(row[12])
            self.Execute_HWM = int(row[13])
            self.Peak_HWM = int(row[14])
            self.ModelLoad_PSS = int(row[15])
            self.Prepare_PSS = int(row[16])
            self.Execute_PSS = int(row[17])
            self.Peak_PSS = int(row[18])

            # if new backend comes,
            if self.Backend not in g_backends:
                g_backends.append(self.Backend)

    def Print(self):
        global g_header
        for attr in g_header:
            print("{}: {}".format(attr, getattr(self, attr)))

    def Row(self):
        global g_header
        row = []
        for attr in g_header:
            row.append(getattr(self, attr))
        val = 1.0 if self.Empty == False else 0.0
        row.append(0.0)  # 'Execute_Time_Mean_Diff_Vs_Tflite_Cpu'
        row.append(val)  # 'Execute_Time_Mean_Ratio_Vs_Tflite_Cpu'
        row.append(0)  # 'Peak_RSS_Diff_Vs_Tflite_Cpu'
        row.append(val)  # 'Peak_RSS_Ratio_Vs_Tflite_Cpu'
        row.append(0)  # 'Peak_HWM_Diff_Vs_Tflite_Cpu'
        row.append(val)  # 'Peak_HWM_Ratio_Vs_Tflite_Cpu'
        row.append(0)  # 'Peak_PSS_Diff_Vs_Tflite_Cpu'
        row.append(val)  # 'Peak_PSS_Ratio_Vs_Tflite_Cpu'
        return row

    def RowVs(self, vs_data):
        row = self.Row()
        vs_exec_mean = vs_data.Execute_Time_Mean
        vs_peak_rss = vs_data.Peak_RSS
        vs_peak_hwm = vs_data.Peak_HWM
        vs_peak_pss = vs_data.Peak_PSS

        # Execute_Time_Mean_Diff_Vs_Tflite_Cpu
        exec_diff = self.Execute_Time_Mean - vs_exec_mean

        # Execute_Time_Mean_Ratio_Vs_Tflite_Cpu
        try:
            exec_ratio = float(vs_exec_mean) / self.Execute_Time_Mean
        except ZeroDivisionError:
            exec_ratio = 0.0

        # Peak_RSS_Diff_Vs_Tflite_Cpu
        rss_diff = self.Peak_RSS - vs_peak_rss

        # Peak_RSS_Mean_Ratio_Vs_Tflite_Cpu
        try:
            rss_ratio = float(self.Peak_RSS) / vs_peak_rss
        except ZeroDivisionError:
            rss_ratio = 0.0

        # Peak_HWM_Diff_Vs_Tflite_Cpu
        hwm_diff = self.Peak_HWM - vs_peak_hwm

        # Peak_HWM_Mean_Ratio_Vs_Tflite_Cpu
        try:
            hwm_ratio = float(self.Peak_HWM) / vs_peak_hwm
        except ZeroDivisionError:
            hwm_ratio = 0.0

        # Peak_PSS_Diff_Vs_Tflite_Cpu
        pss_diff = self.Peak_PSS - vs_peak_pss

        # Peak_PSS_Mean_Ratio_Vs_Tflite_Cpu
        try:
            pss_ratio = float(self.Peak_PSS) / vs_peak_pss
        except ZeroDivisionError:
            pss_ratio = 0.0

        global g_new_header
        row[g_new_header.index('Execute_Time_Mean_Diff_Vs_Tflite_Cpu')] = exec_diff
        row[g_new_header.index('Execute_Time_Mean_Ratio_Vs_Tflite_Cpu')] = exec_ratio
        row[g_new_header.index('Peak_RSS_Diff_Vs_Tflite_Cpu')] = rss_diff
        row[g_new_header.index('Peak_RSS_Ratio_Vs_Tflite_Cpu')] = rss_ratio
        row[g_new_header.index('Peak_HWM_Diff_Vs_Tflite_Cpu')] = hwm_diff
        row[g_new_header.index('Peak_HWM_Ratio_Vs_Tflite_Cpu')] = hwm_ratio
        row[g_new_header.index('Peak_PSS_Diff_Vs_Tflite_Cpu')] = pss_diff
        row[g_new_header.index('Peak_PSS_Ratio_Vs_Tflite_Cpu')] = pss_ratio
        return row


class Model(object):
    def __init__(self, model_name, model_files):
        global g_backends

        self.model_name = model_name
        self.backends = []
        for i in range(len(g_backends)):
            self.backends.append(None)

        for f in model_files:
            with open(f) as csv_file:
                csv_reader = csv.reader(csv_file)
                for i in range(len(g_backends)):
                    b = g_backends[i]
                    if b in f:
                        self.backends[i] = Data(csv_reader)
                        break


def main():
    # Option
    use = "Usage: %prog [options] filename"
    parser = argparse.ArgumentParser(usage=use)
    parser.add_argument("-i",
                        "--input_dir",
                        dest="input_dir",
                        default=".",
                        help="dir to have csv files")
    parser.add_argument("-o",
                        "--output_dir",
                        dest="output_dir",
                        default=".",
                        help="dir to be moved csv files into")
    parser.add_argument("-l",
                        "--model_list",
                        dest="model_list",
                        help="file to have model list")

    options = parser.parse_args()

    # args check
    input_dir = options.input_dir
    if os.path.isdir(input_dir) == False or os.path.exists(input_dir) == False:
        print("Wrong input dir: {}".format(input_dir))
        exit()

    output_dir = options.output_dir
    if os.path.isdir(output_dir) == False or os.path.exists(output_dir) == False:
        print("Wrong output dir: {}".format(output_dir))
        exit()
    output_dir = os.path.abspath(output_dir)

    model_list_file = options.model_list
    if model_list_file == '':
        print("model list file path is empty")
        exit()

    if os.path.exists(model_list_file) == False:
        print("Wrong model list file: {}".format(model_list_file))
        exit()

    # write to one merged csv file
    new_csv_file = os.path.join(output_dir, "merged_benchmark_result.csv")
    if (os.path.exists(new_csv_file)):
        os.remove(new_csv_file)
    print("new csv file: {}".format(new_csv_file))
    print()

    # decl for using global vars
    global g_header
    global g_new_header
    global g_backends

    # init
    global_init()
    model_to_csvs = {}
    model_list = []
    model_data_list = []

    with open(model_list_file) as f:
        model_list = f.read().splitlines()
        print("Current model list")
        for m in model_list:
            print("* " + m)
            model_to_csvs[m] = []
            model_data_list.append(None)
        print()

    for f in glob.glob(os.path.join(input_dir, "*.csv")):
        # TODO handle if file name doesn't come as we follow
        # f's name has {exec}-{model}-{backend}.csv
        model_name = os.path.basename(f).split("-")[1]
        for model in model_to_csvs:
            if model == model_name:
                model_to_csvs[model].append(f)

    print("Current csv file list")
    for model, csvs in model_to_csvs.items():
        print("* {}: {}".format(model, csvs))
    print()

    for model, csvs in model_to_csvs.items():
        assert (model in model_list)
        ind = model_list.index(model)
        model_data_list[ind] = Model(model, csvs)

    for model_data in model_data_list:
        print("{}: {}".format(model_data.model_name, model_data.backends))
    print()

    def getEmptyData(model_name, backend):
        d = Data(None, True)
        d.Model = model_name
        d.Backend = backend
        return d

    with open(new_csv_file, 'w') as new_csv_file:
        # HEADER
        csv_writer = csv.writer(new_csv_file)
        csv_writer.writerow(g_new_header)

        # DATA
        for model_data in model_data_list:
            # tflite_cpu
            tflite_cpu_data = model_data.backends[0]
            if tflite_cpu_data is not None:
                csv_writer.writerow(tflite_cpu_data.Row())
            else:
                ed = getEmptyData(model_data.model_name, g_backends[0])
                csv_writer.writerow(ed.Row())

            # others
            for i in range(1, len(model_data.backends)):
                row = []
                d = model_data.backends[i]
                if d is None:
                    ed = getEmptyData(model_data.model_name, g_backends[i])
                    row = ed.Row()
                else:
                    if tflite_cpu_data is not None:
                        row = d.RowVs(tflite_cpu_data)
                    else:
                        row = d.Row()
                csv_writer.writerow(row)


if __name__ == "__main__":
    main()
