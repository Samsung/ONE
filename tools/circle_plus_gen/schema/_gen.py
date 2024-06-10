'''
(For developers) generate circle_schema_generated.py and circle_traininfo_generated.py

In short, this script downloads flatc and runs the following commands :
./flatc --python --gen-onefile --gen-object-api ../../../nnpackage/schema/circle_schema.fbs
./flatc --python --gen-onefile --gen-object-api ../../../runtime/libs/circle-schema/include/circle_traininfo.fbs
'''

import wget
import zipfile
import os.path
import stat
import subprocess

FLATC_V23_5_26 = 'https://github.com/google/flatbuffers/releases/download/v23.5.26/Linux.flatc.binary.g++-10.zip'
FLATC_ZIP = 'flatc.zip'
FLATC_EXE = 'flatc'

root_dir = subprocess.run(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE)
ONE_REPO = root_dir.stdout.decode('utf-8')[:-1]  # remove \n
CIRCLE_SCHEMA = ONE_REPO + '/nnpackage/schema/circle_schema.fbs'
TINFO_SCHEMA = ONE_REPO + '/runtime/libs/circle-schema/include/circle_traininfo.fbs'


class FlatcCaller:
    def __call__(self, schema_paths):
        self.__download_flatc()
        for schema_path in schema_paths:
            self.__generate_python_schema(schema_path)
        self.__clear_flatc()

    def __download_flatc(self):
        '''Download flatc and unip it in current directory'''
        wget.download(FLATC_V23_5_26, FLATC_ZIP)

        # unzip 'flatc' in current directory
        with zipfile.ZipFile(FLATC_ZIP, 'r') as zip:
            for f in zip.infolist():
                if f.filename == FLATC_EXE:
                    zip.extract(f, '.')

        if not os.path.isfile(FLATC_EXE):
            raise RuntimeError('Failed to download flatc')

        # add permission to execute
        perm = os.stat(FLATC_EXE)
        os.chmod(FLATC_EXE, perm.st_mode | stat.S_IXUSR)

    def __generate_python_schema(self, schema_path, out_dir='.'):
        '''execute flatc to compile *.fbs into python file'''
        if not os.path.isfile(FLATC_EXE):
            raise RuntimeError('Failed to find flatc')
        if not os.stat(FLATC_EXE).st_mode | stat.S_IXUSR:
            raise RuntimeError('No permission to execute flatc')

        # '--gen-object-api' is necessaary to mutate flatbuffer data
        cmd = [
            './' + FLATC_EXE, '--python', '--gen-onefile', '--gen-object-api', '-o',
            out_dir, schema_path
        ]
        try:
            subprocess.check_call(cmd)
        except Exception as e:
            print("failed to compile using flatc", e)

    def __clear_flatc(self):
        if os.path.exists(FLATC_ZIP):
            os.remove(FLATC_ZIP)
        if os.path.exists(FLATC_EXE):
            os.remove(FLATC_EXE)


def add_commit_id():
    '''Prepend commit id to *_generated.py'''
    commit_id = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    commit_id = commit_id.stdout.decode('utf-8')[:-1]  # remove -\n

    gen_files = os.listdir('./')
    for gen_file in gen_files:
        if '_generated.py' in gen_file:
            with open(gen_file, 'r+') as f:
                data = f.read()
                f.seek(0)
                f.write(f"# generated based on commit({commit_id})\n" + data)


run_flatc = FlatcCaller()
run_flatc([CIRCLE_SCHEMA, TINFO_SCHEMA])
add_commit_id()
