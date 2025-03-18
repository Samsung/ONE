# Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

import os
import nox

# Define the minimal nox version required to run
nox.needs_version = ">= 2024.3.2"


@nox.session
def build_and_check_dists(session):
    session.install("build", "check-manifest >= 0.42", "twine", "ai-edge-model-explorer",
                    "flatbuffers")

    session.run("check-manifest", "--ignore", "noxfile.py,tests/**")
    session.run("python", "-m", "build")
    session.run("python", "-m", "twine", "check", "dist/*")


@nox.session(python=["3.10"])
def tests(session):
    session.install("pytest")

    build_and_check_dists(session)

    generated_files = os.listdir("dist/")
    # Sort files in descending order according to the version
    generated_files.sort(
        key=lambda x: [int(i, 10) for i in x.split('-')[1].split('.')[:3]], reverse=True)
    # The sdist file 'model_explorer_circle.x.x.x.tar.gz' would be used
    generated_sdist = os.path.join("dist/", generated_files[0])

    session.install(generated_sdist)

    session.run("py.test", "tests/", *session.posargs)
