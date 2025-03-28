# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:jammy

# Install 'add-apt-repository'
RUN apt-get update && apt-get -qqy install software-properties-common
# Install tzdata without user input
RUN DEBIAN_FRONTEND=noninteractive apt-get -qqy install tzdata

# Install gbs
RUN add-apt-repository -y 'deb [trusted=yes] http://download.tizen.org/tools/latest-release/Ubuntu_22.04/ /'
RUN apt-get update && apt-get -qqy install gbs

# Setup user to match host user, and give superuser permissions
ARG USER_ID=1000
ARG GROUP_ID=${USER_ID}
RUN apt-get update && apt-get -qqy install sudo
RUN addgroup --gid ${GROUP_ID} ubuntu && adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} ubuntu && usermod -aG sudo ubuntu
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Clean archives (to reduce image size)
RUN apt-get clean -y

# Set user to the one we just created
USER ${USER_ID}
ENTRYPOINT [ "gbs" ]
