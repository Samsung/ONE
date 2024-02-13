# Background

Artificial intelligence (AI) techniques are getting popular and utilized in various products and
services. While the cloud-based AI techniques have been used to perform compute/memory intensive
inferences because of the powerful servers on cloud, on-device AI technologies are recently drawing
attention from the mobile industry for response time reduction, privacy protection, and
connection-less AI service. Big mobile players are investing their research effort in the on-device
AI technologies and have already announced hardware and software on-device AI solutions. We are not
leading this trend currently, but since on-device AI area has just started and remains in the initial
stage, there are still opportunities and possibilities to reduce the gap between pioneers and us. We
believe that on-device AI will become a key differentiator for mobile phones, TV, and other home
appliances. Therefore, developing on-device AI software stack is of paramount importance in order to
take leadership in the on-device AI technology.

Although the vision of on-device AI is promising, enabling on-device AI involves unique technical
challenges compared to traditional cloud-based approach. This is because on-device AI tries to
conduct inference tasks solely on device without connecting to cloud resources. Specifically,
hardware resources on device, such as processor performance, memory capacity and power budget are
very scarce and limit the compute capability, which is typically required to execute complex
neural network (NN) models. For example, in one product requirement, a mobile device should consume
less than 1.2W and could use at most 2W only for 10 minutes due to thermal constraints. On-device AI
software stack needs to support diverse device environments, since embedded platforms may consist of
heterogeneous compute devices, such as CPU, GPU, DSP, or neural processing unit (NPU), and use
different OS platforms, such as Tizen, Android, or various Linux systems.

To tackle the challenges above and to have the leadership in on-device AI technology, this project,
as the first step, aims at developing a neural network inference framework specialized and optimized
for on-device AI.
