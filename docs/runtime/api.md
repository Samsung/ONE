# API

## Runtime Layered Architecture

Here is a figure of runtime layered architecture.

![Layered Architecture](api-layered-arch.png)

There are three parts - Frontend, Core and Backend. Core works with Frontend and Backend API. Frontend gets user inputs(neural networks models) and Backend does the actual computation.

## Frontend API

Frontend API is about from creation/loading the model and

Runtime supports NNFW API.

### NNFW API

NNFW API is ONE's own API. It supports loading models from NN Packages. As it is our own API, It can do most of functionalities that One Runtime offers. Representatively, it provides functions for execution with multiple backends.

For usage, refer to [Howto : NNFW API](../howto/how-to-use-nnfw-api.md).

## Backend API

Backend API enables anyone to extend the runtime in terms of operation computation and memory management.

For detailed descriptions, refer to [Backend API](../runtime/backend-api.md).
