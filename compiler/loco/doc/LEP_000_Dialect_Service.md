# Dialect Service

This loco enhancement proposal (_LEP_) discusses how to permit a _loco_ graph without canonical dialect.

## Revision

| Date | Status |
| ---  | --- |
| 2019/09/03 | Proposed |

## Motivation

One of key design principles behind _loco_ is to allow users (= NN compiler writers) to easily define their own intermediate representation (IR) on top of shared infrastructure.

Unfortunately, however, there is a gap between dream and reality.
It is currently impossible to create a _loco_ graph only with non-canonical dialects;
there is no way to express the interaction between graph-level output without _canonical.Push_ node.

This proposal aims to remove this restriction in order to bridge the gap between dream and reality.

## Design

Each dialect is now allowed to expose its internal to its client (such as transformations and core algorithms) through a so-called "Service" interface.

Although this proposal focuses on ``output_nodes`` helper in _loco.core_, its coverage is not limited to this helper.
Any pass and algorithm can take an advantage of this generic infrastructure.

Let us dive into some details.

### What is "service"?

A service declares a collection of APIs that each **client** (not dialect) needs.

Let us consider ``output_nodes``. ``output_nodes`` needs to check whether a node is associated with any graph-level output.

Here is one possible service design that satisfies this need.
```cxx
virtual bool associated(const Node *node) const = 0;
virtual GraphOutputIndex index(const Node *node) const = 0;
```

### How to declare a service

All of these service interfaces should inherit ``loco::DialectService`` interface that _loco.core_ defines.
```cxx
struct DialectService
{
  virtual ~DialectService() = default;
};
```

For example, it is possible to declare the service that ``output_nodes`` needs as follows:
```cxx
struct GraphOutputIndexQueryService : public DialectService
{
  virtual ~GraphOutputIndexQueryService() = default;

  virtual bool associated(const Node *node) const = 0;
  virtual GraphOutputIndex index(const Node *node) const = 0;
};
```

### How to access a service

This proposal extends ``Dialect`` class with ``service`` method.

Each dialect SHOULD return a valid pointer on ``service<Service>`` method call if it implements that service. Otherwise, it SHOULD return a null pointer.

**WARNING** It is impossible to use ``get``. ``get`` is currently reserved for singleton accessor.

Given a ``GraphOutputIndexQueryService``, it is possible to revise ``output_nodes`` as follows:
```cxx
std::vector<loco::Node *> output_nodes(loco::Graph *g)
{
  std::map<GraphOutputIndex, loco::Node *> table;

  for (uint32_t n = 0; n < g->nodes()->size(); ++n)
  {
    auto node = g->nodes()->at(n);

    if (auto service = node->dialect()->service<GraphOutputIndexQueryService>())
    {
      if (service->associated(node))
      {
        auto output_index = service->index(node);
        assert(table.find(output_index) == table.end());
        table[output_index] = node;
      }
    }
  }

  std::vector<loco::Node *> res;

  for (uint32_t n = 0; n < g->outputs()->size(); ++n)
  {
    auto it = table.find(n);
    // NOTE This behavior originates from the current implementation of output_nodes
    res.emplace_back(it == table.end() ? nullptr : it->second);
  }

  return res;
}
```

**PLEASE NOTE THAT** ``output_nodes`` now works with all the dialects that implement ``GraphOutputIndexQueryService``.

### How to register a service

Each dialect should invoke the protected ``service`` method during its construction.
```cxx
AwesomeDialect::AwesomeDialect()
{
  std::unique_ptr<Impl> impl = ...;
  service<GraphOutputIndexQueryService>(std::move(impl));
}
```
