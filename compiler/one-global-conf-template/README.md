# ONE Global Target Configuration Template

This folder provides a reference template (one-global-conf-template) for building and
 distributing global target configuration packages for the ONE Compiler.

A one-global-conf-* package supplies:

- **Target configuration** (${TARGET}.ini) describing core/memory/frequency/bandwidth.
- **Backend command schemas** (codegen.py, profile.py, …) defining the CLI interface of
 backend drivers in a key-value style for onecc.

By following this template, anyone can create and install a new target configuration
 package that integrates seamlessly with onecc.

## Repository Layout

```
one-global-conf-{ARCH_FAMILY_NAME}/
├── backend/
│   └── {BACKEND_NAME}/
│       ├── bin/ (as an example)
│       │   ├── {BACKEND_NAME}-compile
│       │   └── {BACKEND_NAME}-profile
│       └── one-cmds/
│           ├── codegen.py
│           └── profile.py
├── target/
│   └── {TARGET_NAME}/
│       └── {TARGET_NAME}.ini
├── debian/
│   ├── one-global-conf-{TARGET_NAME}.install
│   ├── control
│   └── rules
└── tools/
    └── validate_global_conf.py
```

Replace:
- {TARGET_NAME} → Your target codename (e.g., Rose).
- {BACKEND_NAME} → Your backend name (e.g., dummy).
- {ARCH_FAMILY_NAME} → The architecture family (e.g., TRIV).

## Installation Paths

When installed via .deb package, files are placed as follows:

- Target INI
  - /usr/share/one/target/{TARGET_NAME}.ini

- Backend command schemas
  - /usr/share/one/backends/command/{BACKEND_NAME}/codegen.py
  - /usr/share/one/backends/command/{BACKEND_NAME}/profile.py

- Optional backend drivers
  - /usr/share/one/backends/{BACKEND_NAME}/bin/{BACKEND_NAME}-compile
  - /usr/share/one/backends/{BACKEND_NAME}/bin/{BACKEND_NAME}-profile

## Required Files

1. Target configuration (target/{TARGET_NAME}/{TARGET_NAME}.ini)

Defines TARGET, BACKEND, ARCHITECTURE, memory sizes, frequencies, and bandwidths.

2. Command schemas (backend/{BACKEND_NAME}/one-cmds/*.py)

Provide a command_schema() function that exposes backend driver options to onecc in a key-value style.

- codegen.py: schema for [one-codegen] section.
- profile.py: schema for [one-profile] section.

3. Debian packaging files

- debian/one-global-conf-{TARGET_NAME}.install: installation mapping.
- debian/control, debian/rules: minimal packaging files.

## Example Workflow

### Example example.ini

```ini
[backend]
target=Rose

[one-optimize]
input_path=model.circle
output_path=model.opt.circle

[one-codegen]
input_path=model.opt.circle
output_path=out.tvn
verbose=True
```

### Resolved command

When invoked with onecc, the above translates to:

```bash
dummy-compile --target Rose --verbose --output out.tvn model.opt.circle
```

## Validation

Run validation before packaging or after installation:

### Source-tree validation:

```
python tools/validate_global_conf.py --root . --target Rose --backend dummy
```

### Installed validation:

```
python tools/validate_global_conf.py --installed --target Rose --backend dummy
```

## How to Create Your Own one-global-conf-*

1. Copy this template.
2. Replace {TARGET_NAME}, {BACKEND_NAME}, {ARCH_FAMILY_NAME} with your values.
3. Fill in:
- Target INI with your hardware specs.
- Command schema files with your backend driver arguments.
4. Update debian/ metadata.
5. Build and install the .deb package.
