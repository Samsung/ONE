# Deprecated Compiler Projects

This directory contains compiler projects that are no longer actively maintained or used in production. These projects have been moved here from the main `compiler/` directory to preserve historical context while keeping the active codebase clean and maintainable.

### Purpose

The `_DEPRECATED` directory serves as a repository for projects that:
- Have been superseded by newer implementations or alternatives
- Are no longer supported by the maintenance team
- Have been archived for historical reference
- Contain functionality that is no longer required by the project

### Moving Projects to This Directory

A project should be moved to this directory when:

1. **No Active Development**: The project has not received active development or maintenance for an extended period.
2. **No Active Users**: There are no known users or workflows relying on the project.
3. **Better Alternatives Exist**: The project's functionality has been replaced by a more suitable alternative in the active codebase.

### Maintenance Policy

Projects in this directory are **not actively maintained**. This means:

- Bug fixes will not be applied
- New features will not be added
- Compatibility updates will not be made
- Documentation will not be updated
- Security patches will not be provided

### Removal Policy

Projects in this directory may be **permanently removed** from the repository after **one year** from their deprecation date. Note that the code will remain accessible through Git history for reference purposes.

### Accessing Deprecated Projects

Projects in this directory remain in the repository for:
- Historical reference and audit purposes
- Potential reactivation if business requirements change
- Understanding the evolution of the codebase

If you need to use a deprecated project:
1. **Evaluate Alternatives**: Check if there is a maintained alternative that fulfills your requirements.
2. **Contact Maintainers**: Discuss with the project maintainers before depending on deprecated code.
3. **Accept Risks**: Be aware that deprecated projects may have security vulnerabilities, compatibility issues, or lack of support.

### List of Deprecated Projects

This section will be updated as projects are moved to this directory.

| Project | Deprecation Date | Reason |
|---------|------------------|--------|
| caffe2circle | 2026-03-13 | Caffe models are no longer supported |
| enco | 2026-03-16 | We don't longer maintain Android NN API backend any more |
| enco-intf | 2026-03-16 | We don't maintain `enco` any more |
| encodump | 2026-03-16 | We don't maintain `enco` any more |
| onnx2circle | 2026-03-16 | Use circle-mlir's onnx2circle tool |
| onnx2tflite | 2026-03-16 | onnx2tflite + tflite2circle is replaced as circle-mlir's onnx2circle tool |
| onnx2tflite-integration-test | 2026-03-16 | onnx2tflite is no longer used |
| tf2tflite | 2026-03-18 | Use TF's official tool |
| tf2tflite-dredd-pb-test | 2026-03-18 | `tf2tflite` is deprecated |
| tf2tflite-dredd-pbtxt-test | 2026-03-18 | `tf2tflite` is deprecated |
| tf2tflite-value-pb-test | 2026-03-18 | `tf2tflite` is deprecated |
| tf2tflite-value-pbtxt-test | 2026-03-18 | `tf2tflite` is deprecated |
| tf2circle | 2026-03-18 | Replaced to use official TFLite converter then use tflite2circle |
| tf2circle-conversion-test | 2026-03-18 | `tf2circle` is deprecated |
| tf2circle-dredd-pb-test | 2026-03-18 | `tf2circle` is deprecated |
| tf2circle-dredd-pbtxt-test | 2026-03-18 | `tf2circle` is deprecated |
| tf2circle-model-test | 2026-03-18 | `tf2circle` is deprecated |
| tf2circle-ui-check | 2026-03-18 | `tf2circle` is deprecated |
| tf2circle-value-pbtxt-remoe-test | 2026-03-18 | `tf2circle` is deprecated |
| moco | 2026-03-19 | TF model uses TF's official tools, convert to circle model by TF's TFLite converter and tflite2circle |
| moco-tf | 2026-03-19 | `moco` is deprecated |
| moco-value-pbtxt-test | 2026-03-19 | `moco-tf` is deprecated |
| nnkit-mocotf | 2026-03-19 | `moco-tf` is deprecated |
| tf2nnpkg | 2026-03-19 | Replaced by tf2nnpkg script `infra/packaging/res/tf2nnpkg*` |
| ann-api | 2026-03-24 | `enco` deprecated: No longer maintain Android NN API backend |
| ann-ref | 2026-03-24 | `enco` and `ann-ref` deprecated |
