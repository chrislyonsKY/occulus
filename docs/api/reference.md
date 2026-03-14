# API Reference

Complete reference for all public functions and classes in `occulus`.

---

## Top-Level API

::: occulus
    options:
      members:
        - read
        - write
        - PointCloud
        - AerialCloud
        - TerrestrialCloud
        - UAVCloud
        - Platform
        - ScanPosition
        - AcquisitionMetadata
      show_root_heading: false
      show_source: false

---

## Exceptions

::: occulus.exceptions
    options:
      members:
        - OcculusError
        - OcculusIOError
        - OcculusValidationError
        - OcculusRegistrationError
        - OcculusSegmentationError
        - OcculusMeshError
        - OcculusFeatureError
        - OcculusCppError
        - OcculusNetworkError
        - UnsupportedPlatformError
      show_source: false

---

## Modules

### I/O

::: occulus.io
    options:
      members:
        - read
        - write
      show_source: true

### Filters

::: occulus.filters
    options:
      show_source: true

### Normal Estimation

::: occulus.normals
    options:
      show_source: true

### Registration

::: occulus.registration
    options:
      show_source: true

### Segmentation

::: occulus.segmentation
    options:
      show_source: true

### Meshing

::: occulus.mesh
    options:
      show_source: true

### Features

::: occulus.features
    options:
      show_source: true

### Metrics

::: occulus.metrics
    options:
      show_source: true

### Visualization

::: occulus.viz
    options:
      show_source: true
