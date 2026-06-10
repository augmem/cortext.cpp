# Contributing to sqlite-objstore

Thank you for your interest in contributing to `sqlite-objstore`! We welcome contributions from everyone.

## How to Contribute

1.  **Fork the repository** and create your branch from `main`.
2.  **Install dependencies**: Ensure you have CMake 3.28+, a C17/C++20 toolchain, and Ninja.
3.  **Make your changes**: Implement your feature or fix.
4.  **Run tests**: Verify your changes with `cmake --preset full-debug`, `cmake --build --preset full-debug`, and `ctest --preset full-debug`. When you are touching crash-prone or memory-sensitive code paths, also run `cmake --preset full-asan`, `cmake --build --preset full-asan`, and `ctest --preset full-asan`.
5.  **Submit a Pull Request**: Open a PR against the `main` branch.

## Code Style

- Follow the existing coding style in the project.
- Ensure no compiler warnings are introduced.
- Add tests for new features or bug fixes.

## Reporting Issues

Please use the [GitHub Issues](https://github.com/augmem/sqlite-objstore/issues) tracker to report bugs or request features.

## Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.
