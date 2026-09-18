# Third-Party Notices

This file lists third-party components whose source is included in this
repository (under `third_party/`), along with their licenses. The PSuM source
code itself is licensed under the MIT License (see [`LICENSE`](./LICENSE)).
Each third-party component retains its original license; the terms below apply
only to those components, not to PSuM.

## Vendored Components

| Component | Source | License | License Text |
|-----------|--------|---------|--------------|
| nlohmann/json | <https://github.com/nlohmann/json> | MIT | SPDX header in [`third_party/json.hpp`](./third_party/json.hpp) |

### Notes

- **nlohmann/json** is distributed under the MIT License. Copyright © 2013-2023
  Niels Lohmann. The copyright and license are declared in the SPDX header at
  the top of `third_party/json.hpp`; the full license text is reproduced below
  so that source-archive consumers can obtain it without network access.

**MIT License (nlohmann/json)**

Copyright (c) 2013-2023 Niels Lohmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Bundled Assets

The `test/particle_boundary/` directory contains test-only STL meshes derived from
CC0 / public-domain sources. See
[`test/particle_boundary/README.md`](./test/particle_boundary/README.md) for the file
mapping and sources.

## Runtime Dependencies

PSuM also links against additional libraries during the build (for example,
the SYCL implementation, linear algebra backends, and optional GPU toolkits).
These are **not** part of this repository; users must obtain and install them
separately. Please refer to the respective upstream projects for their current
license terms.
