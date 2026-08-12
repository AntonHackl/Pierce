# Third-Party Notices

This repository contains third-party material.  The notices below identify
the material that is copied into, modified in, or distributed with Pierce.
They supplement, and do not replace, the license notices in the source files.

## TDBase

`baselines/tdbase/` is the upstream TDBase source from
[`tengdj/tdbase`](https://github.com/tengdj/tdbase), pinned at commit
`5058e2f540438a497cd0592b9044e0bcbd745cbb`.

TDBase is distributed under the GNU General Public License, version 3.  Its
unaltered license text is retained at `baselines/tdbase/LICENSE`.  Pierce
contains modified TDBase-derived code in both
`baselines/tdbase_extensions/` and `pierce/preprocess/src/tdbase_lib/`.  The
latter is a copied and modified subset linked into `pierce_preprocess` for
loading and decoding TDBase `.dt` data.

## Material distributed within TDBase

- `popl.h` is the Program Options Parser Library (popl), version 1.3.0,
  Copyright (C) 2015-2021 Johannes Pohl, distributed under the MIT License.
  Its original notice is retained in the file; the MIT license text is
  included at `LICENSES/MIT.txt`.
- `computing_cpu.cpp` contains material Copyright 1999 The University of
  North Carolina at Chapel Hill.  Its educational, research, and non-profit
  permission notice and warranty disclaimer are retained verbatim in the
  source file.
- `himesh.h`, `himesh_IO.cpp`, `himesh_comp.cpp`, and `himesh_decomp.cpp`
  retain the GPL notices for PPMC by Adrien Maglo and Clément Courbet.

All original copyright and license notices in these files must remain intact.
