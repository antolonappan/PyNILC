![pytest](https://github.com/antolonappan/PyNILC/actions/workflows/package.yml/badge.svg)

# PyNILC

Needlet ILC

powered by ducc0 and multi-threading

# Installation

```bash
git clone https://github.com/antolonappan/PyNILC
cd PyNILC
pip install -e .

```

# Usage

```python
from pynilc.nilc import NILC
from pynilc.sht import HealpixDUCC
from pynilc.needlets import NeedletTransform
```

refer to the `/example` for details

# Parallelization benchmark

Test is based on NSIDE=1024, needlet bands = 10, frequency channels = 6

<img src="https://github.com/echoCMB/PyNILC/blob/main/docs/scaling.png" alt="Scaling Example" width="300"/>
