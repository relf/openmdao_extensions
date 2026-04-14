# OpenMDAO Extensions

[![Tests](https://github.com/OneraHub/openmdao_extensions/workflows/Tests/badge.svg)](https://github.com/OneraHub/openmdao_extensions/actions?query=workflow%3ATests)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/54f346b4094a42f081c4f674e2990aee)](https://app.codacy.com/gh/whatsopt/openmdao_extensions/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v0.json)](https://github.com/charliermarsh/ruff)

Set of specialized classes to handle specific methods using OpenMDAO framework.

* `EgoboxEgorDriver` : an OpenMDAO driver for Egor optimizer from [EGObox](https://github.com/relf/egobox#egobox) library
* `EgoboxDOEDriver` : an OpenMDAO driver for Mixint LHS DoE from [EGObox](https://github.com/relf/egobox#egobox) library
* `OneraSegoDriver` : an OpenMDAO driver for Onera Super EGO optimizers
* `SmtDOEDriver` : an OpenMDAO driver for sampling methods from surrogate [SMT](https://smt.readthedocs.io/en/latest/) library
* `SalibDOEDriver` : an OpenMDAO driver for Morris or Saltelli DoE from sensitive analysis [SALib](https://salib.readthedocs.io/en/latest/) library
* `OpenturnsDoeDriver` : an OpenMDAO driver for DoE following distributions from [OpenTURNS](http://www.openturns.org/) library
* `RecklessNonlinearBlockGS` : a specialized version of the `NonlinearBlockGS` solver to select variables used in the convergence criterion.

## Installation

```bash
pip install openmdao-extensions
```

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
