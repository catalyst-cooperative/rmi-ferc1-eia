This repository is a collaboration between RMI and Catalyst Cooperative to connect FERC
Form 1 plant records, EIA plant records and depreciation study records at the most
granular level.

## Installation
To install the software in this repository, clone it to your computer and from the top
level directory of the repository run:

```py
pip install --editable ./
```

This will allow you to edit the modules that are part of the repository and have the
installed software reflect your changes. If you just want to install the software as it
is in the repository, and use it to process the data without editing the code, you can
use `pip` within a conda or other virtual Python environment:

```sh
python -m pip install git+https://github.com/catalyst-cooperative/rmi-ferc1-eia.git
```

The software in this repository depends on [the dev
branch](https://github.com/catalyst-cooperative/pudl/tree/dev) of the [main PUDL
repository](https://github.com/catalyst-cooperative/pudl), and will install it directly
from GitHub. This can be a bit slow, as `pip` clones the entire history of the
repository containing the package being installed. How long it takes will depend on the
speed of your network connection. It could be a few minutes.

In order to use this repository, you will need a recent copy of the PUDL database. You
You can either create one for yourself by [running the ETL
pipeline](https://catalystcoop-pudl.readthedocs.io/en/latest/dev/run_the_etl.html), or
you can follow the instructions in the [PUDL examples
repository](https://github.com/catalyst-cooperative/pudl-examples) to download the
processed data alongside a Docker container.

## Process Overview
Below is a visual overview of the main processes in this repo:
![Design overview:](https://github.com/catalyst-cooperative/rmi-ferc1-eia/blob/master/rmi_design.png?raw=true)

Each of the outputs shown above have a dedicated module:
* EIA Master Unit List: [`make_plant_parts_eia.py`](https://github.com/catalyst-cooperative/rmi-ferc1-eia/blob/master/make_plant_parts_eia.py)
* EIA & Depreciation Connected: [`connect_deprish_to_eia.py`](https://github.com/catalyst-cooperative/rmi-ferc1-eia/blob/master/connect_deprish_to_eia.py)
* EIA & FERC Connected: [`connect_ferc1_to_eia.py`](https://github.com/catalyst-cooperative/rmi-ferc1-eia/blob/master/connect_ferc1_to_eia.py)
* Connected Depreciation & FERC: [`connect_ferc1_to_eia.py`](https://github.com/catalyst-cooperative/rmi-ferc1-eia/blob/master/connect_ferc1_to_eia.py)

RMI Collaborators include:
 * @UdayVaradarajan
 * @SamMardell
