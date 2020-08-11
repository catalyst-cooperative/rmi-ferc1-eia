This repository is a collaboration between RMI and Catalyst Cooperative to connect FERC Form 1 plant records, EIA plant records and depreciation study records at the most granular level.

In order to use this repository, you will need an [updated PUDL database](https://catalystcoop-pudl.readthedocs.io/en/latest/usage.html) and run all scripts and notebooks inside a [pudl-dev conda environment](https://catalystcoop-pudl.readthedocs.io/en/latest/dev_setup.html#create-and-activate-the-pudl-dev-conda-environment).

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
