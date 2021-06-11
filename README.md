This repository is a collaboration between RMI and Catalyst Cooperative to connect FERC
Form 1 plant records, EIA plant records and depreciation study records at the most
granular level.

## Installation
To install the software in this repository, clone it to your computer using git. E.g.

```sh
git clone git@github.com:catalyst-cooperative/rmi-ferc1-eia.git
```
Then in the top level directory of the repository, create a `conda` environment based on
the `environment.yml` file that is stored in the repo:

```sh
conda env create --file environment.yml
```

Note that the software in this repository depends on [the dev
branch](https://github.com/catalyst-cooperative/pudl/tree/dev) of the [main PUDL
repository](https://github.com/catalyst-cooperative/pudl), and the `setup.py` in this
repository indicates that it should be installed directly from GitHub. This can be a bit
slow, as `pip` (which in this case is running inside of a `conda` environment) clones
the entire history of the repository containing the package being installed. How long it
takes will depend on the speed of your network connection. It might take ~5 minutes.

The `environment.yml` file also specifies that the Python package defined within this
repository should be installed such that it is editable.  This will allow you to change
the modules that are part of the repository and have the installed software reflect your
changes.

If you want to make changes to the PUDL software as well, you can clone the PUDL
repository into another directory (outside of this repository), and direct `conda` to
install the package from there. A commented out example of how to do this is included
in `environment.yml`. **NOTE:** if you want to install PUDL in editable mode from a
locally cloned repo, you'll need to comment out the dependency in `setup.py` as it may
otherwise conflict with the local installation (pip can't resolve the precedence of
different git based versions).

After any changes to the environment specification, you'll need to recreate the conda
environment. The most reliable way to do that is to remove the old environment and
create it from scratch. If you're in the top level `rmi-ferc1-eia` directory and have
the `pudl-rmi` environment activated, that process would look like this:

```sh
conda deactivate
conda env remove --name pudl-rmi
conda env create --file environment.yml
conda activate pudl-rmi
```

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
