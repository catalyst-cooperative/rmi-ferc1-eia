![tox-pytest](https://github.com/catalyst-cooperative/rmi-ferc1-eia/actions/workflows/tox-pytest.yml/badge.svg)
![codecov](https://img.shields.io/codecov/c/github/catalyst-cooperative/rmi-ferc1-eia)
![code style](https://img.shields.io/badge/code%20style-black-000000.svg)

This repository is a collaboration between RMI and Catalyst Cooperative to connect FERC
Form 1 plant records, EIA plant records and depreciation study records at the most
granular level.

## Installation

To install the software in this repository, clone it to your computer using git. If
you're authenticating using SSH:

```sh
git clone git@github.com:catalyst-cooperative/rmi-ferc1-eia.git
```

Or if you're authenticating via HTTPS:

```sh
git clone https://github.com/catalyst-cooperative/rmi-ferc1-eia.git
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
pre-processed data alongside a Docker container.

To work with the pre-processed data **outside** of the Docker container, you will need
to tell the PUDL software where to find that data on your computer. When you extract the
pre-processed data archive, it will include a directory named `pudl_data` -- you need to
put the path to that directory in a file called `.pudl.yml` in your home directory. The
contents will need to look like the following (but with real paths...):

```yml
pudl_in: /path/to/your/downloaded/pudl_data
pudl_out: /the/same/path/to/pudl_data
```

**NOTE:** If you get to a point where you need or want to run the PUDL ETL for yourself,
you will need to reset these paths to another location so that you don't accidentally
overwrite the pre-processed data.

* If you're unfamiliar with file paths, directories, and the command line in general, we
  recommend checking out
  [The Basics of the Unix Shell](https://merely-useful.tech/py-rse/bash-basics.html) from
  [Research Software Engineering in Python](https://merely-useful.tech/py-rse/index.html).
* If you'd like more background on reproducible software environments, including
  software package managers and the role played by containerization systems like
  Docker, check out the chapter on
  [Reproducible Research](https://the-turing-way.netlify.app/reproducible-research/renv.html)
  from [The Turing Way](https://the-turing-way.netlify.app/welcome.html).

## Tests

This repo finally has some tests! wahoo! Unfortunately, there are memory issues
getting in the way of letting us run all of the tests via github actions
([PUDL issue #1457](https://github.com/catalyst-cooperative/pudl/issues/1457)).

### Regenerate All Outputs & Validate

``sh
pytest test/integration/rmi_out_test.py
``

### Validate Existing Outputs

If you have recently processed output cached in the output directory
(`pudl_rmi.OUTPUTS_DIR`) and just want to test the consistency of the outputs,
there is a quick test to run. This test checks whether the processing of the
data has or has not introduced errors. There are known errors being stored in
the input directory (`pudl_rmi.INPUTS_DIR`). We expect most of these error exist
because of missing connections between datasets.

``sh
pytest test/integration/rmi_out_test.py::test_consistency_of_data_stages
``


## Process Overview

Below is a visual overview of the main processes in this repo:
![Design overview:](https://github.com/catalyst-cooperative/rmi-ferc1-eia/blob/main/rmi_design.png?raw=true)

Each of the outputs shown above have a dedicated module:

* EIA Master Unit List: [`make_plant_parts_eia.py`](https://github.com/catalyst-cooperative/rmi-ferc1-eia/blob/master/make_plant_parts_eia.py)
* EIA & Depreciation Connected: [`connect_deprish_to_eia.py`](https://github.com/catalyst-cooperative/rmi-ferc1-eia/blob/master/connect_deprish_to_eia.py)
* EIA & FERC Connected: [`connect_ferc1_to_eia.py`](https://github.com/catalyst-cooperative/rmi-ferc1-eia/blob/master/connect_ferc1_to_eia.py)
* Connected Depreciation & FERC: [`connect_ferc1_to_eia.py`](https://github.com/catalyst-cooperative/rmi-ferc1-eia/blob/master/connect_ferc1_to_eia.py)
