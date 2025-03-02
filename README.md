<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/uniovi-simur-wearablepermed-hmc.svg?branch=main)](https://cirrus-ci.com/github/<USER>/uniovi-simur-wearablepermed-hmc)
[![ReadTheDocs](https://readthedocs.org/projects/uniovi-simur-wearablepermed-hmc/badge/?version=latest)](https://uniovi-simur-wearablepermed-hmc.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/uniovi-simur-wearablepermed-hmc/main.svg)](https://coveralls.io/r/<USER>/uniovi-simur-wearablepermed-hmc)
[![PyPI-Server](https://img.shields.io/pypi/v/uniovi-simur-wearablepermed-hmc.svg)](https://pypi.org/project/uniovi-simur-wearablepermed-hmc/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/uniovi-simur-wearablepermed-hmc.svg)](https://anaconda.org/conda-forge/uniovi-simur-wearablepermed-hmc)
[![Monthly Downloads](https://pepy.tech/badge/uniovi-simur-wearablepermed-hmc/month)](https://pepy.tech/project/uniovi-simur-wearablepermed-hmc)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/uniovi-simur-wearablepermed-hmc)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# uniovi-simur-wearablepermed-hmc

> Uniovi Simur WearablePerMed HMC.

## Scaffold your project

- **STEP01**: Install PyScaffold and pyscaffoldext-markdown extension

     - You can install PyScaffold and extensions using a virtual environment:

          Craate a temp folder and use a virtual environment to install PyScaffold tool and scaffold your project. Later will copy the results under the git folder and remove the last temporal folder

          ```
          $ mkdir temp
          $ cd temp
          $ python3 -m venv .venv
          $ source .venv/bin/activate
          $ pip install pyscaffold
          $ pip install pyscaffoldext-markdown
          $ deactivate
          $ source .venv/bin/activate
          $ putup --markdown uniovi-simur-wearablepermed-hmc -p wearablepermed_hmc \
               -d "Uniovi Simur WearablePerMed HMC." \
               -u https://github.com/Simur-project/uniovi-simur-wearablepermed-hmc.git
          ```

     - Also you can install **pyscaffold** and **pyscaffoldext-markdown** packages in your system and avoid the error from Python 3.11+: ```"error-externally-managed-environment" this environemnt is externally managed``` you can execute this command to force instalation:

          ```
          $ pip3 install pyscaffold --break-system-packages
          $ pip3 install pyscaffoldext-markdown --break-system-packages
          $ putup --markdown uniovi-simur-wearablepermed-hmc -p wearablepermed_hmc \
               -d "Uniovi Simur WearablePerMed HMC." \
               -u https://github.com/Simur-project/uniovi-simur-wearablepermed-hmc.git
          ```

          or permanent configure pip3 with this command:

          ```
          $ python3 -m pip config set global.break-system-packages true
          ```

- **STEP02**: creare your repo under SIMUR Organization with the name **uniovi-simur-wearablepermed-hmc** and clone

     ```
     $ cd git
     $ git clone https://github.com/Simur-project/uniovi-simur-wearablepermed-hmc.git
     ```

- **STEP03**: copy PyScaffold project to your git folder without .venv folder

- **STEP04**: install tox project manager used by PyScaffold
     ```
     $ python3 -m venv .venv
     $ source .venv/bin/activate
     $ pip install tox
     $ tox list
     ```

## Start develop your project

## Start service

Start service from Docker

```
$ docker run --rm uniovi-simur-wearablepermed-hmc:1.0.0 python src/wearablepermed_hmc/converter.py --bin_file ./data/PMP1020_W1_PI.BIN --csv_file ./data/PMP1020_W1_PI.CSV
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
