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

## Scaffold your project from scratch

- **STEP01**: Install PyScaffold and pyscaffoldext-markdown extension

     - You can install PyScaffold and extensions globally in your systems but ins recomendes use a virtual environment:

          Craate a temp folder and use a virtual environment to install PyScaffold tool and scaffold your project. Later will copy the results under the final git folder and remove the last temporal one:

          ```
          $ mkdir temp
          $ cd temp
          $ python3 -m venv .venv
          $ source .venv/bin/activate
          $ pip install pyscaffold
          $ pip install pyscaffoldext-markdown
          $ putup --markdown uniovi-simur-wearablepermed-hmc -p wearablepermed_hmc \
               -d "Uniovi Simur WearablePerMed HMC." \
               -u https://github.com/Simur-project/uniovi-simur-wearablepermed-hmc.git
          $ deactivate               
          ```

     - Also you can install **pyscaffold** and **pyscaffoldext-markdown** packages in your system and avoid the error from Python 3.11+: ```"error-externally-managed-environment" this environemnt is externally managed``` you can execute this command to force instalation:

          ```
          $ pip3 install pyscaffold --break-system-packages
          $ pip3 install pyscaffoldext-markdown --break-system-packages
          $ putup --markdown uniovi-simur-wearablepermed-hmc -p wearablepermed_hmc \
               -d "Uniovi Simur WearablePerMed HMC." \
               -u https://github.com/Simur-project/uniovi-simur-wearablepermed-hmc.git
          ```

          or permanent configure pip3 with this command to avoid the previous errors from 3.11+

          ```
          $ python3 -m pip config set global.break-system-packages true
          ```

- **STEP02**: creare your repo under SIMUR Organization with the name **uniovi-simur-wearablepermed-hmc** and clone the previous scaffoled project

     ```
     $ cd git
     $ git clone https://github.com/Simur-project/uniovi-simur-wearablepermed-hmc.git
     ```

- **STEP03**: copy PyScaffold project to your git folder without .venv folder

- **STEP04**: install tox project manager used by PyScaffold. Install project dependencies
     ```
     $ python3 -m venv .venv
     $ source .venv/bin/activate
     $ pip install tox
     $ pip install pandas
     $ tox list
     ```

## Start develop your project
- **STEP01**: Clone your project
     ```
     $ git clone https://github.com/Simur-project/uniovi-simur-wearablepermed-hmc.git
     ```

- **STEP01**: Build and Debug your project
     ```
     $ tox list
     default environments:
     default   -> Invoke pytest to run automated tests

     additional environments:
     build     -> Build the package in isolation according to PEP517, see https://github.com/pypa/build
     clean     -> Remove old distribution files and temporary build artifacts (./build and ./dist)
     docs      -> Invoke sphinx-build to build the docs
     doctests  -> Invoke sphinx-build to run doctests
     linkcheck -> Check for broken links in the documentation
     publish   -> Publish the package you have been developing to a package index server. By default, it uses testpypi. If you really want to publish your package to be publicly accessible in PyPI, use the `-- --repository pypi` option
     ```

     ```
     $ tox -e clean
     $ tox -e build
     $ tox -e docs
     ```

- **STEP02 Build service**
     ```
     $ docker build -t uniovi-simur-wearablepermed-hmc:1.0.0 .
     ```

- **STEP03: Tag service**
     ```
     $ docker tag uniovi-simur-wearablepermed-hmc:1.0.0 ofertoio/uniovi-simur-wearablepermed-hmc:1.0.0
     ```

- **STEP04: Publish service**
     ```
     $ docker logout
     $ docker login
     $ docker push ofertoio/uniovi-simur-wearablepermed-hmc:1.0.0
     ```

- **STEP04: Start service**     
     Set your bin files under Linux **/home/miguel/temp/simur** folder and execute Docker service from **Ubuntu** or **Mac**:

     ```
     $ docker run \
     --rm \
     -v /home/miguel/temp/simur:/app/data \
     ofertoio/uniovi-simur-wearablepermed-hmc:1.0.0 \
     python converter.py --bin-file data/MATA00.BIN --csv-file data/MATA00.xlsx
     ```

     Set your bin files under Windows **c:\Temp\simur** folder and execute Docker service from **Windows** using WSL2 (Ubunut 22.02): 

     ```
     $ docker run \
     --rm \
     -v /mnt/c/Temp/simur:/app/data \
     ofertoio/uniovi-simur-wearablepermed-hmc:1.0.0 \
     python converter.py --bin-file data/MATA00.BIN --csv-file data/MATA00.xlsx
     ```

<!-- pyscaffold-notes -->

## Note
This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
