#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

pip install git+https://github.com/numpy/numpy@main
# there seems to be no way of really overwriting -p no:warnings from setup.cfg
sed -i -e 's/\-p\ no\:warnings//g' setup.cfg
xvfb-run -a py.test ${COMMON_PYTEST_OPTS} -W once::DeprecationWarning -W once::PendingDeprecationWarning
coverage xml
