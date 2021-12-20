#!/bin/bash

PYMOR_ROOT="$(cd "$(dirname "$0")" && cd ../../ && pwd -P )"
cd "${PYMOR_ROOT}"

set -eux

PYTHONS="${1}"
# make sure CI setup is current
./.ci/gitlab/template.ci.py && git diff --exit-code .ci/gitlab/ci.yml
# check if requirements files are up-to-date
./dependencies.py && git diff --exit-code requirements* pyproject.toml

# performs the image+tag in registry check
./.ci/gitlab/template.ci.py "${GITLAB_API_RO}"

# makes sure mailmap is up-to-date
./.ci/check_mailmap.py ./.mailmap
