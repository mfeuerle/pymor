#!/usr/bin/env python3

tpl = '''# THIS FILE IS AUTOGENERATED -- DO NOT EDIT #
#   Edit and Re-run .ci/gitlab/template.ci.py instead       #

stages:
  - sanity
  - test
  - build
  - install_checks
  - deploy

#************ definition of base jobs *********************************************************************************#

.test_base:
    retry:
        max: 2
        when:
            - runner_system_failure
            - stuck_or_timeout_failure
            - api_failure
    tags:
      - autoscaling
    rules:
        - if: $CI_COMMIT_REF_NAME =~ /^staging.*/
          when: never
        - when: on_success
    variables:
        PYPI_MIRROR_TAG: {{pypi_mirror_tag}}
        CI_IMAGE_TAG: {{ci_image_tag}}
        PYMOR_HYPOTHESIS_PROFILE: ci

.pytest:
    extends: .test_base
    tags:
      - long execution time 
      - autoscaling
    environment:
        name: unsafe
    stage: test
    after_script:
      - .ci/gitlab/after_script.bash
    cache:
        key: same_db_on_all_runners
        paths:
          - .hypothesis
    artifacts:
        name: "$CI_JOB_STAGE-$CI_COMMIT_REF_SLUG"
        expire_in: 3 months
        paths:
            - src/pymortests/testdata/check_results/*/*_changed
            - coverage*
            - memory_usage.txt
            - .hypothesis
        reports:
            junit: test_results*.xml

{# note: only Vanilla and numpy runs generate coverage or test_results so we can skip others entirely here #}
.submit:
    extends: .test_base
    image: {{registry}}/pymor/ci_sanity:{{ci_image_tag}}
    variables:
        XDG_CACHE_DIR: /tmp
    retry:
        max: 2
        when:
            - always
    environment:
        name: safe
    rules:
        - if: $CI_COMMIT_REF_NAME =~ /^github\/PR_.*/
          when: never
        - if: $CI_PIPELINE_SOURCE == "schedule"
          when: never
        - when: on_success
    stage: deploy
    script: .ci/gitlab/submit.bash

.docker-in-docker:
    tags:
      - docker-in-docker
      - autoscaling
    extends: .test_base
    timeout: 45 minutes
    retry:
        max: 2
        when:
            - runner_system_failure
            - stuck_or_timeout_failure
            - api_failure
            - unknown_failure
            - job_execution_timeout
    {# this is intentionally NOT moving with CI_IMAGE_TAG #}
    image: {{registry}}/pymor/docker-in-docker:d1b5ebb4dc42a77cae82411da2e503a88bb8fb3a
    variables:
        DOCKER_HOST: tcp://docker:2375/
        DOCKER_DRIVER: overlay2
    before_script:
        - 'export SHARED_PATH="${CI_PROJECT_DIR}/shared"'
        - mkdir -p ${SHARED_PATH}
    services:
        - docker:dind
    environment:
        name: unsafe


# this should ensure binderhubs can still build a runnable image from our repo
.binder:
    extends: .docker-in-docker
    stage: install_checks
    needs: ["ci setup"]
    rules:
        - if: $CI_PIPELINE_SOURCE == "schedule"
          when: never
        - when: on_success
    variables:
        IMAGE: ${CI_REGISTRY_IMAGE}/binder:${CI_COMMIT_REF_SLUG}
        CMD: "jupyter nbconvert --to notebook --execute /pymor/.ci/ci_dummy.ipynb"
        USER: juno

.wheel:
    extends: .docker-in-docker
    stage: build
    needs: ["ci setup"]
    rules:
        - if: $CI_PIPELINE_SOURCE == "schedule"
          when: never
        - when: on_success


.check_wheel:
    extends: .test_base
    stage: install_checks
    rules:
        - if: $CI_PIPELINE_SOURCE == "schedule"
          when: never
        - when: on_success
    services:
      - pymor/devpi:1
    dependencies:
    {%- for PY in pythons %}
    {%- for ML in manylinuxs %}
      - "wheel {{ML}} py{{PY[0]}} {{PY[2]}}"
    {%- endfor %}
    {%- endfor %}
    needs: [{%- for PY in pythons -%}
    {%- for ML in manylinuxs -%}
      "wheel {{ML}} py{{PY[0]}} {{PY[2]}}",
    {%- endfor -%}
    {%- endfor -%}]
    before_script:
      - pip3 install devpi-client
      - devpi use http://pymor__devpi:3141/root/public --set-cfg
      - devpi login root --password none
      - devpi upload --from-dir --formats=* ./shared
    # the docker service adressing fails on other runners
    tags: [mike]

.sanity_checks:
    extends: .test_base
    image: {{registry}}/pymor/ci_sanity:{{ci_image_tag}}
    stage: sanity
#******** end definition of base jobs *********************************************************************************#

# https://docs.gitlab.com/ee/ci/yaml/README.html#workflowrules-templates
include:
  - template: 'Workflows/Branch-Pipelines.gitlab-ci.yml'

#******* sanity stage

# this step makes sure that on older python our install fails with
# a nice message ala "python too old" instead of "SyntaxError"
verify setup.py:
    extends: .sanity_checks
    script:
        - python3 setup.py egg_info

ci setup:
    extends: .sanity_checks
    script:
        - apk add jq
        - ${CI_PROJECT_DIR}/.ci/gitlab/ci_sanity_check.bash "{{ ' '.join(pythons) }}"

#****** test stage

{%- for py in pythons %}
minimal_cpp_demo {{py[0]}} {{py[2]}}:
    extends: .pytest
    rules:
        - if: $CI_PIPELINE_SOURCE == "schedule"
          when: never
        - when: on_success
    services:
        - name: {{registry}}/pymor/pypi-mirror_stable_py{{py}}:{{pypi_mirror_tag}}
          alias: pypi_mirror
    image: {{registry}}/pymor/testing_py{{py}}:{{ci_image_tag}}
    script: ./.ci/gitlab/cpp_demo.bash
{%- endfor %}


{%- for script, py, para in matrix %}
{{script}} {{py[0]}} {{py[2]}}:
    extends: .pytest
    rules:
        - if: $CI_PIPELINE_SOURCE == "schedule"
          when: never
        - when: on_success
    variables:
        COVERAGE_FILE: coverage_{{script}}__{{py}}
    services:
    {%- if script == "oldest" %}
        - name: {{registry}}/pymor/pypi-mirror_oldest_py{{py}}:{{pypi_mirror_tag}}
    {%- else %}
        - name: {{registry}}/pymor/pypi-mirror_stable_py{{py}}:{{pypi_mirror_tag}}
    {%- endif %}
          alias: pypi_mirror
    image: {{registry}}/pymor/testing_py{{py}}:{{ci_image_tag}}
    script:
        - |
          if [[ "$CI_COMMIT_REF_NAME" == *"github/PR_"* ]]; then
            echo selecting hypothesis profile \"ci_pr\" for branch $CI_COMMIT_REF_NAME
            export PYMOR_HYPOTHESIS_PROFILE="ci_pr"
          else
            echo selecting hypothesis profile \"ci\" for branch $CI_COMMIT_REF_NAME
            export PYMOR_HYPOTHESIS_PROFILE="ci"
          fi
        - ./.ci/gitlab/test_{{script}}.bash
        - find . -name "coverage*"
        - ls -la *
{%- endfor %}

{%- for py in pythons %}
ci_weekly {{py[0]}} {{py[2]}}:
    extends: .pytest
    timeout: 5h
    variables:
        COVERAGE_FILE: coverage_ci_weekly
    rules:
        - if: $CI_PIPELINE_SOURCE == "schedule"
          when: always
    services:
        - name: {{registry}}/pymor/pypi-mirror_stable_py{{py}}:{{pypi_mirror_tag}}
          alias: pypi_mirror
    image: {{registry}}/pymor/testing_py{{py}}:{{ci_image_tag}}
    {# PYMOR_HYPOTHESIS_PROFILE is overwritten from web schedule settings #}
    script: ./.ci/gitlab/test_vanilla.bash
{%- endfor %}

submit coverage:
    extends: .submit
    artifacts:
        when: always
        name: "submit"
        paths:
            - cover/*
            - .coverage
    dependencies:
    {%- for script, py, para in matrix if script in ['vanilla', 'oldest', 'numpy_git', 'mpi'] %}
        - {{script}} {{py[0]}} {{py[2]}}
    {%- endfor %}

{%- for py in pythons %}
submit ci_weekly {{py[0]}} {{py[2]}}:
    extends: .submit
    rules:
        - if: $CI_PIPELINE_SOURCE == "schedule"
          when: always
    image: {{registry}}/pymor/python:{{py}}
    dependencies:
        - ci_weekly {{py[0]}} {{py[2]}}
    needs: ["ci_weekly {{py[0]}} {{py[2]}}"]
{%- endfor %}


{% for OS, PY in testos %}
pip {{loop.index}}/{{loop.length}}:
    tags: [mike]
    services:
        - name: pymor/pypi-mirror_stable_py{{PY}}:{{pypi_mirror_tag}}
          alias: pypi_mirror
    rules:
        - if: $CI_PIPELINE_SOURCE == "schedule"
          when: never
        - when: on_success
    stage: install_checks
    image: {{registry}}/pymor/deploy_checks_{{OS}}:{{ci_image_tag}}
    script: ./.ci/gitlab/install_checks/{{OS}}/check.bash
{% endfor %}

repo2docker:
    extends: .binder
    script:
        - repo2docker --user-id 2000 --user-name ${USER} --no-run --debug --image-name ${IMAGE} .
        - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
        - docker run ${IMAGE} ${CMD}
        - docker push ${IMAGE}

local_jupyter:
    extends: .binder
    script:
        - make docker_image
        - make DOCKER_CMD="${CMD}" docker_exec

{% for url in binder_urls %}
trigger_binder {{loop.index}}/{{loop.length}}:
    extends: .test_base
    stage: deploy
    image: {{registry}}/alpine:3.11
    rules:
        - if: $CI_COMMIT_REF_NAME == "master"
          when: on_success
        - if: $CI_COMMIT_TAG != null
          when: on_success
    before_script:
        - apk --update add bash python3
        - pip3 install requests
    script:
        - python3 .ci/gitlab/trigger_binder.py "{{url}}/${CI_COMMIT_REF}"
{% endfor %}

{%- for PY in pythons %}
{%- for ML in manylinuxs %}
wheel {{ML}} py{{PY[0]}} {{PY[2]}}:
    extends: .wheel
    variables:
        PYVER: "{{PY}}"
    artifacts:
        paths:
        - ${CI_PROJECT_DIR}/shared/pymor*manylinux{{ML}}_*whl
        expire_in: 1 week
    script: bash .ci/gitlab/wheels.bash {{ML}}
{% endfor %}
{% endfor %}

pypi deploy:
    extends: .sanity_checks
    stage: deploy
    dependencies:
    {%- for PY in pythons %}
    {%- for ML in manylinuxs %}
      - wheel {{ML}} py{{PY[0]}} {{PY[2]}}
    {% endfor %}
    {% endfor %}
    rules:
        - if: $CI_PIPELINE_SOURCE == "schedule"
          when: never
        - if: $CI_COMMIT_REF_NAME =~ /^github.*/
          when: never
        - when: on_success
    variables:
        ARCHIVE_DIR: pyMOR_wheels-${CI_COMMIT_REF_NAME}
    artifacts:
        paths:
         - ${CI_PROJECT_DIR}/${ARCHIVE_DIR}/pymor*manylinux*whl
        expire_in: 6 months
        name: pymor-wheels
    script:
        - ${CI_PROJECT_DIR}/.ci/gitlab/pypi_deploy.bash
    environment:
        name: safe

{% for OS, PY in testos %}
check_wheel {{loop.index}}:
    extends: .check_wheel
    image: pymor/deploy_checks:devpi_{{OS}}
    script: devpi install pymor[full]
{% endfor %}

{%- for py in pythons %}
docs build {{py[0]}} {{py[2]}}:
    extends: .test_base
    tags: [mike]
    rules:
        - if: $CI_PIPELINE_SOURCE == "schedule"
          when: never
        - when: on_success
    services:
        - name: {{registry}}/pymor/pypi-mirror_stable_py{{py}}:{{pypi_mirror_tag}}
          alias: pypi_mirror
    image: {{registry}}/pymor/jupyter_py{{py}}:{{ci_image_tag}}
    script:
        - ${CI_PROJECT_DIR}/.ci/gitlab/test_docs.bash
    stage: build
    needs: ["ci setup"]
    artifacts:
        paths:
            - docs/_build/html
            - docs/error.log
{% endfor %}

docs:
    extends: .test_base
    # makes sure this doesn't land on the test runner
    tags: [mike]
    image: {{registry}}/alpine:3.11
    stage: deploy
    resource_group: docs_deploy
    dependencies:
        - "docs build 3 7"
    needs: ["docs build 3 7"]
    before_script:
        - apk --update add make python3 bash
        - pip3 install jinja2 pathlib
    script:
        - ${CI_PROJECT_DIR}/.ci/gitlab/deploy_docs.bash
    rules:
        - if: $CI_PIPELINE_SOURCE == "schedule"
          when: never
        - if: $CI_COMMIT_REF_NAME =~ /^github\/PR_.*/
          when: never
        - when: on_success
    environment:
        name: safe

# THIS FILE IS AUTOGENERATED -- DO NOT EDIT #
#   Edit and Re-run .ci/gitlab/template.ci.py instead       #

'''


import os
import jinja2
import sys
from itertools import product
from pathlib import Path  # python3 only
from dotenv import dotenv_values
tpl = jinja2.Template(tpl)
pythons = ['3.6', '3.7', '3.8']
oldest = [pythons[0]]
newest = [pythons[-1]]
test_scripts = [("mpi", pythons, 1), ("pip_installed", pythons, 1),
    ("vanilla", pythons, 1), ("numpy_git", newest, 1), ("oldest", oldest, 1),]
# these should be all instances in the federation
binder_urls = [f'https://{sub}.mybinder.org/build/gh/pymor/pymor' for sub in ('gke', 'ovh', 'gesis')]
testos = [('centos_8','3.6'), ('debian_buster','3.7'), ('debian_bullseye','3.8')]

env_path = Path(os.path.dirname(__file__)) / '..' / '..' / '.env'
env = dotenv_values(env_path)
ci_image_tag = env['CI_IMAGE_TAG']
pypi_mirror_tag = env['PYPI_MIRROR_TAG']
manylinuxs = [1, 2010, 2014]
registry="zivgitlab.wwu.io/pymor/docker"
with open(os.path.join(os.path.dirname(__file__), 'ci.yml'), 'wt') as yml:
    matrix = [(sc, py, pa) for sc, pythons, pa in test_scripts for py in pythons]
    yml.write(tpl.render(**locals()))
