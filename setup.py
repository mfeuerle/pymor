#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import print_function

import sys
import os
import subprocess
from setuptools import find_packages
from setuptools.command.test import test as TestCommand
from distutils.extension import Extension
import itertools

import dependencies

tests_require = dependencies.tests_require
install_requires = dependencies.install_requires
setup_requires = dependencies.setup_requires
install_suggests = dependencies.install_suggests


class PyTest(TestCommand):

    def finalize_options(self):
        TestCommand.finalize_options(self)
        print(sys.argv[3:])
        self.test_args = sys.argv[3:] + ['--cov=pymor', '--cov-report=html', '--cov-report=xml', 'src/pymortests']
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


class DependencyMissing(Exception):

    def __init__(self, names):
        super(DependencyMissing, self).__init__('Try: "for i in {} ; do pip install $i ; done"'.format(' '.join(names)))


def _numpy_monkey():
    '''Apparently we need to monkey numpy's distutils to be able to build
    .pyx with Cython instead of Pyrex. The monkeying below is copied from
    https://github.com/matthew-brett/du-cy-numpy/blob/master/matthew_monkey.py
    via the discussion at http://comments.gmane.org/gmane.comp.python.numeric.general/37752
    '''
    global _build_src
    from os.path import join as pjoin, dirname
    from distutils.dep_util import newer_group
    from distutils.errors import DistutilsError

    from numpy.distutils.misc_util import appendpath
    from numpy.distutils import log

    from numpy.distutils.command import build_src
    _orig_generate_a_pyrex_source = build_src.build_src.generate_a_pyrex_source

    def generate_a_pyrex_source(self, base, ext_name, source, extension):
        ''' Monkey patch for numpy build_src.build_src method

        Uses Cython instead of Pyrex, iff source contains 'pymor'

        Assumes Cython is present
        '''
        if 'pymor' not in source:
            return _orig_generate_a_pyrex_source(self, base, ext_name, source, extension)

        if self.inplace:
            target_dir = dirname(base)
        else:
            target_dir = appendpath(self.build_src, dirname(base))
        target_file = pjoin(target_dir, ext_name + '.c')
        depends = [source] + extension.depends
        if self.force or newer_group(depends, target_file, 'newer'):
            import Cython.Compiler.Main
            log.info("cythonc:> %s" % (target_file))
            self.mkpath(target_dir)
            options = Cython.Compiler.Main.CompilationOptions(
                defaults=Cython.Compiler.Main.default_options,
                include_path=extension.include_dirs,
                output_file=target_file)
            cython_result = Cython.Compiler.Main.compile(source, options=options)
            if cython_result.num_errors != 0:
                raise DistutilsError("%d errors while compiling %r with Cython"
                                     % (cython_result.num_errors, source))
        return target_file

    build_src.build_src.generate_a_pyrex_source = generate_a_pyrex_source


def write_version():
    filename = os.path.join(os.path.dirname(__file__), 'src', 'pymor', 'version.py')
    try:
        if 'PYMOR_DEB_VERSION' in os.environ:
            revstring = os.environ['PYMOR_DEB_VERSION']
        else:
            revstring = subprocess.check_output(['git', 'describe',
                                                 '--tags', '--candidates', '20', '--match', '*.*.*']).strip()
        with open(filename, 'w') as out:
            out.write('revstring = \'{}\''.format(revstring))
    except:
        if os.path.exists(filename):
            loc = {}
            execfile(filename, loc, loc)
            revstring = loc['revstring']
        else:
            revstring = '0.0.0-0-0'
    return revstring


def _setup(**kwargs):

    # the following hack is taken from scipy's setup.py
    # https://github.com/scipy/scipy
    if (len(sys.argv) >= 2 and
            ('--help' in sys.argv[1:] or sys.argv[1] in ('--help-commands', 'egg_info', '--version', 'clean'))):
        # For these actions, NumPy is not required.
        #
        # They are required to succeed without Numpy for example when
        # pip is used to install Scipy when Numpy is not yet present in
        # the system.
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup
        return setup(**kwargs)

    _numpy_monkey()
    import Cython.Distutils
    # numpy sometimes expects this attribute, sometimes not. all seems ok if it's set to none
    if not hasattr(Cython.Distutils.build_ext, 'fcompiler'):
        Cython.Distutils.build_ext.fcompiler = None
    cmdclass = {'build_ext': Cython.Distutils.build_ext,
                'test': PyTest}
    from numpy import get_include
    include_dirs = [get_include()]
    ext_modules = [Extension("pymor.tools.relations", ["src/pymor/tools/relations.pyx"], include_dirs=include_dirs),
                   Extension("pymor.tools.inplace", ["src/pymor/tools/inplace.pyx"], include_dirs=include_dirs),
                   Extension("pymor.grids._unstructured", ["src/pymor/grids/_unstructured.pyx"], include_dirs=include_dirs)]
    # for some reason the *pyx files don't end up in sdist tarballs -> manually add them as package data
    # this _still_ doesn't make them end up in the tarball however -> manually add them in MANIFEST.in
    kwargs['package_data'] = {'pymor': list(itertools.chain(*[i.sources for i in ext_modules]))}

    kwargs['cmdclass'] = cmdclass
    kwargs['ext_modules'] = ext_modules

    # lastly we'll need to tweak matplotlibs config dir or else
    # installing it from setup will result in a SandboxViolation
    import os
    os.environ['MPLCONFIGDIR'] = "."

    from numpy.distutils.core import setup
    return setup(**kwargs)


def _missing(names):
    for name in names:
        try:
            __import__(name)
        except ImportError:
            if name in dependencies.import_names:
                try:
                    __import__(dependencies.import_names[name])
                except ImportError:
                    yield name
            else:
                yield name


def setup_package():
    revstring = write_version()

    _setup(
        name='pymor',
        version=revstring,
        author='pyMOR developers',
        author_email='pymor-dev@listserv.uni-muenster.de',
        maintainer='Rene Milk',
        maintainer_email='rene.milk@wwu.de',
        package_dir={'': 'src'},
        packages=find_packages('src'),
        include_package_data=True,
        scripts=['src/pymor-demo', 'distribute_setup.py', 'dependencies.py'],
        url='http://pymor.org',
        description=' ',
        long_description=open('README.txt').read(),
        setup_requires=setup_requires,
        tests_require=tests_require,
        install_requires=install_requires,
        classifiers=['Development Status :: 4 - Beta',
                     'License :: OSI Approved :: BSD License',
                     'Programming Language :: Python :: 2.7',
                     'Intended Audience :: Science/Research',
                     'Topic :: Scientific/Engineering :: Mathematics',
                     'Topic :: Scientific/Engineering :: Visualization'],
        license='LICENSE.txt',
        zip_safe=False,
    )

    missing = list(_missing(install_suggests.keys()))
    if len(missing):
        import textwrap
        print('\n' + '*' * 79 + '\n')
        print('There are some suggested packages missing:\n')
        col_width = max(map(len, missing)) + 3
        for package in sorted(missing):
            description = textwrap.wrap(install_suggests[package], 79 - col_width)
            print('{:{}}'.format(package + ':', col_width) + description[0])
            for d in description[1:]:
                print(' ' * col_width + d)
            print()
        print("\ntry: 'pip install {}'".format(' '.join(missing)))
        print('\n' + '*' * 79 + '\n')


if __name__ == '__main__':
    setup_package()
