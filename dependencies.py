tests_require = ['tox', 'pytest', 'pytest-cache', 'pytest-capturelog', 'pytest-instafail',
                 'pytest-xdist', 'pytest-cov', 'pytest-flakes', 'pytest-pep8']
install_requires = ['Cython', 'numpy', 'scipy', 'PyContracts', 'Sphinx',
                    'docopt', 'numpydoc', 'dogpile.cache']
pre_setup_requires = ['cython', 'numpy']
setup_requires = pre_setup_requires + ['nose']
install_suggests = ['ipython', 'ipdb', 'matplotlib', 'sympy', 'PyOpenGL', 'PySide', 'glumpy'] + tests_require

import_names = {'ipython': 'IPython',
                'pytest-cache': 'pytest_cache',
                'pytest-capturelog': 'pytest_capturelog',
                'pytest-instafail': 'pytest_instafail',
                'pytest-xdist': 'xdist',
                'pytest-cov': 'pytest_cov',
                'pytest-flakes': 'pytest_flakes',
                'pytest-pep8': 'pytest_pep8',
                'PyOpenGL': 'OpenGL'}

if __name__ == '__main__':
    print(' '.join([i for i in install_requires + install_suggests]))
