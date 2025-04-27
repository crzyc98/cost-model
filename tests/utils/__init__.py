# Make this directory a namespace package and include project utils package in its path
import os
# compute project root (two levels up from tests/utils)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# append the real utils directory to this package's path
__path__.append(os.path.join(project_root, 'utils'))
