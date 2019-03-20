from . import util

use_description = """
Build BMTK models easily.

python -m bmtools.build

"""

if __name__ == '__main__':
    parser = util.get_argparse(use_description)
    util.verify_parse(parser)