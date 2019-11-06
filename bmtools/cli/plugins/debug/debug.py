from . import util

use_description = """
Debug BMTK models easily.

python -m bmtools.debug

"""

if __name__ == '__main__':
    parser = util.get_argparse(use_description)
    util.verify_parse(parser)