from ..util import util

use_description = """
Debug BMTK models easily.

python -m bmtool.debug

"""

if __name__ == '__main__':
    parser = util.get_argparse(use_description)
    util.verify_parse(parser)