from . import util

use_description = """
Plot BMTK models easily.

python -m bmtool.plot 
"""

if __name__ == '__main__':
    parser = util.get_argparse(use_description)
    util.verify_parse(parser)