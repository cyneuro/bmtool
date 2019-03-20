import argparse
from argparse import RawTextHelpFormatter,SUPPRESS
import sys

use_description = """
Build, plot or debug BMTK models easily.

python -m bmtool.build
python -m bmtool.plot 
python -m bmtool.debug

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=use_description, formatter_class=RawTextHelpFormatter,usage=SUPPRESS)
    options = None
    try:
        options = parser.parse_args()
    except:
        args, leftovers = parser.parse_known_args()
        if not args.h:
            parser.print_help()
        sys.exit(0)