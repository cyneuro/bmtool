import argparse
from argparse import RawTextHelpFormatter,SUPPRESS
import sys

use_description = """
Debug BMTK models easily.

python -m bmtool.debug

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=use_description, formatter_class=RawTextHelpFormatter,usage=SUPPRESS)
    
    try:
        options = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)