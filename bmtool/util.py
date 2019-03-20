import argparse
from argparse import RawTextHelpFormatter,SUPPRESS
import sys

def get_argparse(use_description):
    parser = argparse.ArgumentParser(description=use_description, formatter_class=RawTextHelpFormatter,usage=SUPPRESS)
    return parser
    
def verify_parse(parser):
    try:
        if not len(sys.argv) > 1:
            raise
        if sys.argv[1] in ['-h','--h','-help','--help','help']:
            raise
        parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
        
        

        
use_description = """
BMTK model utilties.

python -m bmtool.util 
"""

if __name__ == '__main__':
    parser = get_argparse(use_description)
    verify_parse(parser)