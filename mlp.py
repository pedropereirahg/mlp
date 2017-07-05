#!/usr/bin/env/ python

import os
import sys


def mlp(args):
    print "octave " + sys.prefix + "mlp/main " + args[0] + ", " + args[1] + ", " + args[2]
    os.system("octave " + sys.prefix + "mlp/main " + args[0] + ", " + args[1] + ", " + args[2])

if __name__ == "__main__":
    mlp(sys.argv[1:])
