#!/usr/bin/env/ python

import os
import sys


def mlp(args):
    os.system("octave main(" + args[1] + ", " + args[2] + ", " + args[3] + ")")

if __name__ == "__main__":
    mlp(sys.argv[1:])
