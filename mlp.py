#!/usr/bin/env/ python

import os
import sys


def mlp(args):
    if len(args) > 2:
        os.system("octave " + sys.prefix + "/mlp/startup.m " + args[0] + " " + args[1] + " " + args[2])
    else:
        os.system("octave " + sys.prefix + "/mlp/startup.m " + args[0] + " " + args[1])

if __name__ == "__main__":
    mlp(sys.argv[1:])
