#!/usr/bin/env/ python

import os
import sys


def mlp(args):
    if len(args) > 2:
        os.system("octave -p " + sys.prefix + "/mlp" + sys.prefix + "/mlp/startup.m " +
                  str(args[0]) + " " + str(args[1]) + " " + str(args[2]))
    else:
        os.system("octave -p " + sys.prefix + "/mlp " + sys.prefix + "/mlp/startup.m " +
                  str(args[0]) + " " + str(args[1]))

if __name__ == "__main__":
    mlp(sys.argv[1:])
