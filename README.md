# MLP

Implements a multilayer perceptron (MLP) with k-fold cross-validation in Octave.

*Also compatible with Matlab.*

### Requirements
- Octave (v3.6.2) or Matlab (vR2017a)

### Usage
  You are required to provide the "input" and "output" arguments to the main function. You can optionally add the "verbose" argument.
```sh
$ octave main <input-file> <output-file> [<verbose>]
```
  The input file should have the following format (E.g.: *config.txt*):
```
path = data/*.txt
data = model.mat
expectedOutput = train_5a_ 0,0,1 train_53_ 0,1,0 train_58_ 1,0,0
nh = 5
nkFold = 5
```

  The data files in the path must be *.txt* files with one number (a double) per row. Please notice that the *expectOutput* (variable) contains the output of your MLP. The output file contains the error calculation using a simple mean.
  
  The expected outputs matrix is generated based on the file names. Taking as an example a problem P with three inputs {S, X, Z} in which P = {S, X, Z}. One of the solutions for the problem P is caracter S. Modeling this problem computationally, we can assume that the three inputs will be an ordered vector [S, X, Z]. Still, to be able to making calculations inside MLP network, we must convert that alphanumeric characters for numbers. For this problem, we choose to represent the letters with significant bits. Finally the character S can be represented by S = [1, 0, 0] array. Following the same logic, the X char would be X = [0, 1, 0] and Z = [0, 0, 1].

  | S | X | Z ||
  |---|---|---|--|
  | 1 | 0 | 0 | represents S |
  | 0 | 1 | 0 | represents X |
  | 0 | 0 | 1 | represents Z |
  
  *Verbose* is an boolean: 0 or 1
  
### Docker
You can also use a simple Docker container, following the commands below to build and run it.

```sh
$ cd mlp
$ docker build -t pedrogoncalvesk/mlp .
$ docker run --rm -it -v $(pwd):/source pedrogoncalvesk/mlp
octave:1> main('<input-file>', '<output-file>', [<verbose>])
```

License
----

This repository is licensed under MIT License Copyright (c) 2017. See *LICENSE* for further details.


**Free Software, Hell Yeah!**
