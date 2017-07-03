# MLP

Implements a multilayer perceptron (MLP) with k-fold cross-validation in Octave.

*Also works fine in Matlab.*

### Requirements
- Octave (v3.6.2) or Matlab (vR2017a)

### Usage
  You must pass input and output to main function. Put verbose optionally.
```sh
$ octave main <input-file> <output-file> [<verbose>]
```
  The input file should have the following format (E.g.: config.txt):
```
path = data/*.txt
data = model.mat
expectedOutput = train_5a_ 0,0,1 train_53_ 0,1,0 train_58_ 1,0,0
nh = 5
nkFold = 5
```

  The data files in path must be a *.txt* files with one number (double) per row.
  You will see that *expectOutput* means exactly expect output of your MLP. Output file shows an calculate rate of error by simple average.
  
  Following the example in *config.txt*, we can have a file *train_5a_00001.txt*. Means that letter S is the expected output in a problem that have three characters for expected outputs: S, X and Z. We represents with with array [0,0,1].
  
  *Verbose* is an boolean: 0 or 1
  
### Docker
If you prefer, you can use an simple Docker container.

```sh
$ cd mlp
$ docker build -t pedrogoncalvesk/mlp .
$ docker run --rm -it -v $(pwd):/source pedrogoncalvesk/mlp
octave:1> main('<input-file>', '<output-file>', [<verbose>])
```

License
----

MIT


**Free Software, Hell Yeah!**
