# Randomized PCA Forest

This project implements Randomized PCA Forest algorithm. The script allows you to perform approximate k-nearest neighbors search as presented in the paper, "Randomized PCA Forest For Approximate k-Nearest Neighbors Search".

## Table of Contents
- Installation
- Usage
- Arguments
- Example
- Contributing
- License

## Installation

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage

To run the full algorithm, use the following command:
```bash
python rpcaforest.py [options]
```


To run the fast implementation, use the following command:
```bash
python fastrpcaforest.py [options]
```

## Arguments

The script accepts the following arguments:

- `-d`, `--dataset`: Path to the dataset CSV file. Default is `./data.csv`.
- `-k`, `--k`: Value of k. Default is `5`.
- `-p`, `--principalcomponents`: Number of principal components to use. Default is `1` for full algorithm and `5` for the fast implementation.
- `-l`, `--leafsize`: Maximum size of a node to be considered a leaf. Default is `10`.
- `-f`, `--forestsize`: Number of trees in the forest. Default is `40`.
- `-t`, `--threads`: Number of threads to use. Default is `4`.
- `-r`, `--recursionlimit`: Maximum number of recursions allowed. Default is `1000`.
- `-v`, `--verbos`: Set it to `1` to enable verbosity, `0` to disable it. Default is `1`.

## Example

Here is an example of how to run the script:
```bash
python rpcaforest.py -d ./data.csv -k 5 -p 2 -l 15 -f 50 -t 8 -r 2000 -v 1
```

In the output, you can see the recall and the average discrepancy ratio.

You can also use evaluate function to easily write your own experiemnts:

```python
evaluate(data, knntable, k, p, l, f, t, v)
```

## Citation


If you use our metrics in your research, please cite the original paper:


```
@article{RAJABINASAB2024126254,
title = {Randomized PCA forest for approximate k-nearest neighbor search},
journal = {Expert Systems with Applications},
pages = {126254},
year = {2024},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2024.126254},
url = {https://www.sciencedirect.com/science/article/pii/S095741742403121X},
author = {Muhammad Rajabinasab and Farhad Pakdaman and Arthur Zimek and Moncef Gabbouj},
```
