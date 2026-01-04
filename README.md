# Numerical Analysis Algorithms

A collection of algorithms for solving systems of linear equations, matrix operations, and data approximation, implemented in Python.

## Overview

Developed for the Numerical Analysis course, this repository contains tools for two main areas of computational mathematics:
1.  **Linear Algebra:** Direct and iterative solvers for linear systems using a TUI (Terminal User Interface).
2.  **Data Approximation:** Interpolation and Curve Fitting methods for analyzing datasets.

## Modules

### 1. Linear Algebra Solvers (`src/metodosPy.py`)
Interactive tool built with `curses` for matrix operations. It features a navigation menu for selecting algorithms.

* **Direct Methods:**
    * Gaussian Elimination (Compact)
    * Gauss-Jordan
    * LU Decomposition
    * Cholesky Decomposition
* **Iterative Methods:**
    * Jacobi Method
    * Gauss-Seidel Method
* **Basic Operations:**
    * Determinant Calculation
    * Matrix Inversion
    * Triangular System Solution (Lower/Upper)

### 2. Interpolation & Curve Fitting (`src/metodosPy2.py`)
CLI tool for estimating functions based on discrete points and analyzing data trends (Work #2).

* **Interpolation:**
    * Newton's Divided Differences
    * Newton-Gregory (Forward Differences)
* **Curve Fitting (Regression):**
    * Linear Regression (Least Squares)
    * Polynomial Regression (variable degree)
    * Exponential Fit
* **Statistical Metrics:**
    * Coefficient of Determination ($R^2$) calculation

## Usage

### Dependencies
Install the required libraries:
```bash
pip install -r requirements.txt

Running the Tools

For Linear Systems (Matrix Menu):
Bash

python src/metodosPy.py

Note: Requires a terminal with curses support (Linux/macOS standard).

For Interpolation and Fitting:
Bash

python src/metodosPy2.py

Technologies

    Language: Python 3

    Libraries: Numpy, Curses (Standard Library)

Developed by Kauã Junior Silva Soares.
