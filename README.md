# Numerical Analysis Algorithms (CLI)

A collection of algorithms for solving systems of linear equations and matrix operations, implemented in Python with an interactive Terminal User Interface (TUI) using `curses`.

## Overview

Developed for the Numerical Analysis course, this project implements direct and iterative methods to solve linear algebra problems. It features a navigable menu system allowing users to input matrix dimensions and data directly in the terminal.

## Implemented Methods

The tool includes the following numerical strategies:

| Category | Methods |
| :--- | :--- |
| **Basic Operations** | Determinant Calculation, Inverse Matrix |
| **Direct Solvers** | Gaussian Elimination (Compact), Gauss-Jordan, LU Decomposition, Cholesky Decomposition |
| **Iterative Solvers** | Jacobi Method, Gauss-Seidel Method |
| **Triangular Systems** | Forward Substitution (Lower), Backward Substitution (Upper) |

## Key Features

* **Interactive Menu:** Navigation using keyboard arrows (Up/Down).
* **Input Handling:** Robust reading of matrix elements and vectors via standard screen.
* **Math Backend:** Utilizes `numpy` for array manipulation and linear algebra primitives.

## Structure

* `src/metodos.py`: Main application containing the `curses` menu loop and algorithm implementations.
* `src/metodos2.py`: Supplementary methods and experiments.

## Usage

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    *Note: The `curses` library is part of the standard Python library on Linux/macOS. Windows users may need to install `windows-curses`.*

2.  **Run the application:**
    ```bash
    python src/metodos.py
    ```

## Technologies

* **Language:** Python 3
* **Libraries:** Numpy, Curses (Standard Library)
