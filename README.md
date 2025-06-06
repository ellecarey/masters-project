# Overview

This repository contains the code and resources for my Masters project, focused on error propagation in neural networks.

The core functionality is a Python-based data generator that creates synthetic datasets with Gaussian-distributed features, controlled perturbations, and the ability to create target variables.  The project is structured as a Python package for modularity and reusability.

## Project Structure

The repository is organized as follows:

*   `data/`: Contains any raw data or datasets used by the project.
*   `models/`: Trained and serialized models, model predictions, or model summaries
*   `reports/`: Stores generated reports, including:
    *   `figures/`: Contains plots and visualizations generated by the `gaussian_data_generator`.
*   `notebooks/`: Jupyter Notebooks used for experimentation, analysis, or demonstration. Naming convention is a number for ordering and a short "-" delimited description.
*   `src/`:  The main source code directory.
    *   `generator_package/`: A Python package containing the core modules:
        *   `__init__.py`: Initializes the package.
        *   `config.py`: Defines configuration parameters for the data generator.
        *   `gaussian_data_generator.py`: Contains the `GuassianDataGenerator` class for creating synthetic datasets.
        *   `plotting_style.py`: Defines a custom Matplotlib style for consistent plot appearance.
    *   `main.py`: The main script to run the data generation process.
*   `requirements.txt`: Lists the Python dependencies for the project.
*   `pyproject.toml`: Specifies the project's build system and dependencies (if using modern packaging tools).
*   `README.md`: This file.

## Key Components

*   **`GuassianDataGenerator` Class:** Located in `src/generator_package/gaussian_data_generator.py`, it provides methods for:
    *   Generating features with Gaussian distributions.
    *   Adding controlled perturbations (noise) to the data.
    *   Creating target variables based on linear, polynomial, or logistic functions of the features.
    *   Visualizing feature distributions.
*   **Configuration via `config.py`:** The `src/generator_package/config.py` file allows users to easily configure the data generation process by setting parameters such as the number of samples, number of features, noise levels, and output paths.
*   **Custom Plotting Style:** The `src/generator_package/plotting_style.py` module defines a custom Matplotlib style for generating publication-quality figures. It handles LaTeX rendering (if available) and sets consistent font sizes, colors, and plot styles.
*   **Execution with `main.py`:** The `src/main.py` script orchestrates the data generation process by importing the `GuassianDataGenerator` class, reading configuration parameters from `config.py`, generating data, and visualizing the results.

## Usage

1.  **Clone the repository:**
    ```
    git clone https://github.com/ellecarey/masters-project.git
    ```
2.  **Create a virtual environment (recommended):**
    ```
    python -m venv .venv
    ```
3.  **Activate the virtual environment:**
    *   **Windows:** `.venv\Scripts\activate`
    *   **macOS/Linux:** `source .venv/bin/activate`
4.  **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```
5.  **Configure the data generation process:**
    *   Edit the `src/generator_package/config.py` file to set the desired parameters.
6.  **Run the data generation script:**
    ```
    python src/main.py
    ```
    This will generate a synthetic dataset and save it to the specified output path (if configured). It will also display and save visualizations of the generated features (if visualization is enabled).

## Current Status / Next Steps

The project is currently at an early stage of generating toy Gaussian data.

Planned next steps include:

*   Implementing data validation checks
*   Integrating with a machine learning framework for model training
*   Systematically analyzing the effects of data perbutations on machine learning models

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgements

*   This project utilizes Matplotlib, NumPy, Pandas, Scikit-learn and, PyTorch.
*   With thanks to my MSc supervisor Professor Adrian Bevan 