# Census Projection Cost Model

This project provides a framework for projecting future employee census data based on historical trends and configurable parameters. It can utilize a LightGBM machine learning model to predict employee turnover or fall back to a rule-based system if ML libraries are unavailable or training data is insufficient.

## Project Structure

The codebase is organized into several Python modules:

*   `main.py`: The main script to run the projection process. Orchestrates data loading, model training (optional), projection, and results saving.
*   `config.py`: Contains configuration parameters for the projection (e.g., projection years, growth rate, turnover rate, compensation increase rate).
*   `data_processing.py`: Handles loading and basic cleaning of input census CSV files.
*   `utils.py`: Provides helper functions for calculations like age and tenure.
*   `ml_logic.py`: Contains all logic related to the machine learning turnover model, including:
    *   Building training data from historical census files.
    *   Training the LightGBM classifier.
    *   Preparing features for prediction.
    *   A rule-based turnover scoring function as a fallback.
    *   Handles optional import of `scikit-learn` and `lightgbm`.
*   `projection.py`: Implements the core year-by-year census projection logic.
*   `dummy_data.py`: Includes a function to generate dummy historical census CSV files for testing and demonstration purposes.

## Requirements

*   Python 3.x
*   pandas
*   numpy
*   **Optional (for ML-based turnover prediction):**
    *   scikit-learn
    *   lightgbm

## Installation

1.  **Clone the repository (if applicable).**
2.  **Install required libraries:**

    ```bash
    pip install pandas numpy
    ```

3.  **Install optional ML libraries (if desired):**

    ```bash
    pip install scikit-learn lightgbm
    ```
    *(Note: If these libraries are not installed, the script will automatically fall back to the rule-based turnover calculation.)*

## Usage

1.  **Prepare Input Data:**
    *   Ensure you have historical census data in CSV format. The expected columns (minimum) are: `ssn`, `birth_date`, `hire_date`, `termination_date`, `gross_compensation`.
    *   Dates should be in a format recognizable by pandas (e.g., `YYYY-MM-DD`).
    *   Place your CSV files in a location accessible by the script.
2.  **Configure Parameters (Optional):**
    *   Edit `config.py` to adjust projection settings like `PROJECTION_YEARS`, `ANNUAL_GROWTH_RATE`, `ANNUAL_TURNOVER_RATE`, etc.
3.  **Run the Projection:**
    *   **Using Actual Files:** Modify `main.py` to replace the call to `create_dummy_census_files` with a list containing the paths to your actual historical census files, ensuring they are in chronological order.
      ```python
      # In main.py
      # Comment out or remove:
      # historical_files = create_dummy_census_files(num_years=3, base_year=2022, n_employees=500)

      # Add your files:
      historical_files = [
          'path/to/your/census_2022.csv',
          'path/to/your/census_2023.csv',
          'path/to/your/census_2024.csv'
      ]
      historical_files.sort() # Optional, but good practice
      ```
    *   **Using Dummy Files (Default):** The script will generate dummy files (`dummy_census_YYYY.csv`) in the current directory by default.
    *   Execute the main script from your terminal:
      ```bash
      python main.py
      ```
4.  **Output:**
    *   The script will print progress information to the console.
    *   The final projections for each year will be saved to an Excel file named `projected_census_all_years.xlsx` in the same directory, with each projected year on a separate sheet.

## Notes

*   The ML model training requires at least two historical census files.
*   The quality of the projection depends heavily on the quality and representativeness of the historical data and the configured parameters.
*   The dummy data generation is for demonstration; replace it with actual data for meaningful projections.
