## File Structure

- data_processing.py: Package that process the data
- EDA.ipynb: Exploratory Data Analysis
- models.ipynb: Model testing
- auxiliary_processing.ipynb: Produce new train test data with additional data information
- combine_auxiliary.py: Utility function to combine primary school, mrt and shopping mall information.

Computation for nearest MRT, mall, school is expensive. Hence, we pre-generated the files for both training and testing data.

Running script need to be triggered at top level, because relative path is used.
