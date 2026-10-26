# New York Lottery Retail Sales and Location Analysis
## Project overview

This project examines how geographic, demographic, transportation, and commercial characteristics relate to lottery sales across retail locations in New York State.

The project is being completed as part of the University of Michigan Master of Applied Data Science Milestone. It includes two primary analytical components:
1. Supervised learning: Predict lottery sales for a hypothetical new retail location using variables that would be observable before the location begins selling lottery products.
2. Unsupervised learning: Identify meaningful groups of lottery retailers based on their location characteristics and available store features.

The intended result is an academically defensible analysis of retail location performance rather than a production-ready commercial forecasting system.

## Data Sources
100-record samples of all raw data can be found in data/raw/hundred_record_samples.

### New York Lottery Retailer Sales
Data Source: https://data.ny.gov/Government-Finance/Lottery-Daily-Retailer-Sales-By-Game-Beginning-202/xyvi-fbb9/about_data

The primary dataset is the New York Lottery retailer sales dataset published through data.ny.gov. It includes daily sales observations and retailer information such as:
- Retailer identifier (agtno)
- Business name
- Business address
- Date of recorded sales
- Geographic coordinates
- Lottery sales measures

The current analysis uses records from January 1 through December 31, 2025. Retailer-level location data and daily sales data are separated into related tables using agtno as the key.

### External Data for use as Potential Predictive Features
- [US Census Bureau's ACS 5-Year Demographic Data](https://api.census.gov/data/2024/acs/acs5.html)
- [EPA Smart Locations](https://catalog.data.gov/dataset/smart-location-database)
- [LODES](https://lehd.ces.census.gov/data/lodes/LODES8/ny/)
- [Overture Map Data](https://docs.overturemaps.org/getting-data/cloud-sources/)
- [NYS Annual Average Daily Traffic](https://www.dot.ny.gov/tdv)

### How to Access Each Raw Dataset
- NY Lottery Retail Sales: Follow the environment setup instructions below to establish a Socrata client. Execute src.config then src.01_ingest to load the raw sales data. WARNING: Executing this module may take close to one hour due to the required volume of paginated API requests.
- ACS 5-Year Demographic Data: Follow the environment setup instructions below to obtain an API Key. Execute src.config then src.05_acs.ingest to load raw ACS data.
- EPA Smart Locations: Download 'Resource 5' from the webpage provided. Move file into "{ide_project_folder}/data/raw/EPA_NY.csv".
- LODES: Download rac/ny_rac_S000_JT00_2023.csv.gz and wac/ny_rac_S000_JT00_2023.csv.gz from the provided online repository. Move files into "{ide_project_folder}/data/raw/".
- Overture Maps: Execute src.05_overture_places.ingest to query the Overture Places Amazon S3 repository. WARNING: Executing this module will take over one hour due to the computationally expensive process to filter places by information in two different Overture databases.
- NYS AADT: At the bottom of the webpage provided, download "GDB Shapefile AADT 2024-2025.zip" onto your local computer. Move it into "{ide_project_folder}/data/external/GBD_Shapefile_AADT_2024-2025.gdb.zip".

## Repository Structure
### Shared Repository
- data/
    - raw/
    - interim/
    - final/
    - modeling/
    - geography/
    - corrections/
- notebooks/
- src/
    - 05_aadt/
        - ingest_update
    - 05_acs/
        - ingest
    - 05_epa/
        - ingest
    - 05_lodes/
        - rac_ingest
        - wac_ingest
    - 05_overture_places
        - 05a_ingest
        - 05b_feature_eng
    - supervised/
        - a_preprocess_super
        - b_train_ridge
        - b_knn/
            - knn
        - b_rf_gb_model/
            - b1_model_supervised.py
    - unsupervised_clustering/
        - a_preprocess
        - b_gmm
    - config
    - 01_ingest
    - 02_location_sales_split
    - 03_location_clean
    - 04_sales_clean
    - 06_merge_and_split
- .gitignore
- requirements.txt
- README.md

### Excluded Data Files
The complete 2025 sales dataset contains approximately 4.76 million records. Because the intermediate Parquet files are large and can be regenerated, they are not stored in GitHub. Likewise, the AADT GDB shapefile and US Census Shapefile are excluded for large file sizes.
The following directories and files should remain excluded through .gitignore:
- data/interim/lottery_sales_2025
- data/interim/combined_lottery_sales_2025.parquet
- data/external/aadt/GDB_Shapefile_AADT_2024-2025.gdb.zip
- data/external/census_shapefile/tl_2025_36_bg.zip

### Local Machine Files
- .venv
- .env
- data/external/aadt/GDB_Shapefile_AADT_2024-2025.gdb.zip
- data/external/census_shapefile/tl_2025_36_bg.zip

## Environment Setup
### 1. Clone the repository
git clone repository-url
cd repository-name
### 2. Create a virtual environment
On Windows PowerShell:
- python -m venv .venv
- .venv\Scripts\Activate.ps1
### 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
### 4. Configure API access
- Socrata
    - Create an account on data.ny.gov
    - Navigate to developer settings and request an app token
    - Create a .env file in the project root:
    - SOCRATA_APP_TOKEN=your_data_ny_gov_app_token
    - Each contributor should use their own local .env file. The actual .env file must not be committed to GitHub.
- US Census
    - Request US Census API Key at https://api.census.gov/data/key_signup.html
    - Copy the API Key sent to your email
    - In your local .env file in the project root:
    - ACS_API_KEY=your_us_census_api_key
    - Each contributor should use their own local .env file. The actual .env file must not be committed to GitHub.

## Collaboration Best Practices
### Pushing Changes
1. At the start of your session, pull in the updated repository.
2. Make desired changes.
3. Before pushing, pull again to catch any recent updates.
4. Commit and push one change at a time.

## Module Execution Order
Execute in terminal with "python -m src.{file_path}". Do not include the .py suffix.
1. config
2. 01_ingest
3. 02_location_sales_split
4. 03_location_clean
5. 04_sales_clean
6. 05_aadt/ingest_update
7. 05_acs/ingest
8. 05_epa/ingest
9. 05_lodes/rac_ingest
10. 05_lodes/wac_ingest
11. 05_overture_places/ingest
12. 05_overture_places/feature_eng
13. unsupervised_clustering/a_preprocess
14. unsupervised_clustering/b_gmm
15. supervised/a_preprocess_super
16. supervised/b_train_ridge
17. supervised/b_knn/knn
18. supervised/b_rf_gb_model/b1_model_supervised

