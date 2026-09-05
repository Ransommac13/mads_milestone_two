# New York Lottery Retail Sales and Location Analysis
## Project overview

This project examines how geographic, demographic, transportation, and commercial characteristics relate to lottery sales across retail locations in New York State.

The project is being completed as part of the University of Michigan Master of Applied Data Science Milestone. It includes two primary analytical components:
1. Supervised learning: Predict lottery sales for a hypothetical new retail location using variables that would be observable before the location begins selling lottery products.
2. Unsupervised learning: Identify meaningful groups of lottery retailers based on their location characteristics and available store features.

The intended result is an academically defensible analysis of retail location performance rather than a production-ready commercial forecasting system.

## Data Sources
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

### Planned External Data
- [US Census Bureau's ACS 5-Year Demographic Data](https://www.census.gov/data/developers/data-sets/acs-5year.2022.html#list-tab-1806015614)
- [EPA Smart Locations](https://www.epa.gov/smartgrowth/smart-location-mapping#SLD)
- [LODES](https://lehd.ces.census.gov/data/)
- [Overture Map Data](https://docs.overturemaps.org/guides/places/)
- [NYS Annual Average Daily Traffic](https://data.ny.gov/Transportation/Annual-Average-Daily-Traffic-AADT-Beginning-1977/6amx-2pbv/about_data)

## Repository Structure
### Shared Repository
- data/
    - interim/
    - final/
    - corrections/
- notebooks/
- outputs/
- src/
    - data/
    - config
    - ingest
    - location_sales_split
    - locaiton_clean
- .gitignore
- requirements.txt
- README.md

### Excluded Data Files
The complete 2025 sales dataset contains approximately 4.76 million records. Because the intermediate Parquet files are large and can be regenerated, they are not stored in GitHub.The following directories and files should remain excluded through .gitignore:
- data/interim/lottery_sales_2025
- data/interim/combined_lottery_sales_2025.parquet

Processed datasets may also be excluded if they exceed GitHub’s file-size limits or contain data that can be reproduced from the project pipeline.

### Local Machine Files
- .venv
- .env
- others

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
pip install -r requirements.txt
### 4. Configure API access
- Create an account on data.ny.gov
- Navigate to developer settings and request an app token
- Create a .env file in the project root:
- SOCRATA_APP_TOKEN=your_data_ny_gov_app_token
- Each contributor should use their own local .env file. The actual .env file must not be committed to GitHub.

## Data Pipeline
1. Query the data.ny.gov API for 2025 sales records.
2. Download the records in deterministic, paginated batches.
3. Save each batch locally in Parquet format.
4. Separate retailer-location attributes from daily sales observations. Use DuckDB to query the raw data so that all 4.76 million records do not need to be loaded into memory simultaneously. Save location data and sales data as intermediate Parquet files.
5. (Clean retailer-location attributes to account for addresses with typos or other discrepancies.)
6. (Clean sales data and aggregate into target period suitable for modeling.)
7. (Match retailer coordinates to geospatial data from planned secondary sources.)
8. (...)

