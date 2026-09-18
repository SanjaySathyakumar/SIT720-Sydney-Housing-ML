# SIT720 8.1D - Sydney Housing Price Prediction and Decision Support System

This folder contains a reproducible machine-learning mini project based on 102 manually transcribed sold-property records from Epping, Parramatta and Liverpool, NSW.

## Files
- `sydney_housing_102.csv` - collected dataset (34 records per suburb)
- `SIT720_8_1D_Sydney_Housing.ipynb` - analysis notebook
- `analysis_pipeline.py` - reproducible training/evaluation script used by the notebook
- `best_model.joblib` - fitted model selected by 5-fold cross-validation
- `app.py` - Gradio web application
- `model_metrics.csv`, `top5_prediction_errors.csv`, `comparison_10_properties.csv` - report evidence
- `*.png` - figures used in the report

## Run from a fresh environment
1. Open a terminal in this folder.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run the notebook from top to bottom, or run:
   `python analysis_pipeline.py`
4. Start the web app:
   `python app.py`
5. Open `http://127.0.0.1:7860` in a browser.

## Data sources
Publicly visible sold-property records were manually transcribed from OnTheHouse suburb sold pages:
- https://www.onthehouse.com.au/sold/nsw/epping-2121
- https://www.onthehouse.com.au/sold/nsw/parramatta-2150
- https://www.onthehouse.com.au/sold/nsw/liverpool-2170

Accessed for this project on 18 September 2026. The source URL is retained for every row in the CSV.

## Important submission step
The assessment brief requires an accessible OneDrive/Dropbox ZIP link or a GitHub project link in the report. Upload the ZIP submission pack to one of those services, make access available to the tutor, and replace the highlighted placeholder link in the editable report before exporting the final PDF.

The 10-property comparison includes a `manual_estimate_draft` column only as a starting point. Review or replace those values with your own estimates before submission so the human-judgement component genuinely reflects your reasoning.
