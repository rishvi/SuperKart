# SuperKart Sales Forecasting Project

This folder contains the completed full-code assignment, trained pipeline, deployment applications, container configuration, and sample batch data.

## Final model

- Selected model: tuned Random Forest pipeline
- Cross-validated RMSE: 283.34
- Validation RMSE: 287.64
- Test RMSE: 267.61
- Test MAE: 101.56
- Test R-squared: 0.9369
- Best parameters: 200 trees, unrestricted depth, `max_features=0.8`, and `min_samples_leaf=2`

The serialized artifact contains median/mode imputation, one-hot encoding, and the fitted regressor. The model therefore applies the same preprocessing during training and inference.

## Project contents

- `SuperKart_Model_Deployment_Completed.ipynb`: completed and executed notebook
- `SuperKart_Model_Deployment_Completed.html`: submission-ready HTML export
- `SuperKart.csv`: model-development dataset
- `Batch_Data_SuperKart.csv`: batch-inference example
- `backend_files/`: Flask API, model, dependencies, and Dockerfile
- `frontend_files/`: Streamlit app, dependencies, and Dockerfile
- `docker-compose.yml`: backend/frontend orchestration and internal networking

## Run with Docker

From this directory:

```bash
docker compose up --build
```

Open:

- Frontend: `http://localhost:8501`
- Backend health check: `http://localhost:7860/health`

Stop the services with:

```bash
docker compose down
```

## GitHub Codespaces deployment

1. Push this project to a GitHub repository.
2. Open the repository in a Codespace.
3. Run `docker compose up --build` in the Codespace terminal.
4. In the **Ports** tab, set ports `7860` and `8501` to **Public**.
5. Open the forwarded URL for port `8501` to use the Streamlit app.
6. Copy the forwarded URLs into the notebook's `FORWARDED_BACKEND_URL` and `FORWARDED_FRONTEND_URL` variables.
7. Capture screenshots of the Ports tab, a single prediction, and a batch prediction before the final HTML submission.

Do not store a GitHub Personal Access Token in the notebook or repository.

## API endpoints

- `GET /health`
- `POST /v1/predict` for one JSON record
- `POST /v1/predictbatch` for a CSV uploaded under the multipart field `file`

Example single request:

```bash
curl -X POST http://localhost:7860/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Product_Weight": 12.66,
    "Product_Sugar_Content": "Low Sugar",
    "Product_Allocated_Area": 0.027,
    "Product_MRP": 117.08,
    "Store_Size": "Medium",
    "Store_Location_City_Type": "Tier 2",
    "Store_Type": "Supermarket Type2",
    "Product_Id_char": "FD",
    "Store_Age_Years": 16,
    "Product_Type_Category": "Non Perishables"
  }'
```

Example batch request:

```bash
curl -X POST http://localhost:7860/v1/predictbatch \
  -F "file=@Batch_Data_SuperKart.csv"
```

## Modeling limitation

The supplied table has no date/time field, so this is a supervised revenue-estimation model rather than a classical time-series forecast. It cannot learn seasonality or trends. A production forecasting program should add date-stamped sales, units, promotions, holidays, stockouts, and local-market drivers, then monitor error and drift by store segment.
