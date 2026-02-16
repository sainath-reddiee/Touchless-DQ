# Touchless DQ — Snowflake Streamlit Application

AI-powered Data Quality check generation using **Snowflake Cortex** and **SODA CL**.

## Features

- Upload Collibra metadata and DQ Checks catalog (from Snowflake tables or Excel files)
- AI-powered matching of metadata columns to Critical Data Elements (CDEs)
- Auto-generate SODACL YAML check definitions
- AI-generated synthetic test values for validation
- Interactive marketplace to review, customize, and approve checks
- One-by-one review mode with inline AI-powered testing
- Export approved checks to Snowflake tables

## Folder Structure

```
collibra_upload.py          ← Main app (entry point)
pages/
├── dq_marketplace.py       ← Marketplace: select & customize checks
└── dq_review_page.py       ← Review & approve checks one-by-one
requirements.txt
environment.yml
README.md
```

## Deployment to Snowflake Streamlit

### Prerequisites

- Snowflake account with **Cortex LLM** access enabled
- Role with `CREATE STREAMLIT` privilege
- A warehouse for running the app

### Deploy via SnowSQL

```sql
-- Create database & schema
CREATE DATABASE IF NOT EXISTS TOUCHLESS_DQ_DB;
CREATE SCHEMA IF NOT EXISTS TOUCHLESS_DQ_DB.APP;

-- Create stage
CREATE STAGE IF NOT EXISTS TOUCHLESS_DQ_DB.APP.STREAMLIT_STAGE
  DIRECTORY = (ENABLE = TRUE);
```

```bash
# Upload files via SnowSQL
PUT file://./collibra_upload.py @TOUCHLESS_DQ_DB.APP.STREAMLIT_STAGE/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://./pages/dq_marketplace.py @TOUCHLESS_DQ_DB.APP.STREAMLIT_STAGE/pages/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://./pages/dq_review_page.py @TOUCHLESS_DQ_DB.APP.STREAMLIT_STAGE/pages/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://./requirements.txt @TOUCHLESS_DQ_DB.APP.STREAMLIT_STAGE/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file://./environment.yml @TOUCHLESS_DQ_DB.APP.STREAMLIT_STAGE/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
```

```sql
-- Create the Streamlit app
CREATE STREAMLIT IF NOT EXISTS TOUCHLESS_DQ_DB.APP.TOUCHLESS_DQ_APP
  ROOT_LOCATION = '@TOUCHLESS_DQ_DB.APP.STREAMLIT_STAGE'
  MAIN_FILE = 'collibra_upload.py'
  QUERY_WAREHOUSE = 'COMPUTE_WH';
```

Then open the **Streamlit** section in Snowflake UI → **TOUCHLESS_DQ_APP**.

## Models Supported

| Model | Description |
|---|---|
| llama3.1-70b | Best quality (default) |
| llama3.1-8b | Faster, lighter |
| mistral-large2 | Alternative large model |
| mixtral-8x7b | Mixture of experts |
