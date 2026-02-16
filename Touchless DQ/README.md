# Touchless DQ — Snowflake Streamlit Application

AI-powered Data Quality check generation using **Snowflake Cortex** and **SODA CL**.

## Features

- **Snowflake Table Browser** — Cascading Database / Schema / Table dropdowns to select data sources directly from Snowflake (no need to type fully-qualified names)
- Upload Collibra metadata and DQ Checks catalog (from Snowflake tables or Excel/CSV files)
- **AI Data Profiling** — Cortex-powered analysis of metadata with quality readiness scoring
- **AI Column Classification** — Uses `CORTEX.CLASSIFY_TEXT` to categorize columns (PII, Financial, Geographic, etc.)
- AI-powered matching of metadata columns to Critical Data Elements (CDEs)
- Auto-generate SODACL YAML check definitions
- AI-generated synthetic test values for validation
- **AI Coverage Assessment** — Evaluates check completeness across DQ dimensions
- **AI Quality Summary** — Scores and analyzes the overall check suite on the marketplace
- Interactive marketplace to review, customize, and approve checks
- **AI Check Improvement** — One-click Cortex-powered YAML refinement on the review page
- **AI Check Explanation** — Plain-English explanation of what each check does
- One-by-one review mode with inline AI-powered testing
- Export approved checks to Snowflake tables

## Snowflake Cortex Functions Used

| Function | Where Used | Purpose |
|---|---|---|
| `CORTEX.COMPLETE` | Matching, Profiling, Synthetic Values, Testing, Coverage, Improvement, Explanation | LLM-powered analysis and generation |
| `CORTEX.CLASSIFY_TEXT` | Column Classification | Categorize columns by data type (PII, Financial, etc.) |

## Folder Structure

```
collibra_upload.py          <- Main app (entry point)
pages/
├── dq_marketplace.py       <- Marketplace: select & customize checks
└── dq_review_page.py       <- Review & approve checks one-by-one
requirements.txt
environment.yml
README.md
```

## Deployment to Snowflake Streamlit

### Prerequisites

- Snowflake account with **Cortex LLM** access enabled
- Role with `CREATE STREAMLIT` privilege
- A warehouse for running the app
- **Streamlit >= 1.26.0** (configured in `environment.yml` for file upload support)

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

Then open the **Streamlit** section in Snowflake UI -> **TOUCHLESS_DQ_APP**.

## Models Supported

| Model | Description |
|---|---|
| llama3.1-70b | Best quality (default) |
| llama3.1-8b | Faster, lighter |
| mistral-large2 | Alternative large model |
| mixtral-8x7b | Mixture of experts |

## App Flow

```
Step 1: Load DQ Catalog        (Snowflake browser or Excel upload)
Step 2: Load Collibra Metadata  (Snowflake browser or Excel/CSV upload)
Step 2.5: AI Data Profiling     (Optional — profile & classify columns)
Step 3: AI Matching             (Match metadata to catalog CDEs)
   -> AI Coverage Assessment    (Evaluate check completeness)
Step 4: Marketplace             (Select, customize, AI quality summary)
Step 5: Review                  (One-by-one review, AI improve, AI explain, test)
   -> Export                    (Save to Snowflake or download)
```
