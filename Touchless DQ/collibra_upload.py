import streamlit as st
import pandas as pd
import json

# App configuration — only in main file
st.set_page_config(
    page_title="Touchless DQ",
    page_icon="📤",
    layout="wide"
)

# Hide default navigation
st.markdown("""
<style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("📤 Upload Collibra Metadata & Catalog")

# ---------------------------------------------------------------------------
# Snowflake Session — auto-detect SiS vs local environment
# ---------------------------------------------------------------------------
IS_LOCAL = False

def get_snowflake_session():
    """Return a Snowpark session. Works in both SiS and local dev."""
    global IS_LOCAL
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        IS_LOCAL = False
        return session
    except Exception:
        IS_LOCAL = True
        return None

def get_local_connection():
    """Create a local snowflake.connector connection via username/password."""
    import snowflake.connector
    conn = snowflake.connector.connect(
        user=st.session_state.get("sf_user", ""),
        password=st.session_state.get("sf_password", ""),
        account=st.session_state.get("sf_account", ""),
        warehouse=st.session_state.get("sf_warehouse", None) or None,
        database=st.session_state.get("sf_database", None) or None,
        schema=st.session_state.get("sf_schema", None) or None,
    )
    return conn

# Try SiS first
session = get_snowflake_session()

# Initialize session state
if "connected" not in st.session_state:
    st.session_state.connected = session is not None
if "connection" not in st.session_state:
    st.session_state.connection = None
if "collibra_metadata" not in st.session_state:
    st.session_state.collibra_metadata = None
if "catalog_df" not in st.session_state:
    st.session_state.catalog_df = None
if "generated_checks" not in st.session_state:
    st.session_state.generated_checks = None

if session is not None:
    st.session_state.connected = True
    st.session_state.session = session


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    if IS_LOCAL:
        # ---- Local dev: show Snowflake connection form ----
        st.header("⚙️ Snowflake Connection")
        st.session_state.sf_account = st.text_input("Account Identifier", help="e.g. xy12345.us-east-1")
        st.session_state.sf_user = st.text_input("Username")
        st.session_state.sf_password = st.text_input("Password", type="password")
        st.session_state.sf_warehouse = st.text_input("Warehouse", value="COMPUTE_WH")
        st.session_state.sf_database = st.text_input("Database", help="Optional")
        st.session_state.sf_schema = st.text_input("Schema", help="Optional")

        if st.button("🔐 Connect", use_container_width=True):
            if not st.session_state.sf_account or not st.session_state.sf_user or not st.session_state.sf_password:
                st.error("Please provide Account Identifier, Username, and Password")
            else:
                try:
                    with st.spinner("Connecting to Snowflake..."):
                        conn = get_local_connection()
                        st.session_state.connection = conn
                        st.session_state.connected = True
                        st.success("✅ Connected!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
                    st.session_state.connected = False

        if st.session_state.connected:
            st.success("✅ Connected to Snowflake")
        st.divider()
    else:
        # ---- SiS: auto-connected ----
        st.success("✅ Connected to Snowflake")
        st.divider()

    st.header("🤖 Model Settings")
    model = st.selectbox(
        "Select Model",
        ["llama3.1-70b", "llama3.1-8b", "mistral-large2", "mixtral-8x7b"],
        index=0
    )
    st.session_state.model = model

    st.divider()
    st.caption(f"ℹ️ Mode: {'Local Dev' if IS_LOCAL else 'Snowflake Streamlit'}")


# ---------------------------------------------------------------------------
# Helper — call Snowflake Cortex LLM (works in both modes)
# ---------------------------------------------------------------------------
def call_llm(prompt: str, model_name: str):
    """Call Snowflake Cortex LLM — supports both Snowpark session and connector."""
    try:
        escaped_prompt = prompt.replace("'", "''")
        query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model_name}',
            '{escaped_prompt}'
        ) AS response
        """

        if session is not None:
            # SiS: use Snowpark session
            result = session.sql(query).collect()
            return result[0]["RESPONSE"] if result else None
        elif st.session_state.get("connection"):
            # Local: use connector cursor
            cur = st.session_state.connection.cursor()
            cur.execute(query)
            result = cur.fetchone()
            cur.close()
            return result[0] if result else None
        else:
            st.error("Not connected to Snowflake.")
            return None
    except Exception as e:
        st.error(f"LLM Error: {str(e)}")
        return None


# ---------------------------------------------------------------------------
# Helper — parse JSON from LLM response (handles markdown fences)
# ---------------------------------------------------------------------------
def parse_llm_json(raw: str):
    """Best-effort extraction of a JSON object from a raw LLM response."""
    cleaned = raw.strip()
    if '```' in cleaned:
        parts = cleaned.split('```')
        if len(parts) >= 3:
            cleaned = parts[1].replace('json', '', 1)
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Helper — save dataframe to Snowflake (works in both modes)
# ---------------------------------------------------------------------------
def save_to_snowflake(df: pd.DataFrame, table_name: str):
    """Save a pandas DataFrame to a Snowflake table."""
    try:
        if session is not None:
            snowpark_df = session.create_dataframe(df)
            snowpark_df.write.mode("overwrite").save_as_table(table_name)
        elif st.session_state.get("connection"):
            from snowflake.connector.pandas_tools import write_pandas
            write_pandas(
                st.session_state.connection,
                df,
                table_name,
                auto_create_table=True,
                overwrite=True
            )
        else:
            st.error("Not connected to Snowflake.")
            return False
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Helper — load table (works in both modes)
# ---------------------------------------------------------------------------
def load_table(table_name: str) -> pd.DataFrame:
    """Load a Snowflake table as a pandas DataFrame."""
    if session is not None:
        return session.table(table_name).to_pandas()
    elif st.session_state.get("connection"):
        cur = st.session_state.connection.cursor()
        cur.execute(f"SELECT * FROM {table_name}")
        columns = [desc[0] for desc in cur.description]
        data = cur.fetchall()
        cur.close()
        return pd.DataFrame(data, columns=columns)
    else:
        raise Exception("Not connected to Snowflake.")


# ===========================================================================
# Guard: must be connected
# ===========================================================================
if not st.session_state.connected:
    st.info("👈 Please connect to Snowflake in the sidebar")
    st.stop()


# ===========================================================================
# Step 1 — Load DQ Checks Catalog
# ===========================================================================
st.header("Step 1: Load DQ Checks Catalog")

catalog_source = st.radio(
    "Choose catalog source",
    ["Snowflake Table", "Upload Excel File"],
    horizontal=True,
    key="catalog_source"
)

if catalog_source == "Snowflake Table":
    catalog_table = st.text_input(
        "Fully-qualified table name",
        value="",
        placeholder="DB.SCHEMA.DQ_CHECKS_CATALOG",
        key="catalog_table_input"
    )
    if st.button("📥 Load Catalog", key="load_catalog_btn") and catalog_table:
        try:
            catalog_df = load_table(catalog_table)
            st.session_state.catalog_df = catalog_df
            st.success(f"✅ Catalog loaded: {len(catalog_df)} checks")
        except Exception as e:
            st.error(f"Failed to load table: {e}")
else:
    catalog_file = st.file_uploader(
        "Upload the catalog Excel file",
        type=['xlsx', 'xls'],
        key="catalog_upload"
    )
    if catalog_file:
        catalog_df = pd.read_excel(catalog_file)
        st.session_state.catalog_df = catalog_df
        st.success(f"✅ Catalog loaded: {len(catalog_df)} checks")

if st.session_state.catalog_df is not None:
    with st.expander("📋 View Catalog"):
        st.dataframe(st.session_state.catalog_df, use_container_width=True)

st.divider()

# ===========================================================================
# Step 2 — Load Collibra Metadata
# ===========================================================================
st.header("Step 2: Load Collibra Metadata")

metadata_source = st.radio(
    "Choose metadata source",
    ["Snowflake Table", "Upload Excel/CSV File"],
    horizontal=True,
    key="metadata_source"
)

metadata_df = None
if metadata_source == "Snowflake Table":
    metadata_table = st.text_input(
        "Fully-qualified table name",
        value="",
        placeholder="DB.SCHEMA.COLLIBRA_METADATA",
        key="metadata_table_input"
    )
    if st.button("📥 Load Metadata", key="load_metadata_btn") and metadata_table:
        try:
            metadata_df = load_table(metadata_table)
            st.session_state.collibra_metadata = metadata_df
            st.success(f"✅ Metadata loaded: {len(metadata_df)} columns")
        except Exception as e:
            st.error(f"Failed to load table: {e}")
else:
    metadata_file = st.file_uploader(
        "Upload Collibra metadata Excel/CSV",
        type=['xlsx', 'xls', 'csv'],
        key="metadata_upload"
    )
    if metadata_file:
        if metadata_file.name.endswith('.csv'):
            metadata_df = pd.read_csv(metadata_file)
        else:
            metadata_df = pd.read_excel(metadata_file)
        st.session_state.collibra_metadata = metadata_df
        st.success(f"✅ Metadata loaded: {len(metadata_df)} columns")

# Use whatever is in session state
if st.session_state.collibra_metadata is not None:
    metadata_df = st.session_state.collibra_metadata
    with st.expander("📋 View Metadata"):
        st.dataframe(metadata_df, use_container_width=True)

st.divider()

# ===========================================================================
# Step 3 — AI Matching
# ===========================================================================
if st.session_state.collibra_metadata is not None and st.session_state.catalog_df is not None:
    st.header("Step 3: AI-Powered Matching")

    metadata_df = st.session_state.collibra_metadata
    catalog_df = st.session_state.catalog_df

    if st.button("🧠 Match Metadata to Catalog", type="primary", use_container_width=True):
        with st.spinner("AI is analyzing and matching..."):

            # Prepare info for the prompt
            catalog_cdes = catalog_df['Critical Data Element'].unique().tolist()
            catalog_domains = catalog_df['Domain'].unique().tolist()

            metadata_info = []
            for _, row in metadata_df.iterrows():
                metadata_info.append({
                    'column_name': row.get('Column name', ''),
                    'description': row.get('Column description', ''),
                    'datatype': row.get('Datatype', ''),
                    'related_fields': str(row.get('Related Fields (Min/Max/Sample Values)', ''))
                })

            matching_prompt = f"""You are a data quality expert. Match Collibra metadata columns to catalog Critical Data Elements (CDEs).

Catalog Domains: {catalog_domains}
Catalog CDEs: {catalog_cdes}

Metadata Columns:
{json.dumps(metadata_info, indent=2)}

For each metadata column:
1. Identify the best matching Domain from catalog
2. Identify the best matching CDE from catalog
3. If no good match, use "Unknown"

Respond ONLY with valid JSON:
{{
  "matches": [
    {{
      "column_name": "PROPERTY_LATITUDE",
      "matched_domain": "Property",
      "matched_cde": "latitude",
      "confidence": "high",
      "reason": "latitude coordinate matches catalog CDE"
    }}
  ]
}}

Output only JSON, no markdown."""

            response = call_llm(matching_prompt, st.session_state.model)

            if response:
                try:
                    matches = parse_llm_json(response)

                    st.success("✅ Matching complete!")
                    with st.expander("🔍 View Matches"):
                        st.json(matches)

                    # Build output rows
                    output_rows = []
                    for match in matches.get('matches', []):
                        column_name = match['column_name']
                        matched_cde = match['matched_cde']

                        if matched_cde != "Unknown":
                            cde_checks = catalog_df[catalog_df['Critical Data Element'] == matched_cde]
                            for _, check_row in cde_checks.iterrows():
                                output_row = {
                                    'Domain': check_row['Domain'],
                                    'Critical Data Element': column_name,
                                    'Short description of Critical Data Element': check_row['Short description of Critical Data Element'],
                                    'DQ Dimension': check_row['DQ Dimension'],
                                    'Check name': f"{column_name}_{check_row['DQ Dimension'].lower()}",
                                    'Check Description': check_row['Check Description'],
                                    'SODACL Yaml Check Definition': check_row['SODACL Yaml Check Definition'].replace(matched_cde, column_name)
                                }
                                output_rows.append(output_row)

                    if output_rows:
                        output_df = pd.DataFrame(output_rows)
                        st.session_state.generated_checks = output_df
                        st.session_state.df = output_df  # For marketplace

                        st.success(f"✅ Generated {len(output_df)} checks!")

                        # Generate synthetic test values
                        st.info("🧪 Generating synthetic test values...")
                        synthetic_values = {}
                        unique_cdes = output_df['Critical Data Element'].unique()

                        for cde in unique_cdes:
                            cde_checks = output_df[output_df['Critical Data Element'] == cde]
                            first_check = cde_checks.iloc[0]

                            original_metadata = None
                            for _, meta_row in metadata_df.iterrows():
                                if meta_row.get('Column name', '') == cde:
                                    original_metadata = meta_row
                                    break

                            synthetic_prompt = f"""Generate synthetic test values for data quality testing.

Column: {cde}
Description: {first_check['Short description of Critical Data Element']}
Domain: {first_check['Domain']}

Metadata:
{json.dumps({
    'datatype': original_metadata.get('Datatype', '') if original_metadata is not None else '',
    'related_fields': str(original_metadata.get('Related Fields (Min/Max/Sample Values)', '')) if original_metadata is not None else ''
}, indent=2)}

Generate 10 synthetic values that include:
- 6 VALID values (should pass checks)
- 4 INVALID values (should fail checks - edge cases, out of range, wrong format, nulls)

Respond ONLY with valid JSON:
{{
  "values": [
    {{"value": "32.95", "expected": "PASS", "reason": "Valid latitude"}},
    {{"value": "-91.5", "expected": "FAIL", "reason": "Out of range"}}
  ]
}}

Output only JSON, no markdown."""

                            synth_response = call_llm(synthetic_prompt, st.session_state.model)

                            if synth_response:
                                try:
                                    synth_data = parse_llm_json(synth_response)
                                    synthetic_values[cde] = synth_data.get('values', [])
                                except Exception:
                                    synthetic_values[cde] = []

                        st.session_state.synthetic_values = synthetic_values
                        st.success(f"✅ Generated synthetic test values for {len(synthetic_values)} columns!")
                        st.rerun()
                    else:
                        st.warning("No matches found")

                except Exception as e:
                    st.error(f"Failed to parse: {str(e)}")
                    with st.expander("Raw response"):
                        st.code(response)

# ===========================================================================
# Display results & export
# ===========================================================================
if st.session_state.generated_checks is not None:
    st.divider()
    st.header("📊 Generated Checks")

    generated_df = st.session_state.generated_checks

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.metric("Total Checks", len(generated_df))
    with col2:
        st.metric("CDEs Matched", generated_df['Critical Data Element'].nunique())
    with col3:
        if st.button("💾 Save to Snowflake", key="save_checks_btn", use_container_width=True):
            if save_to_snowflake(generated_df, "DQ_GENERATED_CHECKS"):
                st.success("✅ Saved to table `DQ_GENERATED_CHECKS`")

    st.dataframe(generated_df, use_container_width=True, height=400)

    st.divider()

    st.header("Step 4: Proceed to Marketplace")

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.info("""
        ✅ Checks generated and ready!

        Click 'Next' to:
        - Review and customize checks
        - Select which checks to use
        - Export final YAML
        """)
    with col_btn:
        st.write("")
        st.write("")
        if st.button("➡️ Next: Marketplace", type="primary", use_container_width=True):
            st.switch_page("pages/dq_marketplace.py")
