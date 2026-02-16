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
    """Create a local snowflake.connector connection via SSO (externalbrowser)."""
    import snowflake.connector
    conn = snowflake.connector.connect(
        account=st.session_state.get("sf_account", ""),
        user=st.session_state.get("sf_user", ""),
        authenticator="externalbrowser",
        role=st.session_state.get("sf_role", None) or None,
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
        st.header("⚙️ Snowflake Connection")

        if not st.session_state.connected:
            # ---- Connection form: all fields up front ----
            st.session_state["sf_account"] = st.text_input(
                "Account Identifier *",
                value=st.session_state.get("sf_account", ""),
                placeholder="xy12345.us-east-1"
            )
            st.session_state["sf_user"] = st.text_input(
                "Username *",
                value=st.session_state.get("sf_user", ""),
                placeholder="your_username"
            )
            st.caption("🔑 Authentication: SSO (browser will open)")
            st.session_state["sf_role"] = st.text_input(
                "Role",
                value=st.session_state.get("sf_role", ""),
                placeholder="SYSADMIN"
            )
            st.session_state["sf_warehouse"] = st.text_input(
                "Warehouse",
                value=st.session_state.get("sf_warehouse", ""),
                placeholder="COMPUTE_WH"
            )
            st.session_state["sf_database"] = st.text_input(
                "Database",
                value=st.session_state.get("sf_database", ""),
                placeholder="MY_DATABASE"
            )
            st.session_state["sf_schema"] = st.text_input(
                "Schema",
                value=st.session_state.get("sf_schema", ""),
                placeholder="PUBLIC"
            )

            if st.button("🔐 Connect", use_container_width=True, type="primary"):
                if not st.session_state.sf_account or not st.session_state.sf_user:
                    st.error("Account and Username are required")
                else:
                    try:
                        with st.spinner("Connecting to Snowflake..."):
                            conn = get_local_connection()
                            st.session_state.connection = conn
                            st.session_state.connected = True
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
        else:
            # ---- Connected: show summary ----
            st.success("✅ Connected to Snowflake")
            st.caption(f"**Account:** {st.session_state.get('sf_account', '')}")
            st.caption(f"**User:** {st.session_state.get('sf_user', '')}")
            if st.session_state.get("sf_role"):
                st.caption(f"**Role:** {st.session_state.sf_role}")
            if st.session_state.get("sf_warehouse"):
                st.caption(f"**Warehouse:** {st.session_state.sf_warehouse}")
            if st.session_state.get("sf_database"):
                st.caption(f"**Database:** {st.session_state.sf_database}")
            if st.session_state.get("sf_schema"):
                st.caption(f"**Schema:** {st.session_state.sf_schema}")

            if st.button("🔌 Disconnect", use_container_width=True):
                try:
                    conn = st.session_state.get("connection")
                    if conn:
                        conn.close()
                except Exception:
                    pass
                st.session_state.clear()
                st.experimental_rerun()

        st.divider()
    else:
        # ============================================================
        # SiS MODE — auto-connected
        # ============================================================
        st.success("✅ Connected to Snowflake (SiS)")
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
# Helper — run SQL query and return DataFrame (works in both modes)
# ---------------------------------------------------------------------------
def run_query_df(query):
    """Execute a SQL query and return results as a pandas DataFrame."""
    try:
        if session is not None:
            # SiS: try to_pandas() first, fall back to collect()
            try:
                df = session.sql(query).to_pandas()
            except Exception:
                # Fallback: use collect() and build DataFrame manually
                rows = session.sql(query).collect()
                if rows:
                    cols = list(rows[0].asDict().keys())
                    data = [list(r.asDict().values()) for r in rows]
                    df = pd.DataFrame(data, columns=cols)
                else:
                    df = pd.DataFrame()
        elif st.session_state.get("connection"):
            cur = st.session_state.connection.cursor()
            cur.execute(query)
            columns = [desc[0] for desc in cur.description]
            data = cur.fetchall()
            cur.close()
            df = pd.DataFrame(data, columns=columns)
        else:
            return pd.DataFrame()
        # Normalize column names to lowercase for consistency
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Query error: {e}")
        return pd.DataFrame()


def get_cached_list(cache_key, query, column_name):
    """Fetch and cache a list of values from a SQL query."""
    if cache_key not in st.session_state:
        df = run_query_df(query)
        col = column_name.lower()
        if col in df.columns:
            st.session_state[cache_key] = sorted(
                df[col].dropna().astype(str).unique().tolist()
            )
        elif len(df.columns) > 0:
            # Fallback: use the first column if exact name not found
            st.session_state[cache_key] = sorted(
                df.iloc[:, 0].dropna().astype(str).unique().tolist()
            )
        else:
            st.session_state[cache_key] = []
    return st.session_state[cache_key]


# ---------------------------------------------------------------------------
# Helper — cascading Database → Schema → Table picker
# ---------------------------------------------------------------------------
def snowflake_table_picker(prefix):
    """Render cascading Database / Schema / Table dropdowns.
    Returns a fully-qualified table name like "DB"."SCHEMA"."TABLE", or None.
    """
    # Shared data caches (so both pickers benefit from the same fetch)
    databases = get_cached_list("_sf_databases", "SHOW DATABASES", "name")

    col_db, col_schema, col_table, col_refresh = st.columns([3, 3, 3, 1])

    with col_db:
        selected_db = st.selectbox(
            "Database", [""] + databases, key=f"{prefix}_db",
            format_func=lambda x: x if x else "Select database..."
        )

    # Fetch schemas when a database is selected
    schemas = []
    if selected_db:
        schemas = get_cached_list(
            f"_sf_schemas_{selected_db}",
            f'SHOW SCHEMAS IN DATABASE "{selected_db}"',
            "name"
        )

    with col_schema:
        selected_schema = st.selectbox(
            "Schema", [""] + schemas, key=f"{prefix}_schema",
            format_func=lambda x: x if x else "Select schema..."
        )

    # Fetch tables when a schema is selected
    tables = []
    if selected_db and selected_schema:
        tables = get_cached_list(
            f"_sf_tables_{selected_db}_{selected_schema}",
            f'SHOW TABLES IN SCHEMA "{selected_db}"."{selected_schema}"',
            "name"
        )

    with col_table:
        selected_table = st.selectbox(
            "Table", [""] + tables, key=f"{prefix}_table",
            format_func=lambda x: x if x else "Select table..."
        )

    with col_refresh:
        st.write("")  # spacer to align with selectbox labels
        st.write("")
        if st.button("🔄", key=f"{prefix}_refresh", help="Refresh database/schema/table lists"):
            keys_to_clear = [k for k in list(st.session_state.keys())
                            if k.startswith("_sf_")]
            for k in keys_to_clear:
                del st.session_state[k]
            st.experimental_rerun()

    if selected_db and selected_schema and selected_table:
        return f'"{selected_db}"."{selected_schema}"."{selected_table}"'
    return None


# ---------------------------------------------------------------------------
# Helper — call Snowflake Cortex LLM (works in both modes)
# ---------------------------------------------------------------------------
def call_llm(prompt, model_name):
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
# Helper — call Cortex CLASSIFY_TEXT (column classification)
# ---------------------------------------------------------------------------
def classify_columns(descriptions):
    """Use SNOWFLAKE.CORTEX.CLASSIFY_TEXT to classify column descriptions."""
    categories = ['PII', 'Financial', 'Geographic', 'Temporal',
                  'Categorical', 'Identifier', 'Measurement', 'Text', 'Other']
    categories_sql = ", ".join([f"'{c}'" for c in categories])

    results = {}
    for col_name, description in descriptions.items():
        if not description or str(description).strip() == '':
            results[col_name] = {"label": "Other", "score": 0.0}
            continue
        try:
            escaped = str(description).replace("'", "''")
            query = f"""
            SELECT SNOWFLAKE.CORTEX.CLASSIFY_TEXT(
                '{escaped}',
                ARRAY_CONSTRUCT({categories_sql})
            ) AS classification
            """
            if session is not None:
                result = session.sql(query).collect()
                raw = result[0]["CLASSIFICATION"] if result else "{}"
            elif st.session_state.get("connection"):
                cur = st.session_state.connection.cursor()
                cur.execute(query)
                result = cur.fetchone()
                cur.close()
                raw = result[0] if result else "{}"
            else:
                raw = "{}"

            if isinstance(raw, str):
                parsed = json.loads(raw)
            else:
                parsed = raw
            results[col_name] = parsed
        except Exception:
            results[col_name] = {"label": "Other", "score": 0.0}
    return results


# ---------------------------------------------------------------------------
# Helper — parse JSON from LLM response (handles markdown fences)
# ---------------------------------------------------------------------------
def parse_llm_json(raw):
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
def save_to_snowflake(df, table_name):
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
def load_table(table_name):
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


# ---------------------------------------------------------------------------
# Helper — Snowflake Stage file operations (works in both modes)
# ---------------------------------------------------------------------------
import tempfile, os

def _list_stages():
    """List available stages in the current schema."""
    try:
        conn = st.session_state.get("connection")
        if conn:
            cur = conn.cursor()
            cur.execute("SHOW STAGES")
            cols = [d[0].lower() for d in cur.description]
            rows = cur.fetchall()
            cur.close()
            idx = cols.index("name") if "name" in cols else 0
            return sorted(set(r[idx] for r in rows if r[idx]))
        elif session is not None:
            df = session.sql("SHOW STAGES").to_pandas()
            df.columns = [c.lower() for c in df.columns]
            return sorted(df["name"].dropna().unique().tolist())
    except Exception:
        pass
    return []


def _list_stage_files(stage_name):
    """List files in a Snowflake stage. Returns list of file paths."""
    try:
        query = f'LIST @"{stage_name}"'
        conn = st.session_state.get("connection")
        if conn:
            cur = conn.cursor()
            cur.execute(query)
            cols = [d[0].lower() for d in cur.description]
            rows = cur.fetchall()
            cur.close()
            idx = cols.index("name") if "name" in cols else 0
            return [r[idx] for r in rows if r[idx]]
        elif session is not None:
            df = session.sql(query).to_pandas()
            df.columns = [c.lower() for c in df.columns]
            return df["name"].dropna().tolist()
    except Exception as e:
        st.error(f"Failed to list stage files: {e}")
    return []


def _read_stage_file(stage_name, file_path):
    """Download a file from a Snowflake stage and read it as a DataFrame."""
    tmp_dir = tempfile.mkdtemp()
    file_name = file_path.split("/")[-1]
    local_path = os.path.join(tmp_dir, file_name)

    try:
        # Build the GET command
        # file_path from LIST is like: stage_name/filename.xlsx
        # We need: @"STAGE_NAME"/filename.xlsx
        if file_path.startswith(stage_name):
            relative_path = file_path[len(stage_name):].lstrip("/")
        else:
            relative_path = file_path

        get_query = f'GET @"{stage_name}/{relative_path}" file://{tmp_dir}/'

        conn = st.session_state.get("connection")
        if conn:
            cur = conn.cursor()
            cur.execute(get_query)
            cur.close()
        elif session is not None:
            session.sql(get_query).collect()

        # Find the downloaded file (may have .gz extension)
        downloaded = None
        for f in os.listdir(tmp_dir):
            downloaded = os.path.join(tmp_dir, f)
            break

        if not downloaded or not os.path.exists(downloaded):
            raise FileNotFoundError(f"File not found after GET: {file_name}")

        # Read based on extension
        fname = downloaded.lower()
        if fname.endswith(".csv") or fname.endswith(".csv.gz"):
            return pd.read_csv(downloaded)
        elif fname.endswith(".xlsx") or fname.endswith(".xls"):
            return pd.read_excel(downloaded)
        elif fname.endswith(".parquet"):
            return pd.read_parquet(downloaded)
        else:
            # Try CSV as default
            return pd.read_csv(downloaded)

    except Exception as e:
        st.error(f"Failed to read file from stage: {e}")
        return None
    finally:
        # Cleanup temp files
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def stage_file_picker(prefix, file_types=None):
    """Render a Stage → File picker UI. Returns (stage_name, file_path) or (None, None)."""
    # Cache stages
    cache_key = "_sf_stages"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = _list_stages()
    stages = st.session_state[cache_key]

    col_stage, col_file, col_refresh = st.columns([3, 5, 1])

    with col_stage:
        selected_stage = st.selectbox(
            "Stage",
            [""] + stages,
            key=f"{prefix}_stage",
            format_func=lambda x: x if x else "Select stage..."
        )

    # List files when stage is selected
    files = []
    if selected_stage:
        files_cache_key = f"_sf_stage_files_{selected_stage}"
        if files_cache_key not in st.session_state:
            st.session_state[files_cache_key] = _list_stage_files(selected_stage)
        files = st.session_state[files_cache_key]

        # Filter by file type if specified
        if file_types and files:
            files = [f for f in files
                     if any(f.lower().endswith(ext) for ext in file_types)]

    with col_file:
        # Show just filenames for readability
        display_files = [f.split("/")[-1] for f in files]
        file_map = dict(zip(display_files, files))

        selected_display = st.selectbox(
            "File",
            [""] + display_files,
            key=f"{prefix}_stage_file",
            format_func=lambda x: x if x else "Select file..."
        )

    with col_refresh:
        st.write("")  # spacer
        st.write("")
        if st.button("🔄", key=f"{prefix}_stage_refresh", help="Refresh stage/file lists"):
            for k in list(st.session_state.keys()):
                if k.startswith("_sf_stage"):
                    del st.session_state[k]
            st.experimental_rerun()

    if selected_stage and selected_display:
        return selected_stage, file_map[selected_display]
    return None, None


# ===========================================================================
# Guard: must be connected
# ===========================================================================
if not st.session_state.connected:
    st.info("👈 Please connect to Snowflake in the sidebar")
    st.stop()


# ===========================================================================
# Step 1 — Load DQ Checks Catalog  (Excel / CSV from Stage only)
# ===========================================================================
st.header("📘 Step 1: Load DQ Checks Catalog")
st.caption("Upload your DQ rules catalog from a Snowflake Stage (Excel or CSV)")

cat_stage, cat_file = stage_file_picker(
    "catalog", file_types=[".xlsx", ".xls", ".csv", ".csv.gz"]
)

if st.button("📥 Load Catalog", key="load_catalog_stage_btn", type="primary"):
    if cat_stage and cat_file:
        try:
            with st.spinner(f"Loading from @{cat_stage}..."):
                catalog_df = _read_stage_file(cat_stage, cat_file)
                if catalog_df is not None:
                    st.session_state.catalog_df = catalog_df
                    st.success(f"✅ Catalog loaded: **{len(catalog_df)}** checks")
        except Exception as e:
            st.error(f"Failed to load file: {e}")
    else:
        st.warning("Please select a stage and file first")

if st.session_state.catalog_df is not None:
    with st.expander(f"📋 View Catalog ({len(st.session_state.catalog_df)} rows)", expanded=False):
        st.dataframe(st.session_state.catalog_df, use_container_width=True)

st.divider()

# ===========================================================================
# Step 2 — Load Collibra Metadata  (Snowflake Table or Stage)
# ===========================================================================
st.header("📗 Step 2: Load Collibra Metadata")

metadata_source = st.radio(
    "Choose metadata source",
    ["Snowflake Table", "Snowflake Stage (Excel/CSV)"],
    horizontal=True,
    key="metadata_source",
    help="Load metadata from a Snowflake table or from an Excel/CSV file on a stage"
)

metadata_df = None
if metadata_source == "Snowflake Table":
    st.caption("Browse and select a table from your Snowflake account")
    selected_metadata_table = snowflake_table_picker("metadata")

    if st.button("📥 Load Metadata", key="load_metadata_btn", type="primary"):
        if selected_metadata_table:
            try:
                with st.spinner("Loading metadata..."):
                    metadata_df = load_table(selected_metadata_table)
                    st.session_state.collibra_metadata = metadata_df
                    st.success(f"✅ Metadata loaded: **{len(metadata_df)}** columns")
            except Exception as e:
                st.error(f"Failed to load table: {e}")
        else:
            st.warning("Please select a database, schema, and table first")
else:
    st.caption("Select a stage and pick an Excel/CSV file")
    meta_stage, meta_file = stage_file_picker(
        "metadata", file_types=[".xlsx", ".xls", ".csv", ".csv.gz"]
    )

    if st.button("📥 Load from Stage", key="load_metadata_stage_btn", type="primary"):
        if meta_stage and meta_file:
            try:
                with st.spinner(f"Loading from @{meta_stage}..."):
                    metadata_df = _read_stage_file(meta_stage, meta_file)
                    if metadata_df is not None:
                        st.session_state.collibra_metadata = metadata_df
                        st.success(f"✅ Metadata loaded: **{len(metadata_df)}** columns")
            except Exception as e:
                st.error(f"Failed to load file: {e}")
        else:
            st.warning("Please select a stage and file first")

# Use whatever is in session state
if st.session_state.collibra_metadata is not None:
    metadata_df = st.session_state.collibra_metadata
    with st.expander(f"📋 View Metadata ({len(metadata_df)} rows)", expanded=False):
        st.dataframe(metadata_df, use_container_width=True)

st.divider()


# ===========================================================================
# Step 2.5 — AI Data Profiling (Cortex-powered)
# ===========================================================================
if st.session_state.collibra_metadata is not None:
    st.header("Step 2.5: AI Data Profiling")
    st.caption("Use Snowflake Cortex AI to analyze your metadata and get intelligent recommendations")

    col_profile, col_classify = st.columns(2)

    with col_profile:
        if st.button("🔬 AI Profile Data", key="ai_profile_btn", use_container_width=True):
            with st.spinner("Cortex AI is profiling your metadata..."):
                meta = st.session_state.collibra_metadata
                sample = meta.head(20).to_dict('records')
                columns_list = meta.columns.tolist()

                profiling_prompt = f"""You are a data quality expert. Analyze this metadata and provide a data profiling report.

Columns available: {columns_list}

Sample data (first 20 rows):
{json.dumps(sample, indent=2, default=str)}

For each column, provide:
1. Inferred data type and pattern
2. Potential quality issues
3. Recommended DQ checks
4. Completeness assessment

Also provide an overall data quality readiness score (0-100).

Respond ONLY with valid JSON:
{{
    "overall_score": 85,
    "summary": "Brief overall assessment...",
    "columns": [
        {{
            "name": "column_name",
            "inferred_type": "string/numeric/date/etc",
            "quality_issues": ["issue1", "issue2"],
            "recommended_checks": ["completeness", "validity"],
            "completeness": "high/medium/low"
        }}
    ],
    "recommendations": ["rec1", "rec2", "rec3"]
}}

Output only JSON, no markdown."""

                response = call_llm(profiling_prompt, st.session_state.model)
                if response:
                    try:
                        profile = parse_llm_json(response)
                        st.session_state.ai_profile = profile
                    except Exception as e:
                        st.error(f"Failed to parse profiling results: {e}")
                        with st.expander("Raw response"):
                            st.code(response)

    with col_classify:
        if st.button("🏷️ AI Classify Columns", key="ai_classify_btn", use_container_width=True,
                     help="Uses CORTEX.CLASSIFY_TEXT to categorize each column"):
            with st.spinner("Cortex CLASSIFY_TEXT is categorizing columns..."):
                meta = st.session_state.collibra_metadata
                desc_col = None
                for c in ['Column description', 'Description', 'COLUMN_DESCRIPTION', 'description']:
                    if c in meta.columns:
                        desc_col = c
                        break

                name_col = None
                for c in ['Column name', 'COLUMN_NAME', 'column_name', 'Name']:
                    if c in meta.columns:
                        name_col = c
                        break

                if desc_col and name_col:
                    descriptions = {}
                    for _, row in meta.head(20).iterrows():
                        col_name = str(row[name_col])
                        descriptions[col_name] = str(row.get(desc_col, col_name))

                    classifications = classify_columns(descriptions)
                    st.session_state.ai_classifications = classifications
                else:
                    st.warning("Could not find column name/description fields in metadata. "
                              "Expected columns like 'Column name' and 'Column description'.")

    # Display profiling results
    if st.session_state.get("ai_profile"):
        profile = st.session_state.ai_profile

        col_score, col_summary = st.columns([1, 3])
        with col_score:
            score = profile.get("overall_score", 0)
            st.metric("Quality Readiness", f"{score}/100")
        with col_summary:
            st.info(profile.get("summary", "No summary available"))

        with st.expander("📊 Column-Level Analysis", expanded=True):
            col_data = profile.get("columns", [])
            if col_data:
                profile_df = pd.DataFrame(col_data)
                st.dataframe(profile_df, use_container_width=True)

        with st.expander("💡 AI Recommendations"):
            for rec in profile.get("recommendations", []):
                st.markdown(f"- {rec}")

    # Display classification results
    if st.session_state.get("ai_classifications"):
        with st.expander("🏷️ Column Classifications (via CORTEX.CLASSIFY_TEXT)"):
            cls_data = []
            for col_name, cls in st.session_state.ai_classifications.items():
                cls_data.append({
                    "Column": col_name,
                    "Category": cls.get("label", "Unknown"),
                    "Confidence": f"{cls.get('score', 0):.2f}" if isinstance(cls.get('score'), (int, float)) else str(cls.get('score', ''))
                })
            if cls_data:
                st.dataframe(pd.DataFrame(cls_data), use_container_width=True)

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
                        st.experimental_rerun()
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

    col1, col2, col3, col4 = st.columns([2, 2, 1, 2])
    with col1:
        st.metric("Total Checks", len(generated_df))
    with col2:
        st.metric("CDEs Matched", generated_df['Critical Data Element'].nunique())
    with col3:
        if st.button("💾 Save to Snowflake", key="save_checks_btn", use_container_width=True):
            if save_to_snowflake(generated_df, "DQ_GENERATED_CHECKS"):
                st.success("✅ Saved to table `DQ_GENERATED_CHECKS`")
    with col4:
        if st.button("📈 AI Coverage Assessment", key="ai_coverage_btn", use_container_width=True):
            with st.spinner("Cortex AI is assessing DQ coverage..."):
                checks_summary = []
                for _, row in generated_df.iterrows():
                    checks_summary.append({
                        "cde": row["Critical Data Element"],
                        "domain": row["Domain"],
                        "dimension": row["DQ Dimension"],
                        "check": row["Check name"]
                    })

                coverage_prompt = f"""You are a data quality expert. Assess the completeness and coverage of these generated DQ checks.

Generated Checks:
{json.dumps(checks_summary, indent=2)}

Standard DQ Dimensions: Completeness, Validity, Accuracy, Consistency, Timeliness, Uniqueness

Evaluate:
1. Which CDEs have good coverage across dimensions?
2. Which CDEs are missing important checks?
3. Are there any DQ dimensions underrepresented?
4. Overall coverage score (0-100)
5. Specific recommendations to improve coverage

Respond ONLY with valid JSON:
{{
    "coverage_score": 75,
    "summary": "Overall assessment...",
    "well_covered": ["CDE1 has completeness + validity + accuracy"],
    "gaps": ["CDE2 missing uniqueness check", "No timeliness checks found"],
    "dimension_coverage": {{
        "Completeness": "good",
        "Validity": "good",
        "Accuracy": "partial",
        "Consistency": "missing",
        "Timeliness": "missing",
        "Uniqueness": "partial"
    }},
    "recommendations": ["Add uniqueness checks for ID columns", "Consider timeliness checks for date fields"]
}}

Output only JSON, no markdown."""

                coverage_response = call_llm(coverage_prompt, st.session_state.model)
                if coverage_response:
                    try:
                        coverage = parse_llm_json(coverage_response)
                        st.session_state.ai_coverage = coverage
                    except Exception as e:
                        st.error(f"Failed to parse coverage assessment: {e}")
                        with st.expander("Raw response"):
                            st.code(coverage_response)

    st.dataframe(generated_df, use_container_width=True, height=400)

    # Show AI Coverage Assessment results
    if st.session_state.get("ai_coverage"):
        coverage = st.session_state.ai_coverage
        st.divider()
        st.subheader("📈 AI Coverage Assessment")

        cov_col1, cov_col2 = st.columns([1, 3])
        with cov_col1:
            cov_score = coverage.get("coverage_score", 0)
            st.metric("Coverage Score", f"{cov_score}/100")
        with cov_col2:
            st.info(coverage.get("summary", ""))

        cov_col_left, cov_col_right = st.columns(2)
        with cov_col_left:
            st.markdown("**Well Covered:**")
            for item in coverage.get("well_covered", []):
                st.markdown(f"✅ {item}")

            st.markdown("**Dimension Coverage:**")
            dim_cov = coverage.get("dimension_coverage", {})
            for dim, status in dim_cov.items():
                icon = "🟢" if status == "good" else "🟡" if status == "partial" else "🔴"
                st.markdown(f"{icon} **{dim}**: {status}")

        with cov_col_right:
            st.markdown("**Gaps Identified:**")
            for gap in coverage.get("gaps", []):
                st.markdown(f"⚠️ {gap}")

            st.markdown("**Recommendations:**")
            for rec in coverage.get("recommendations", []):
                st.markdown(f"💡 {rec}")

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
            st.info("👈 Navigate to **DQ Marketplace** in the sidebar")
