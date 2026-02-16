import streamlit as st
import pandas as pd
import json
import tempfile
from pathlib import Path
from typing import Optional

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
    """Create a local snowflake.connector connection.
    
    Uses username/password for local testing.
    SSO is used automatically in Snowflake Streamlit environment.
    """
    import snowflake.connector
    
    # Build connection parameters
    conn_params = {
        "account": st.session_state.get("sf_account", ""),
        "user": st.session_state.get("sf_user", ""),
        "password": st.session_state.get("sf_password", ""),
        "role": st.session_state.get("sf_role", None) or None,
        "warehouse": st.session_state.get("sf_warehouse", None) or None,
        "database": st.session_state.get("sf_database", None) or None,
        "schema": st.session_state.get("sf_schema", None) or None,
    }
    
    conn = snowflake.connector.connect(**conn_params)
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
if "catalog_context" not in st.session_state:
    st.session_state.catalog_context = {
        "database": st.session_state.get("sf_database", ""),
        "schema": st.session_state.get("sf_schema", ""),
        "stage": "",
        "stage_prefix": ""
    }
if "metadata_context" not in st.session_state:
    st.session_state.metadata_context = {
        "database": st.session_state.get("sf_database", ""),
        "schema": st.session_state.get("sf_schema", ""),
        "stage": "",
        "stage_prefix": ""
    }

# Pre-populate context from SiS session if available
if session is not None:
    st.session_state.connected = True
    st.session_state.session = session
    # Read current context from the active session (only once)
    if "ctx_initialized" not in st.session_state:
        try:
            st.session_state.setdefault("sf_account", session.sql("SELECT CURRENT_ACCOUNT()").collect()[0][0] or "")
            st.session_state.setdefault("sf_user", session.sql("SELECT CURRENT_USER()").collect()[0][0] or "")
            st.session_state.setdefault("sf_role", session.sql("SELECT CURRENT_ROLE()").collect()[0][0] or "")
            st.session_state.setdefault("sf_warehouse", session.sql("SELECT CURRENT_WAREHOUSE()").collect()[0][0] or "")
            st.session_state.setdefault("sf_database", session.sql("SELECT CURRENT_DATABASE()").collect()[0][0] or "")
            st.session_state.setdefault("sf_schema", session.sql("SELECT CURRENT_SCHEMA()").collect()[0][0] or "")
        except Exception:
            pass
        st.session_state.ctx_initialized = True


# ---------------------------------------------------------------------------
# Sidebar — Snowflake Connection (unified for SiS + Local)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🔌 Snowflake Connection")
    
    # Show authentication mode
    if IS_LOCAL:
        st.caption("🖥️ **Local Mode:** Username/Password authentication")
    else:
        st.caption("☁️ **Snowflake Streamlit:** SSO authentication")

    if session is not None:
        st.success("✅ Connected natively to Snowflake (SSO)")
    elif st.session_state.connected:
        st.success("✅ Connected to Snowflake")
    else:
        if IS_LOCAL:
            st.info("Enter credentials (username/password) and connect")
        else:
            st.info("Enter credentials and connect")

    # Context text inputs — always visible
    st.session_state["sf_account"] = st.text_input(
        "🏢 Account",
        value=st.session_state.get("sf_account", ""),
        placeholder="xy12345.us-east-1"
    )
    st.session_state["sf_user"] = st.text_input(
        "👤 User",
        value=st.session_state.get("sf_user", ""),
        placeholder="your_username"
    )
    
    # Password field — only for local environment
    if IS_LOCAL:
        st.session_state["sf_password"] = st.text_input(
            "🔑 Password",
            value=st.session_state.get("sf_password", ""),
            type="password",
            placeholder="your_password"
        )
    st.session_state["sf_role"] = st.text_input(
        "🎭 Role",
        value=st.session_state.get("sf_role", ""),
        placeholder="PUBLIC"
    )
    st.session_state["sf_warehouse"] = st.text_input(
        "🏭 Warehouse",
        value=st.session_state.get("sf_warehouse", ""),
        placeholder="COMPUTE_WH"
    )
    st.session_state["sf_database"] = st.text_input(
        "🗄️ Database",
        value=st.session_state.get("sf_database", ""),
        placeholder="MY_DATABASE"
    )
    st.session_state["sf_schema"] = st.text_input(
        "📂 Source Schema",
        value=st.session_state.get("sf_schema", ""),
        placeholder="PUBLIC"
    )

    # Test Connection / Connect button
    if IS_LOCAL and not st.session_state.connected:
        if st.button("🔐 Connect", use_container_width=True, type="primary"):
            if not st.session_state.sf_account or not st.session_state.sf_user:
                st.error("Account and User are required")
            elif not st.session_state.get("sf_password"):
                st.error("Password is required for local connection")
            else:
                try:
                    with st.spinner("Connecting to Snowflake..."):
                        conn = get_local_connection()
                        st.session_state.connection = conn
                        st.session_state.connected = True
                        st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {e}")
    else:
        if st.button("⚡ Test Connection", use_container_width=True):
            try:
                # Simple validation query - sidebar values are informational only
                if session is not None:
                    session.sql("SELECT 1").collect()
                elif st.session_state.get("connection"):
                    cur = st.session_state.connection.cursor()
                    cur.execute("SELECT 1")
                    cur.close()

                st.success("✅ Connection OK!")
            except Exception as e:
                st.error(f"Connection test failed: {e}")

        if IS_LOCAL and st.session_state.connected:
            if st.button("🔌 Disconnect", use_container_width=True):
                try:
                    conn = st.session_state.get("connection")
                    if conn:
                        conn.close()
                except Exception:
                    pass
                st.session_state.clear()
                st.rerun()

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
    # Clear previous errors
    if "_last_query_error" in st.session_state:
        del st.session_state["_last_query_error"]
    
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
            st.session_state["_last_query_error"] = "Not connected to Snowflake"
            return pd.DataFrame()
        # Normalize column names to lowercase and strip quotes
        if not df.empty:
            df.columns = [c.lower().strip('"').strip("'") for c in df.columns]
        return df
    except Exception as e:
        error_msg = str(e)
        st.session_state["_last_query_error"] = error_msg
        st.error(f"Query error: {error_msg}")
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
# Helper — Table picker (uses sidebar Database + Schema context)
# ---------------------------------------------------------------------------
def snowflake_table_picker(prefix):
    """(Deprecated) Placeholder to maintain backwards compatibility."""
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


def resolve_table_name(raw_name: str, context: dict) -> str:
    """Return a fully-qualified table name using the provided context."""
    cleaned = (raw_name or "").strip()
    if not cleaned:
        raise ValueError("Please enter a table name.")
    if "." in cleaned:
        return cleaned
    database = (context.get("database") or "").strip()
    schema = (context.get("schema") or "").strip()
    if database and schema:
        return f"{database}.{schema}.{cleaned}"
    raise ValueError("Set database and schema in the step context or provide a fully-qualified name.")


def _normalize_stage(stage_name: str) -> str:
    stage_name = (stage_name or "").strip()
    if not stage_name:
        raise ValueError("Stage name is required.")
    return stage_name if stage_name.startswith("@") else f"@{stage_name}"


def list_stage_entries(stage_name: str, prefix: str = "") -> list:
    """Return parsed rows from LIST @stage for UI dropdowns."""
    if not stage_name:
        return []
    reference = _normalize_stage(stage_name)
    if prefix:
        clean_prefix = prefix.strip().lstrip('/')
        if clean_prefix:
            reference = f"{reference}/{clean_prefix}"
    
    sql_query = f"LIST {reference}"
    df = run_query_df(sql_query)
    
    # Store debug info in session state
    st.session_state["_stage_list_debug"] = {
        "query": sql_query,
        "reference": reference,
        "row_count": len(df),
        "columns": list(df.columns) if not df.empty else [],
        "has_name_column": "name" in df.columns if not df.empty else False
    }
    
    if df.empty or "name" not in df.columns:
        return []

    entries = []
    for _, row in df.iterrows():
        file_name = row.get("name", "")
        label = file_name.split("/")[-1]
        meta_parts = []
        last_modified = row.get("last_modified")
        if last_modified:
            try:
                ts = pd.to_datetime(last_modified)
                meta_parts.append(ts.strftime("%Y-%m-%d %H:%M"))
            except Exception:
                meta_parts.append(str(last_modified))
        size_bytes = row.get("size")
        if size_bytes:
            try:
                size = int(size_bytes)
                if size >= 1048576:
                    meta_parts.append(f"{size / 1048576:.1f} MB")
                elif size >= 1024:
                    meta_parts.append(f"{size / 1024:.1f} KB")
                else:
                    meta_parts.append(f"{size} B")
            except Exception:
                meta_parts.append(str(size_bytes))
        display = " • ".join([label] + meta_parts) if meta_parts else label
        entries.append({
            "value": f"@{file_name}",
            "label": display
        })
    return entries


def list_tables_for_context(context: dict, cache_label: str) -> list:
    """Return formatted table entries for the provided database/schema."""
    database = (context.get("database") or "").strip()
    schema = (context.get("schema") or "").strip()
    if not database or not schema:
        return []

    cache_key = f"_tables_{cache_label}_{database}_{schema}"
    debug_key = f"_tables_debug_{cache_label}"
    if cache_key not in st.session_state:
        sql_query = f'SHOW TABLES IN SCHEMA {database}.{schema}'
        st.session_state[debug_key] = {
            "database": database,
            "schema": schema,
            "query": sql_query,
            "status": "querying"
        }
        df = run_query_df(sql_query)
        
        # Check if there was an error
        query_error = st.session_state.get("_last_query_error")
        if query_error:
            st.session_state[debug_key]["status"] = "error"
            st.session_state[debug_key]["error"] = query_error
            st.session_state[debug_key]["row_count"] = 0
            st.session_state[debug_key]["columns"] = []
            st.session_state[cache_key] = []
            return []
        
        entries = []
        if not df.empty and "name" in df.columns:
            for _, row in df.iterrows():
                name = row.get("name")
                if not name:
                    continue
                comment = row.get("comment")
                created = row.get("created_on") or row.get("created")
                label_parts = [name]
                if comment:
                    label_parts.append(str(comment))
                elif created:
                    try:
                        ts = pd.to_datetime(created)
                        label_parts.append(ts.strftime("%Y-%m-%d"))
                    except Exception:
                        label_parts.append(str(created))
                entries.append({
                    "value": name,
                    "label": " • ".join(label_parts)
                })
            st.session_state[debug_key]["status"] = "success"
            st.session_state[debug_key]["table_count"] = len(entries)
        else:
            st.session_state[debug_key]["status"] = "empty"
            st.session_state[debug_key]["columns"] = list(df.columns)
            st.session_state[debug_key]["row_count"] = len(df)
        st.session_state[cache_key] = entries

    return st.session_state.get(cache_key, [])


def load_stage_file(stage_reference: str) -> Optional[pd.DataFrame]:
    """Download a staged file to a temp directory and load it into pandas."""
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_dir = Path(tmp_dir)
            if session is not None:
                session.file.get(stage_reference, str(target_dir))
            elif st.session_state.get("connection"):
                cur = st.session_state.connection.cursor()
                safe_target = target_dir.as_posix() + "/"
                cur.execute(f"GET {stage_reference} 'file://{safe_target}'")
                cur.close()
            else:
                raise Exception("Not connected to Snowflake.")

            local_files = list(target_dir.glob("*"))
            if not local_files:
                raise FileNotFoundError("Stage download returned no files.")
            file_path = local_files[0]
            suffixes = [s.lower() for s in file_path.suffixes]
            if suffixes[-2:] == [".csv", ".gz"]:
                return pd.read_csv(file_path, compression="gzip")
            if suffixes and suffixes[-1] in (".csv", ".txt"):
                return pd.read_csv(file_path)
            if suffixes and suffixes[-1] in (".xls", ".xlsx"):
                return pd.read_excel(file_path)
            if suffixes and suffixes[-1] == ".parquet":
                return pd.read_parquet(file_path)
            return pd.read_csv(file_path)
    except Exception as exc:
        st.error(f"Stage load failed: {exc}")
        return None




# ===========================================================================
# Guard: must be connected
# ===========================================================================
if not st.session_state.connected:
    st.info("👈 Please connect to Snowflake in the sidebar")
    st.stop()


# ===========================================================================
# Step 1 — Load DQ Checks Catalog (Stage only)
# ===========================================================================
st.header("📘 Step 1: Load DQ Checks Catalog")

catalog_context = st.session_state.catalog_context
st.caption("Provide the stage that contains your catalog file.")
cat_stage_col, cat_prefix_col = st.columns([2, 1])
catalog_context["stage"] = cat_stage_col.text_input(
    "Catalog Stage (DB.SCHEMA.STAGE)",
    value=catalog_context.get("stage", ""),
    key="catalog_stage_input",
    placeholder="DEV_GDP_UTIL_DB.EXT_VOL.CATALOG_STAGE"
)
catalog_context["stage_prefix"] = cat_prefix_col.text_input(
    "Stage Folder (optional)",
    value=catalog_context.get("stage_prefix", ""),
    key="catalog_stage_prefix_input",
    placeholder="folder/subfolder"
)

if not catalog_context.get("stage"):
    st.warning("Provide a fully-qualified stage name (e.g., DEV_GDP_UTIL_DB.EXT_VOL.STAGE_NAME) to list files.")
else:
    try:
        entries = list_stage_entries(
            catalog_context.get("stage", ""),
            catalog_context.get("stage_prefix", "")
        )
    except Exception as exc:
        entries = []
        st.error(f"Error listing stage: {exc}")

    # Show debug info if no files found
    stage_debug = st.session_state.get("_stage_list_debug")
    if not entries and stage_debug:
        with st.expander("🔍 Debug: stage listing returned no files", expanded=True):
            st.code(stage_debug.get('query', ''), language='sql')
            st.write(f"**Stage reference:** {stage_debug.get('reference')}")
            st.write(f"**Rows returned:** {stage_debug.get('row_count', 0)}")
            st.write(f"**Columns returned:** {', '.join(stage_debug.get('columns', []))}")
            st.write(f"**Has 'name' column:** {stage_debug.get('has_name_column', False)}")
            if stage_debug.get('row_count', 0) == 0:
                st.info("✓ Query executed successfully but returned 0 files. The stage may be empty or the folder path may not exist.")
            else:
                st.warning("Query returned rows but 'name' column was not found. This is unusual.")
            st.caption("Try running the SQL above in a Snowflake worksheet to verify the stage exists and contains files.")

    if entries:
        option_values = [entry["value"] for entry in entries]
        label_map = {entry["value"]: entry["label"] for entry in entries}
        selected_stage_file = st.selectbox(
            "Select catalog file",
            [""] + option_values,
            format_func=lambda val: label_map.get(val, "Select file..." if val == "" else val),
            key="catalog_stage_file_select"
        )
        if st.button("📥 Load Catalog", key="load_catalog_stage_btn", type="primary"):
            if selected_stage_file:
                df = load_stage_file(selected_stage_file)
                if df is not None:
                    st.session_state.catalog_df = df
                    st.success(f"✅ Catalog loaded from stage: **{len(df)}** checks")
            else:
                st.warning("Choose a file from the dropdown first.")

if st.session_state.catalog_df is not None:
    with st.expander(f"📋 View Catalog ({len(st.session_state.catalog_df)} rows)", expanded=False):
        st.dataframe(st.session_state.catalog_df, use_container_width=True)

st.divider()

# ===========================================================================
# Step 2 — Load Collibra Metadata
# ===========================================================================
st.header("📗 Step 2: Load Collibra Metadata")

# Show current Snowflake session context
if session is not None or st.session_state.get("connection"):
    with st.expander("ℹ️ Current Snowflake Session Context", expanded=False):
        try:
            context_query = """
            SELECT 
                CURRENT_ROLE() as current_role,
                CURRENT_DATABASE() as current_database,
                CURRENT_SCHEMA() as current_schema,
                CURRENT_WAREHOUSE() as current_warehouse
            """
            context_df = run_query_df(context_query)
            if not context_df.empty:
                st.info("**Your actual session is running as:**")
                col1, col2 = st.columns(2)
                col1.metric("Role", context_df["current_role"].iloc[0])
                col1.metric("Database", context_df["current_database"].iloc[0] or "(none)")
                col2.metric("Schema", context_df["current_schema"].iloc[0] or "(none)")
                col2.metric("Warehouse", context_df["current_warehouse"].iloc[0])
                st.caption("⚠️ If these don't match your expectations, the session may be using a different role than you expected. The text inputs in the sidebar are for display only when running in Snowflake Streamlit.")
        except Exception as e:
            st.warning(f"Could not fetch session context: {e}")

metadata_context = st.session_state.metadata_context
st.caption("Metadata context (independent from Step 1).")
meta_db_col, meta_schema_col = st.columns(2)
metadata_context["database"] = meta_db_col.text_input(
    "Metadata Database",
    value=metadata_context.get("database", ""),
    key="metadata_db_input"
)
metadata_context["schema"] = meta_schema_col.text_input(
    "Metadata Schema",
    value=metadata_context.get("schema", ""),
    key="metadata_schema_input"
)
meta_stage_col, meta_prefix_col = st.columns([2, 1])
metadata_context["stage"] = meta_stage_col.text_input(
    "Metadata Stage (DB.SCHEMA.STAGE)",
    value=metadata_context.get("stage", ""),
    key="metadata_stage_input",
    placeholder="DEV_GDP_UTIL_DB.EXT_VOL.METADATA_STAGE"
)
metadata_context["stage_prefix"] = meta_prefix_col.text_input(
    "Stage Folder (optional)",
    value=metadata_context.get("stage_prefix", ""),
    key="metadata_stage_prefix_input",
    placeholder="folder/subfolder"
)

metadata_source = st.radio(
    "Choose metadata source",
    ["Snowflake Table", "Snowflake Stage (Excel/CSV)"],
    horizontal=True,
    key="metadata_source",
    help="Load metadata from a Snowflake table or from an Excel/CSV file on a stage"
)

metadata_df = None
if metadata_source == "Snowflake Table":
    metadata_table_entries = list_tables_for_context(metadata_context, "metadata")
    selected_metadata_table = ""
    meta_select_col, meta_refresh_col = st.columns([7, 1])
    with meta_select_col:
        if metadata_table_entries:
            option_values = [entry["value"] for entry in metadata_table_entries]
            label_map = {entry["value"]: entry["label"] for entry in metadata_table_entries}
            selected_metadata_table = st.selectbox(
                "Choose metadata table",
                [""] + option_values,
                key="metadata_table_select",
                format_func=lambda val: label_map.get(val, "Select table..." if val == "" else val)
            )
        else:
            st.info("No tables found for the provided metadata context.")
    with meta_refresh_col:
        st.write("\u200b")
        if st.button("🔄", key="metadata_tables_refresh", help="Refresh table list"):
            keys_to_clear = [k for k in list(st.session_state.keys()) if k.startswith("_tables_metadata_") or k.startswith("_tables_debug_metadata")]
            for k in keys_to_clear:
                del st.session_state[k]
            st.rerun()

    debug_info = st.session_state.get("_tables_debug_metadata")
    if debug_info and debug_info.get("status") in ["empty", "error"]:
        is_error = debug_info.get("status") == "error"
        with st.expander(f"🔍 Debug: {('Query failed' if is_error else 'no tables found')}", expanded=True):
            st.code(debug_info.get('query', ''), language='sql')
            st.write(f"**Database:** {debug_info.get('database')}")
            st.write(f"**Schema:** {debug_info.get('schema')}")
            
            if is_error:
                st.error(f"**Error:** {debug_info.get('error', 'Unknown error')}")
                st.warning("""**Common causes:**
- The database or schema does not exist
- Your current role lacks USAGE privilege on the database or schema
- The schema name has special characters and needs quotes

**To fix:**
1. Run `SHOW SCHEMAS IN DATABASE {db}` to verify the schema exists and check exact spelling/casing
2. Run `SHOW GRANTS ON SCHEMA {db}.{schema}` to verify your role has USAGE privilege
3. Try entering the table name manually in the text box below if you know it exists""")
            else:
                st.write(f"**Rows returned:** {debug_info.get('row_count', 0)}")
                st.write(f"**Columns returned:** {', '.join(debug_info.get('columns', []))}")
                st.info("✓ Query executed successfully but returned 0 tables. The schema may be empty or your role lacks USAGE privileges on objects within it.")
            
            st.caption("💡 Try running the SQL above in a Snowflake worksheet with your current role to confirm access.")

    metadata_table = st.text_input(
        "Or enter table name",
        value=st.session_state.get("metadata_table_name", ""),
        placeholder="COLLIBRA_METADATA",
        key="metadata_table_name_input"
    )
    if st.button("📥 Load Metadata", key="load_metadata_btn", type="primary"):
        try:
            table_choice = metadata_table.strip() or selected_metadata_table.strip()
            if not table_choice:
                raise ValueError("Select or enter a metadata table name.")
            table_name = resolve_table_name(table_choice, metadata_context)
            metadata_df = load_table(table_name)
            st.session_state.collibra_metadata = metadata_df
            st.session_state.metadata_table_name = table_choice
            st.success(f"✅ Metadata loaded: **{len(metadata_df)}** columns")
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Failed to load table: {e}")
else:
    if not metadata_context.get("stage"):
        st.warning("Provide a fully-qualified stage name to list files.")
    else:
        entries = list_stage_entries(
            metadata_context.get("stage", ""),
            metadata_context.get("stage_prefix", "")
        )
        if entries:
            option_values = [entry["value"] for entry in entries]
            label_map = {entry["value"]: entry["label"] for entry in entries}
            selected_metadata_stage_file = st.selectbox(
                "Select metadata file",
                [""] + option_values,
                format_func=lambda val: label_map.get(val, "Select file..." if val == "" else val),
                key="metadata_stage_file_select"
            )
            if st.button("📥 Load from Stage", key="load_metadata_stage_btn", type="primary"):
                if selected_metadata_stage_file:
                    df = load_stage_file(selected_metadata_stage_file)
                    if df is not None:
                        st.session_state.collibra_metadata = df
                        st.success(f"✅ Metadata loaded from stage: **{len(df)}** columns")
                else:
                    st.warning("Choose a file from the dropdown first.")
        else:
            st.info("No files found for the provided stage/prefix.")

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
