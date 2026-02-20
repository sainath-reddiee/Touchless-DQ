import streamlit as st
import pandas as pd
import json
import tempfile
from pathlib import Path
from typing import Optional
import streamlit.components.v1 as components

# App configuration — only in main file
st.set_page_config(
    page_title="Touchless DQ",
    page_icon="📤",
    layout="wide"
)

# Hide default sidebar nav — replaced by top step-indicator bar
st.markdown('<style>[data-testid="stSidebarNav"]{display:none;}</style>', unsafe_allow_html=True)


def _nav_to(page_file):
    """Navigate to another page — works across all Streamlit versions."""
    try:
        st.switch_page(page_file)
    except AttributeError:
        slug = page_file.replace("pages/", "").replace(".py", "")
        components.html(f"""<script>
var links = window.parent.document.querySelectorAll('[data-testid="stSidebarNav"] a');
for (var i = 0; i < links.length; i++) {{
    if (links[i].href.toLowerCase().includes('{slug}')) {{
        links[i].click();
        break;
    }}
}}
</script>""", height=0, width=0)


def render_top_nav(current_page):
    """Render a step-indicator navigation bar at the top."""
    steps = [
        ("📤", "Upload", "collibra_upload", "collibra_upload.py"),
        ("✅", "Marketplace", "dq_marketplace", "pages/dq_marketplace.py"),
        ("📊", "Review", "dq_review_page", "pages/dq_review_page.py"),
    ]
    cols = st.columns(len(steps))
    for i, (icon, label, slug, file_path) in enumerate(steps):
        with cols[i]:
            btn_label = f"{icon} Step {i + 1}: {label}"
            if slug == current_page:
                st.button(btn_label, disabled=True, use_container_width=True,
                          type="primary", key=f"topnav_{slug}")
            else:
                if st.button(btn_label, use_container_width=True,
                             key=f"topnav_{slug}"):
                    _nav_to(file_path)
    st.markdown("---")


st.title("📤 Upload Collibra Metadata & Catalog")
render_top_nav("collibra_upload")

# ---------------------------------------------------------------------------
# Snowflake Session — auto-detect SiS vs local environment
# ---------------------------------------------------------------------------
IS_LOCAL = False

def get_snowflake_session():
    """Return a Snowpark session. Works in both SiS and local dev.

    In SiS: get_active_session() should always succeed.
    Locally: the import fails with ImportError OR get_active_session()
    raises error 1403 (no default session) if snowpark is installed
    locally but no SiS runtime is present.
    """
    global IS_LOCAL
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        IS_LOCAL = False
        return session
    except ImportError:
        # snowflake.snowpark not installed -> local dev environment
        IS_LOCAL = True
        return None
    except Exception as e:
        error_str = str(e)
        if "1403" in error_str or "No default Session" in error_str:
            # Snowpark is installed but no active SiS session exists.
            # This happens when running locally with snowpark installed,
            # or if the SiS app was not deployed via CREATE STREAMLIT.
            # Fall back to local mode so the user can connect manually.
            IS_LOCAL = True
            st.warning(
                "No active Snowflake session detected (snowpark is installed "
                "but no SiS runtime found). Falling back to manual connection.\n\n"
                "**If you are in Snowflake Streamlit:** ensure the app was "
                "deployed via `CREATE STREAMLIT` and the warehouse is active.\n\n"
                "**If you are running locally:** enter your credentials in the "
                "sidebar to connect."
            )
            return None
        # Any other unexpected error -- surface it
        st.error(
            f"Snowflake session error: {e}\n\n"
            "This may be caused by a package version conflict in requirements.txt. "
            "Ensure requirements.txt does not list streamlit, "
            "snowflake-connector-python, or snowflake-snowpark-python."
        )
        IS_LOCAL = False
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

# Default output configuration — used until config is loaded from Snowflake
_DEFAULT_DQ_CONFIG = {
    "CONFIG_DATABASE": "",
    "CONFIG_SCHEMA": "",
    "OUTPUT_DATABASE": "",
    "OUTPUT_SCHEMA": "",
    "GENERATED_CHECKS_TABLE": "DQ_GENERATED_CHECKS",
    "SELECTED_CHECKS_TABLE": "DQ_SELECTED_CHECKS",
    "APPROVED_CHECKS_TABLE": "DQ_APPROVED_CHECKS",
}
if "dq_config" not in st.session_state:
    st.session_state.dq_config = dict(_DEFAULT_DQ_CONFIG)

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


# ===========================================================================
# ALL HELPER FUNCTIONS — defined here so they are available to sidebar + UI
# ===========================================================================

# ---------------------------------------------------------------------------
# Helper — run SQL query and return DataFrame (works in both modes)
# ---------------------------------------------------------------------------
def run_query_df(query):
    """Execute a SQL query and return results as a pandas DataFrame."""
    if "_last_query_error" in st.session_state:
        del st.session_state["_last_query_error"]
    try:
        if session is not None:
            try:
                df = session.sql(query).to_pandas()
            except Exception:
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
            st.session_state[cache_key] = sorted(
                df.iloc[:, 0].dropna().astype(str).unique().tolist()
            )
        else:
            st.session_state[cache_key] = []
    return st.session_state[cache_key]


def snowflake_table_picker(prefix):
    """(Deprecated) Placeholder to maintain backwards compatibility."""
    return None


def call_llm(prompt, model_name):
    """Call Snowflake Cortex LLM — supports both Snowpark session and connector."""
    try:
        if session is not None:
            # Snowpark: use $$ dollar-quoting to avoid escaping issues with large prompts
            escaped_prompt = prompt.replace('$$', '$ $')
            query = f"""
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                '{model_name}',
                $${escaped_prompt}$$
            ) AS response
            """
            result = session.sql(query).collect()
            return result[0]["RESPONSE"] if result else None
        elif st.session_state.get("connection"):
            # Connector: use parameterised query to handle special characters safely
            query = """
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                %s,
                %s
            ) AS response
            """
            cur = st.session_state.connection.cursor()
            cur.execute(query, (model_name, prompt))
            result = cur.fetchone()
            cur.close()
            return result[0] if result else None
        else:
            st.error("Not connected to Snowflake.")
            return None
    except Exception as e:
        st.error(f"LLM Error: {str(e)}")
        return None


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


def parse_llm_json(raw):
    """Best-effort extraction of a JSON object from a raw LLM response.

    Handles truncated JSON from large LLM responses by attempting to
    repair unbalanced braces/brackets.
    """
    cleaned = raw.strip()
    if '```' in cleaned:
        parts = cleaned.split('```')
        if len(parts) >= 3:
            cleaned = parts[1].replace('json', '', 1)
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]
    elif start != -1:
        # Truncated response — no closing brace found; take from opening brace
        cleaned = cleaned[start:]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Attempt to repair truncated JSON by balancing braces/brackets
        repaired = _repair_truncated_json(cleaned)
        return json.loads(repaired)


def _repair_truncated_json(text):
    """Attempt to fix truncated JSON by closing open braces/brackets.

    Strips any trailing incomplete key-value pair, then appends the
    necessary closing characters.
    """
    # Remove trailing partial entries (e.g. a key without a value)
    import re
    text = re.sub(r',\s*"[^"]*"\s*:\s*$', '', text.rstrip())
    text = re.sub(r',\s*\{[^}]*$', '', text.rstrip())
    text = re.sub(r',\s*$', '', text.rstrip())

    # Count unbalanced braces/brackets (ignoring those inside strings)
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            open_braces += 1
        elif ch == '}':
            open_braces -= 1
        elif ch == '[':
            open_brackets += 1
        elif ch == ']':
            open_brackets -= 1

    # Append missing closers
    text += ']' * max(open_brackets, 0)
    text += '}' * max(open_braces, 0)
    return text


def save_to_snowflake(df, table_name):
    """Save a pandas DataFrame to a Snowflake table.

    Uses fully-qualified table names throughout to avoid unsupported
    USE DATABASE / USE SCHEMA commands in SiS environments.
    Column names are sanitised (spaces -> underscores, uppercased) to
    avoid quoting issues in Snowflake SQL.
    """
    try:
        # Sanitise column names: replace spaces with underscores, uppercase
        save_df = df.copy()
        save_df.columns = [
            c.strip().replace(' ', '_').replace('(', '').replace(')', '')
             .replace('/', '_').upper()
            for c in save_df.columns
        ]

        parts = table_name.split(".")
        if session is not None:
            # In SiS, USE statements are not allowed.  Use write_pandas
            # from the underlying connection which accepts FQ names directly.
            try:
                from snowflake.connector.pandas_tools import write_pandas
                raw_conn = session.connection   # underlying connector
                if len(parts) == 3:
                    write_pandas(
                        raw_conn,
                        save_df,
                        parts[2],
                        database=parts[0],
                        schema=parts[1],
                        auto_create_table=True,
                        overwrite=True,
                    )
                else:
                    write_pandas(
                        raw_conn,
                        save_df,
                        table_name,
                        auto_create_table=True,
                        overwrite=True,
                    )
            except AttributeError:
                # Fallback: session.connection not available — use Snowpark
                snowpark_df = session.create_dataframe(save_df)
                snowpark_df.write.mode("overwrite").save_as_table(table_name)
        elif st.session_state.get("connection"):
            from snowflake.connector.pandas_tools import write_pandas
            conn = st.session_state.connection
            if len(parts) == 3:
                write_pandas(
                    conn,
                    save_df,
                    parts[2],
                    database=parts[0],
                    schema=parts[1],
                    auto_create_table=True,
                    overwrite=True,
                )
            else:
                write_pandas(
                    conn,
                    save_df,
                    table_name,
                    auto_create_table=True,
                    overwrite=True,
                )
        else:
            st.error("Not connected to Snowflake.")
            return False
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False


def _config_table_fq() -> str:
    """Return the fully-qualified config table name from current settings."""
    cfg = st.session_state.dq_config
    db = (cfg.get("CONFIG_DATABASE") or "").strip()
    schema = (cfg.get("CONFIG_SCHEMA") or "").strip()
    if db and schema:
        return f"{db}.{schema}.DQ_APP_CONFIG"
    return "DQ_APP_CONFIG"


def load_dq_config():
    """Read DQ_APP_CONFIG from Snowflake and populate session state."""
    fq = _config_table_fq()
    try:
        df = run_query_df(f"SELECT * FROM {fq}")
        if df.empty:
            return False
        key_col = "config_key" if "config_key" in df.columns else df.columns[0]
        val_col = "config_value" if "config_value" in df.columns else df.columns[1]
        for _, row in df.iterrows():
            k = str(row[key_col]).strip()
            v = str(row[val_col]).strip() if row[val_col] is not None else ""
            if k in st.session_state.dq_config:
                st.session_state.dq_config[k] = v
        return True
    except Exception:
        return False


def save_dq_config():
    """Write current dq_config to the DQ_APP_CONFIG table in Snowflake."""
    fq = _config_table_fq()
    rows = []
    for k, v in st.session_state.dq_config.items():
        rows.append({"CONFIG_KEY": k, "CONFIG_VALUE": v})
    cfg_df = pd.DataFrame(rows)
    return save_to_snowflake(cfg_df, fq)


def get_output_table(config_key: str) -> str:
    """Build a fully-qualified output table name from config."""
    cfg = st.session_state.dq_config
    db = (cfg.get("OUTPUT_DATABASE") or "").strip()
    schema = (cfg.get("OUTPUT_SCHEMA") or "").strip()
    table = (cfg.get(config_key) or "").strip()
    if not table:
        table = _DEFAULT_DQ_CONFIG.get(config_key, config_key)
    if db and schema:
        return f"{db}.{schema}.{table}"
    return table


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
                        st.experimental_rerun()
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
                st.experimental_rerun()

    st.divider()

    st.header("🤖 Model Settings")
    model = st.selectbox(
        "Select Model",
        ["llama3.1-70b", "llama3.1-8b", "mistral-large2", "mixtral-8x7b"],
        index=0
    )
    st.session_state.model = model

    st.divider()

    # ---- Output Configuration (reads/writes DQ_APP_CONFIG) ----
    st.header("⚙️ Output Config")
    cfg = st.session_state.dq_config

    cfg["CONFIG_DATABASE"] = st.text_input(
        "Config DB",
        value=cfg.get("CONFIG_DATABASE") or st.session_state.get("sf_database", ""),
        key="cfg_config_db",
        help="Database where DQ_APP_CONFIG table lives"
    )
    cfg["CONFIG_SCHEMA"] = st.text_input(
        "Config Schema",
        value=cfg.get("CONFIG_SCHEMA") or st.session_state.get("sf_schema", ""),
        key="cfg_config_schema",
        help="Schema where DQ_APP_CONFIG table lives"
    )

    cfg_load_col, cfg_save_col = st.columns(2)
    with cfg_load_col:
        if st.button("📥 Load", key="cfg_load_btn", use_container_width=True,
                      help="Load config from DQ_APP_CONFIG table"):
            ok = load_dq_config()
            if ok:
                st.success("Config loaded!")
                st.experimental_rerun()
            else:
                st.warning("No config table found — using defaults. Click Save to create it.")
    with cfg_save_col:
        if st.button("💾 Save", key="cfg_save_btn", use_container_width=True,
                      help="Save config to DQ_APP_CONFIG table"):
            if save_dq_config():
                st.success("Config saved!")
            else:
                st.error("Save failed — check permissions.")

    with st.expander("📋 Output Table Settings", expanded=False):
        cfg["OUTPUT_DATABASE"] = st.text_input(
            "Output Database",
            value=cfg.get("OUTPUT_DATABASE") or st.session_state.get("sf_database", ""),
            key="cfg_output_db",
            help="Database for all output tables"
        )
        cfg["OUTPUT_SCHEMA"] = st.text_input(
            "Output Schema",
            value=cfg.get("OUTPUT_SCHEMA") or st.session_state.get("sf_schema", ""),
            key="cfg_output_schema",
            help="Schema for all output tables"
        )
        cfg["GENERATED_CHECKS_TABLE"] = st.text_input(
            "Generated Checks Table",
            value=cfg.get("GENERATED_CHECKS_TABLE", "DQ_GENERATED_CHECKS"),
            key="cfg_gen_table",
        )
        cfg["SELECTED_CHECKS_TABLE"] = st.text_input(
            "Selected Checks Table",
            value=cfg.get("SELECTED_CHECKS_TABLE", "DQ_SELECTED_CHECKS"),
            key="cfg_sel_table",
        )
        cfg["APPROVED_CHECKS_TABLE"] = st.text_input(
            "Approved Checks Table",
            value=cfg.get("APPROVED_CHECKS_TABLE", "DQ_APPROVED_CHECKS"),
            key="cfg_appr_table",
        )
        # Show resolved fully-qualified names
        st.caption("**Resolved table names:**")
        st.code(
            f"{get_output_table('GENERATED_CHECKS_TABLE')}\n"
            f"{get_output_table('SELECTED_CHECKS_TABLE')}\n"
            f"{get_output_table('APPROVED_CHECKS_TABLE')}",
            language="text"
        )

    st.divider()
    st.caption(f"ℹ️ Mode: {'Local Dev' if IS_LOCAL else 'Snowflake Streamlit'}")


# Expected column names for catalog and metadata DataFrames.
# Used to normalise columns from any source (table or stage file).
_CATALOG_COLUMNS = [
    'Critical Data Element', 'Domain', 'DQ Dimension', 'Check name',
    'Check Description', 'Short description of Critical Data Element',
    'SODACL Yaml Check Definition',
]
_METADATA_COLUMNS = [
    'Column name', 'Column description', 'Datatype',
    'Related Fields (Min/Max/Sample Values)',
]


def _normalize_df_columns(df: pd.DataFrame, expected: list) -> pd.DataFrame:
    """Map DataFrame columns to expected names via case-insensitive match.

    Handles variations like underscores vs spaces
    (e.g., CRITICAL_DATA_ELEMENT -> Critical Data Element) and common
    typos (e.g., DOMIAN -> Domain, CHCK -> Check).
    """
    col_map = {}
    expected_set = set(expected)
    # Primary lookup: normalized expected name -> canonical name
    lower_lookup = {e.lower().replace('_', ' ').strip(): e for e in expected}
    # Track which expected columns have already been matched
    matched_expected = set()

    for actual in df.columns:
        if actual in expected_set:
            matched_expected.add(actual)
            continue  # already matches exactly
        key = actual.lower().replace('_', ' ').strip()
        if key in lower_lookup:
            col_map[actual] = lower_lookup[key]
            matched_expected.add(lower_lookup[key])

    # Fuzzy fallback for remaining unmatched columns — handles typos
    # like DOMIAN->Domain, SODACL_YAML_CHCK_DEFINITION->SODACL Yaml Check Definition
    unmatched_expected = [e for e in expected if e not in matched_expected
                         and e not in col_map.values()]
    if unmatched_expected:
        unmatched_actual = [c for c in df.columns
                           if c not in col_map and c not in expected_set]
        for exp in unmatched_expected:
            exp_key = exp.lower().replace('_', ' ').strip()
            exp_words = set(exp_key.split())
            best_match = None
            best_score = 0
            for act in unmatched_actual:
                act_key = act.lower().replace('_', ' ').strip()
                act_words = set(act_key.split())
                # Calculate word-overlap score
                common = exp_words & act_words
                if common:
                    score = len(common) / max(len(exp_words), len(act_words))
                    if score > best_score:
                        best_score = score
                        best_match = act
                # Also check if one contains most of the other (typo-tolerant)
                elif len(exp_words) == 1 and len(act_words) == 1:
                    # Single-word columns: check edit-distance-like similarity
                    e_w = exp_key
                    a_w = act_key
                    # Check if one is a substring of the other or nearly so
                    shorter, longer = (e_w, a_w) if len(e_w) <= len(a_w) else (a_w, e_w)
                    if len(shorter) >= 3 and shorter in longer:
                        score = len(shorter) / len(longer)
                        if score > best_score:
                            best_score = score
                            best_match = act
                    # Also check character overlap for typos like domian/domain
                    elif len(shorter) >= 3:
                        common_chars = sum(1 for c in shorter if c in longer)
                        ratio = common_chars / max(len(shorter), len(longer))
                        if ratio >= 0.75 and ratio > best_score:
                            best_score = ratio
                            best_match = act
            if best_match and best_score >= 0.5:
                col_map[best_match] = exp
                unmatched_actual.remove(best_match)

    if col_map:
        df = df.rename(columns=col_map)
    return df


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


def list_stages_for_context(context: dict, cache_label: str) -> list:
    """Return formatted stage entries for the provided database/schema.

    Runs SHOW STAGES IN SCHEMA {db}.{schema} and returns a list of
    dicts with 'value' and 'label' keys, suitable for st.selectbox.
    Results are cached in session_state, keyed by cache_label + db + schema.
    """
    database = (context.get("database") or "").strip()
    schema = (context.get("schema") or "").strip()
    if not database or not schema:
        return []

    cache_key = f"_stages_{cache_label}_{database}_{schema}"
    debug_key = f"_stages_debug_{cache_label}"
    if cache_key not in st.session_state:
        sql_query = f'SHOW STAGES IN SCHEMA {database}.{schema}'
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
                # Build fully-qualified stage name for use in LIST @stage
                fq_stage = f"{database}.{schema}.{name}"
                comment = row.get("comment")
                created = row.get("created_on") or row.get("created")
                owner = row.get("owner")
                label_parts = [name]
                if comment:
                    label_parts.append(str(comment))
                elif created:
                    try:
                        ts = pd.to_datetime(created)
                        label_parts.append(ts.strftime("%Y-%m-%d"))
                    except Exception:
                        label_parts.append(str(created))
                if owner:
                    label_parts.append(f"owner: {owner}")
                entries.append({
                    "value": fq_stage,
                    "label": " \u2022 ".join(label_parts)
                })
            st.session_state[debug_key]["status"] = "success"
            st.session_state[debug_key]["stage_count"] = len(entries)
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
# Step 1 — Load DQ Checks Catalog
# ===========================================================================
st.header("📘 Step 1: Load DQ Checks Catalog")

catalog_context = st.session_state.catalog_context

# Context inputs (database + schema)
st.caption("Catalog context — set the database and schema where your catalog lives.")
cat_db_col, cat_schema_col = st.columns(2)
catalog_context["database"] = cat_db_col.text_input(
    "Catalog Database",
    value=catalog_context.get("database", ""),
    key="catalog_db_input"
)
catalog_context["schema"] = cat_schema_col.text_input(
    "Catalog Schema",
    value=catalog_context.get("schema", ""),
    key="catalog_schema_input"
)

# Source toggle
catalog_source = st.radio(
    "Choose catalog source",
    ["Snowflake Table", "Snowflake Stage (Excel/CSV)"],
    horizontal=True,
    key="catalog_source",
    help="Load the DQ checks catalog from a Snowflake table or from an Excel/CSV file on a stage"
)

if catalog_source == "Snowflake Table":
    # ---- TABLE PATH ----
    catalog_table_entries = list_tables_for_context(catalog_context, "catalog")
    selected_catalog_table = ""
    cat_select_col, cat_refresh_col = st.columns([7, 1])
    with cat_select_col:
        if catalog_table_entries:
            option_values = [entry["value"] for entry in catalog_table_entries]
            label_map = {entry["value"]: entry["label"] for entry in catalog_table_entries}
            selected_catalog_table = st.selectbox(
                "Choose catalog table",
                [""] + option_values,
                key="catalog_table_select",
                format_func=lambda val: label_map.get(val, "Select table..." if val == "" else val)
            )
        else:
            st.info("No tables found for the provided catalog context.")
    with cat_refresh_col:
        st.write("\u200b")
        if st.button("🔄", key="catalog_tables_refresh", help="Refresh table list"):
            keys_to_clear = [k for k in list(st.session_state.keys())
                             if k.startswith("_tables_catalog_") or k.startswith("_tables_debug_catalog")]
            for k in keys_to_clear:
                del st.session_state[k]
            st.experimental_rerun()

    # Debug expander for table listing failures
    debug_info = st.session_state.get("_tables_debug_catalog")
    if debug_info and debug_info.get("status") in ["empty", "error"]:
        is_error = debug_info.get("status") == "error"
        with st.expander(f"🔍 Debug: {('Query failed' if is_error else 'no tables found')}", expanded=True):
            st.code(debug_info.get('query', ''), language='sql')
            st.write(f"**Database:** {debug_info.get('database')}")
            st.write(f"**Schema:** {debug_info.get('schema')}")
            if is_error:
                st.error(f"**Error:** {debug_info.get('error', 'Unknown error')}")
            else:
                st.write(f"**Rows returned:** {debug_info.get('row_count', 0)}")
            st.caption("Try running the SQL above in a Snowflake worksheet to verify.")

    # Manual table name fallback
    catalog_table = st.text_input(
        "Or enter table name",
        value=st.session_state.get("catalog_table_name", ""),
        placeholder="DQ_CHECKS_CATALOG",
        key="catalog_table_name_input"
    )

    if st.button("📥 Load Catalog", key="load_catalog_table_btn", type="primary"):
        try:
            table_choice = catalog_table.strip() or selected_catalog_table.strip()
            if not table_choice:
                raise ValueError("Select or enter a catalog table name.")
            table_name = resolve_table_name(table_choice, catalog_context)
            df = _normalize_df_columns(load_table(table_name), _CATALOG_COLUMNS)
            st.session_state.catalog_df = df
            st.session_state.catalog_table_name = table_choice
            st.success(f"✅ Catalog loaded: **{len(df)}** checks")
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Failed to load table: {e}")

else:
    # ---- STAGE PATH (dynamic stage dropdown) ----
    stage_entries = list_stages_for_context(catalog_context, "catalog")
    cat_stage_select_col, cat_stage_refresh_col = st.columns([7, 1])
    with cat_stage_select_col:
        if stage_entries:
            stage_option_values = [entry["value"] for entry in stage_entries]
            stage_label_map = {entry["value"]: entry["label"] for entry in stage_entries}
            selected_stage = st.selectbox(
                "Choose stage",
                [""] + stage_option_values,
                key="catalog_stage_select",
                format_func=lambda val: stage_label_map.get(val, "Select stage..." if val == "" else val)
            )
            catalog_context["stage"] = selected_stage
        else:
            st.info("No stages found. Enter a stage name manually below.")
            selected_stage = ""
    with cat_stage_refresh_col:
        st.write("\u200b")
        if st.button("🔄", key="catalog_stages_refresh", help="Refresh stage list"):
            keys_to_clear = [k for k in list(st.session_state.keys())
                             if k.startswith("_stages_catalog_") or k.startswith("_stages_debug_catalog")]
            for k in keys_to_clear:
                del st.session_state[k]
            st.experimental_rerun()

    # Debug expander for stage listing failures
    stages_debug = st.session_state.get("_stages_debug_catalog")
    if stages_debug and stages_debug.get("status") in ["empty", "error"]:
        is_error = stages_debug.get("status") == "error"
        with st.expander(f"🔍 Debug: {('Query failed' if is_error else 'no stages found')}", expanded=True):
            st.code(stages_debug.get('query', ''), language='sql')
            st.write(f"**Database:** {stages_debug.get('database')}")
            st.write(f"**Schema:** {stages_debug.get('schema')}")
            if is_error:
                st.error(f"**Error:** {stages_debug.get('error', 'Unknown error')}")
            else:
                st.write(f"**Rows returned:** {stages_debug.get('row_count', 0)}")
            st.caption("Try running the SQL above in a Snowflake worksheet to verify.")

    # Manual stage override + prefix
    cat_manual_stage_col, cat_prefix_col = st.columns([2, 1])
    manual_stage = cat_manual_stage_col.text_input(
        "Or enter stage name (DB.SCHEMA.STAGE)",
        value="" if selected_stage else catalog_context.get("stage", ""),
        key="catalog_stage_manual_input",
        placeholder="DEV_GDP_UTIL_DB.EXT_VOL.CATALOG_STAGE"
    )
    catalog_context["stage_prefix"] = cat_prefix_col.text_input(
        "Stage Folder (optional)",
        value=catalog_context.get("stage_prefix", ""),
        key="catalog_stage_prefix_input",
        placeholder="folder/subfolder"
    )

    # Determine effective stage name: dropdown takes priority over manual
    effective_stage = selected_stage or manual_stage.strip()
    if effective_stage:
        catalog_context["stage"] = effective_stage

    if not effective_stage:
        st.warning("Select a stage from the dropdown or enter a fully-qualified stage name.")
    else:
        try:
            entries = list_stage_entries(
                effective_stage,
                catalog_context.get("stage_prefix", "")
            )
        except Exception as exc:
            entries = []
            st.error(f"Error listing stage: {exc}")

        # Debug for empty file listing
        stage_debug = st.session_state.get("_stage_list_debug")
        if not entries and stage_debug:
            with st.expander("🔍 Debug: stage listing returned no files", expanded=True):
                st.code(stage_debug.get('query', ''), language='sql')
                st.write(f"**Stage reference:** {stage_debug.get('reference')}")
                st.write(f"**Rows returned:** {stage_debug.get('row_count', 0)}")
                if stage_debug.get('row_count', 0) == 0:
                    st.info("Query executed successfully but returned 0 files.")
                st.caption("Try running the SQL above in a Snowflake worksheet.")

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
                        df = _normalize_df_columns(df, _CATALOG_COLUMNS)
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
metadata_source = st.radio(
    "Choose metadata source",
    ["Snowflake Data Table", "Snowflake Metadata Table", "Snowflake Stage (Excel/CSV)"],
    horizontal=True,
    key="metadata_source",
    help=("**Data Table**: select a table and auto-generate column metadata from INFORMATION_SCHEMA. "
          "**Metadata Table**: load a pre-existing metadata table with Column name/description columns. "
          "**Stage**: load metadata from an Excel/CSV file on a stage.")
)

metadata_df = None
if metadata_source == "Snowflake Data Table":
    # ---- DATA TABLE PATH: auto-generate metadata from INFORMATION_SCHEMA ----
    data_table_entries = list_tables_for_context(metadata_context, "data_table")
    selected_data_table = ""
    dt_select_col, dt_refresh_col = st.columns([7, 1])
    with dt_select_col:
        if data_table_entries:
            option_values = [entry["value"] for entry in data_table_entries]
            label_map = {entry["value"]: entry["label"] for entry in data_table_entries}
            selected_data_table = st.selectbox(
                "Choose data table",
                [""] + option_values,
                key="data_table_select",
                format_func=lambda val: label_map.get(val, "Select table..." if val == "" else val)
            )
        else:
            st.info("No tables found for the provided context.")
    with dt_refresh_col:
        st.write("\u200b")
        if st.button("🔄", key="data_tables_refresh", help="Refresh table list"):
            keys_to_clear = [k for k in list(st.session_state.keys())
                             if k.startswith("_tables_data_table_") or k.startswith("_tables_debug_data_table")]
            for k in keys_to_clear:
                del st.session_state[k]
            st.experimental_rerun()

    data_table_manual = st.text_input(
        "Or enter table name",
        value=st.session_state.get("data_table_name", ""),
        placeholder="MY_DATA_TABLE",
        key="data_table_name_input"
    )

    if st.button("📥 Generate Metadata from Table", key="load_data_table_btn", type="primary"):
        try:
            table_choice = data_table_manual.strip() or selected_data_table.strip()
            if not table_choice:
                raise ValueError("Select or enter a data table name.")
            fq_table = resolve_table_name(table_choice, metadata_context)
            parts = fq_table.split(".")
            if len(parts) != 3:
                raise ValueError(f"Need fully-qualified name (DB.SCHEMA.TABLE), got: {fq_table}")
            info_query = f"""
                SELECT
                    COLUMN_NAME,
                    DATA_TYPE,
                    COMMENT,
                    IS_NULLABLE,
                    CHARACTER_MAXIMUM_LENGTH,
                    NUMERIC_PRECISION,
                    NUMERIC_SCALE
                FROM {parts[0]}.INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = '{parts[1]}'
                  AND TABLE_NAME   = '{parts[2]}'
                ORDER BY ORDINAL_POSITION
            """
            info_df = run_query_df(info_query)
            if info_df.empty:
                raise ValueError(f"No columns found for {fq_table}. Check the table exists and you have access.")

            # Build metadata DataFrame in the expected format
            rows = []
            for _, r in info_df.iterrows():
                col_name = r.get("column_name", "")
                dtype = r.get("data_type", "")
                comment = r.get("comment", "") or ""
                nullable = r.get("is_nullable", "")
                char_len = r.get("character_maximum_length", "")
                num_prec = r.get("numeric_precision", "")
                num_scale = r.get("numeric_scale", "")
                # Build a type detail string
                type_detail = dtype
                if char_len and str(char_len) not in ("", "None"):
                    type_detail += f"({char_len})"
                elif num_prec and str(num_prec) not in ("", "None"):
                    scale_part = f",{num_scale}" if num_scale and str(num_scale) not in ("", "None") else ""
                    type_detail += f"({num_prec}{scale_part})"
                related = f"Nullable: {nullable}"
                rows.append({
                    'Column name': col_name,
                    'Column description': comment if comment else f"Column {col_name} ({type_detail})",
                    'Datatype': type_detail,
                    'Related Fields (Min/Max/Sample Values)': related,
                })
            metadata_df = pd.DataFrame(rows)
            st.session_state.collibra_metadata = metadata_df
            st.session_state.data_table_name = table_choice
            st.session_state.data_table_fq = fq_table
            st.success(f"✅ Metadata auto-generated: **{len(metadata_df)}** columns from `{fq_table}`")
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Failed to generate metadata: {e}")

elif metadata_source == "Snowflake Metadata Table":
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
            st.experimental_rerun()

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
            metadata_df = _normalize_df_columns(load_table(table_name), _METADATA_COLUMNS)
            st.session_state.collibra_metadata = metadata_df
            st.session_state.metadata_table_name = table_choice
            st.success(f"✅ Metadata loaded: **{len(metadata_df)}** columns")
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Failed to load table: {e}")
elif metadata_source == "Snowflake Stage (Excel/CSV)":
    # ---- STAGE PATH (dynamic stage dropdown) ----
    stage_entries = list_stages_for_context(metadata_context, "metadata")
    meta_stage_select_col, meta_stage_refresh_col = st.columns([7, 1])
    with meta_stage_select_col:
        if stage_entries:
            stage_option_values = [entry["value"] for entry in stage_entries]
            stage_label_map = {entry["value"]: entry["label"] for entry in stage_entries}
            selected_meta_stage = st.selectbox(
                "Choose stage",
                [""] + stage_option_values,
                key="metadata_stage_select",
                format_func=lambda val: stage_label_map.get(val, "Select stage..." if val == "" else val)
            )
            metadata_context["stage"] = selected_meta_stage
        else:
            st.info("No stages found. Enter a stage name manually below.")
            selected_meta_stage = ""
    with meta_stage_refresh_col:
        st.write("\u200b")
        if st.button("🔄", key="metadata_stages_refresh", help="Refresh stage list"):
            keys_to_clear = [k for k in list(st.session_state.keys())
                             if k.startswith("_stages_metadata_") or k.startswith("_stages_debug_metadata")]
            for k in keys_to_clear:
                del st.session_state[k]
            st.experimental_rerun()

    # Debug expander for stage listing failures
    stages_debug = st.session_state.get("_stages_debug_metadata")
    if stages_debug and stages_debug.get("status") in ["empty", "error"]:
        is_error = stages_debug.get("status") == "error"
        with st.expander(f"🔍 Debug: {('Query failed' if is_error else 'no stages found')}", expanded=True):
            st.code(stages_debug.get('query', ''), language='sql')
            st.write(f"**Database:** {stages_debug.get('database')}")
            st.write(f"**Schema:** {stages_debug.get('schema')}")
            if is_error:
                st.error(f"**Error:** {stages_debug.get('error', 'Unknown error')}")
            else:
                st.write(f"**Rows returned:** {stages_debug.get('row_count', 0)}")
            st.caption("Try running the SQL above in a Snowflake worksheet to verify.")

    # Manual stage override + prefix
    meta_manual_stage_col, meta_prefix_col = st.columns([2, 1])
    manual_meta_stage = meta_manual_stage_col.text_input(
        "Or enter stage name (DB.SCHEMA.STAGE)",
        value="" if selected_meta_stage else metadata_context.get("stage", ""),
        key="metadata_stage_manual_input",
        placeholder="DEV_GDP_UTIL_DB.EXT_VOL.METADATA_STAGE"
    )
    metadata_context["stage_prefix"] = meta_prefix_col.text_input(
        "Stage Folder (optional)",
        value=metadata_context.get("stage_prefix", ""),
        key="metadata_stage_prefix_input",
        placeholder="folder/subfolder"
    )

    # Determine effective stage name: dropdown takes priority over manual
    effective_meta_stage = selected_meta_stage or manual_meta_stage.strip()
    if effective_meta_stage:
        metadata_context["stage"] = effective_meta_stage

    if not effective_meta_stage:
        st.warning("Select a stage from the dropdown or enter a fully-qualified stage name.")
    else:
        try:
            entries = list_stage_entries(
                effective_meta_stage,
                metadata_context.get("stage_prefix", "")
            )
        except Exception as exc:
            entries = []
            st.error(f"Error listing stage: {exc}")

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
                        df = _normalize_df_columns(df, _METADATA_COLUMNS)
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

    metadata_df = _normalize_df_columns(st.session_state.collibra_metadata, _METADATA_COLUMNS)
    catalog_df = _normalize_df_columns(st.session_state.catalog_df, _CATALOG_COLUMNS)

    # Validate required columns exist
    _missing_cat = [c for c in ['Critical Data Element', 'Domain'] if c not in catalog_df.columns]
    _missing_meta = [c for c in ['Column name'] if c not in metadata_df.columns]
    if _missing_cat or _missing_meta:
        if _missing_cat:
            st.error(f"Catalog is missing columns: {_missing_cat}. "
                     f"Available columns: {list(catalog_df.columns)}")
        if _missing_meta:
            st.error(f"Metadata is missing columns: {_missing_meta}. "
                     f"Available columns: {list(metadata_df.columns)}")
        st.stop()

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

            # --- Batch metadata into chunks to avoid LLM token limits ---
            MATCH_BATCH_SIZE = 50
            all_match_results = []
            total_batches = (len(metadata_info) + MATCH_BATCH_SIZE - 1) // MATCH_BATCH_SIZE
            progress_bar = st.progress(0, text="Matching columns...")

            for batch_idx in range(0, len(metadata_info), MATCH_BATCH_SIZE):
                batch = metadata_info[batch_idx:batch_idx + MATCH_BATCH_SIZE]
                batch_num = batch_idx // MATCH_BATCH_SIZE + 1
                progress_bar.progress(
                    batch_num / total_batches,
                    text=f"Matching batch {batch_num}/{total_batches} ({len(batch)} columns)..."
                )

                matching_prompt = f"""You are a data quality expert. Match Collibra metadata columns to catalog Critical Data Elements (CDEs).

Catalog Domains: {catalog_domains}
Catalog CDEs: {catalog_cdes}

Metadata Columns:
{json.dumps(batch, indent=2)}

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
                        batch_matches = parse_llm_json(response)
                        all_match_results.extend(batch_matches.get('matches', []))
                    except Exception as e:
                        st.warning(f"Batch {batch_num}/{total_batches} parse error: {e}. Skipping {len(batch)} columns.")
                        with st.expander(f"Raw response (batch {batch_num})"):
                            st.code(response)

            progress_bar.empty()
            matches = {"matches": all_match_results}

            if all_match_results:
                try:
                    st.success(f"✅ Matching complete! Matched {len(all_match_results)} columns.")
                    with st.expander("🔍 View Matches"):
                        st.json(matches)

                    # Build output rows
                    output_rows = []
                    for match in matches.get('matches', []):
                        column_name = match.get('column_name', '')
                        matched_cde = match.get('matched_cde', 'Unknown')

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

                        # Generate synthetic test values in batches
                        st.info("🧪 Generating synthetic test values...")
                        synthetic_values = {}
                        unique_cdes = output_df['Critical Data Element'].unique()
                        SYNTH_BATCH_SIZE = 10
                        synth_progress = st.progress(0, text="Generating synthetic values...")

                        for synth_batch_idx in range(0, len(unique_cdes), SYNTH_BATCH_SIZE):
                            synth_batch = unique_cdes[synth_batch_idx:synth_batch_idx + SYNTH_BATCH_SIZE]
                            synth_batch_num = synth_batch_idx // SYNTH_BATCH_SIZE + 1
                            synth_total_batches = (len(unique_cdes) + SYNTH_BATCH_SIZE - 1) // SYNTH_BATCH_SIZE
                            synth_progress.progress(
                                synth_batch_num / synth_total_batches,
                                text=f"Generating synthetic values (batch {synth_batch_num}/{synth_total_batches})..."
                            )

                            # Build batch prompt with multiple columns
                            columns_info = []
                            for cde in synth_batch:
                                cde_checks = output_df[output_df['Critical Data Element'] == cde]
                                first_check = cde_checks.iloc[0]
                                original_metadata = None
                                for _, meta_row in metadata_df.iterrows():
                                    if meta_row.get('Column name', '') == cde:
                                        original_metadata = meta_row
                                        break
                                columns_info.append({
                                    'column': cde,
                                    'description': first_check['Short description of Critical Data Element'],
                                    'domain': first_check['Domain'],
                                    'datatype': original_metadata.get('Datatype', '') if original_metadata is not None else '',
                                    'related_fields': str(original_metadata.get('Related Fields (Min/Max/Sample Values)', '')) if original_metadata is not None else ''
                                })

                            synthetic_prompt = f"""Generate synthetic test values for data quality testing for these columns.

Columns:
{json.dumps(columns_info, indent=2)}

For EACH column, generate 10 synthetic values: 6 VALID (pass checks) and 4 INVALID (fail - edge cases, wrong format, nulls).

Respond ONLY with valid JSON:
{{
  "columns": {{
    "COLUMN_NAME": {{
      "values": [
        {{"value": "32.95", "expected": "PASS", "reason": "Valid value"}},
        {{"value": "-91.5", "expected": "FAIL", "reason": "Out of range"}}
      ]
    }}
  }}
}}

Output only JSON, no markdown."""

                            synth_response = call_llm(synthetic_prompt, st.session_state.model)

                            if synth_response:
                                try:
                                    synth_data = parse_llm_json(synth_response)
                                    columns_data = synth_data.get('columns', {})
                                    for cde in synth_batch:
                                        if cde in columns_data:
                                            synthetic_values[cde] = columns_data[cde].get('values', [])
                                        else:
                                            synthetic_values[cde] = []
                                except Exception:
                                    for cde in synth_batch:
                                        synthetic_values[cde] = []

                        synth_progress.empty()
                        st.session_state.synthetic_values = synthetic_values
                        st.success(f"✅ Generated synthetic test values for {len(synthetic_values)} columns!")
                        st.experimental_rerun()
                    else:
                        st.warning("No matches found")

                except Exception as e:
                    st.error(f"Failed to parse matching results: {str(e)}")
            else:
                st.warning("No matches found. AI could not match any columns.")

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
        _gen_table_fq = get_output_table("GENERATED_CHECKS_TABLE")
        st.caption(f"Target: `{_gen_table_fq}`")
        if st.button("💾 Save to Snowflake", key="save_checks_btn", use_container_width=True):
            if save_to_snowflake(generated_df, _gen_table_fq):
                st.success(f"✅ Saved to `{_gen_table_fq}`")
    with col4:
        if st.button("📈 AI Coverage Assessment", key="ai_coverage_btn", use_container_width=True):
            with st.spinner("Cortex AI is assessing DQ coverage..."):
                # Summarise checks to keep the prompt compact for large tables
                checks_summary = []
                if len(generated_df) > 200:
                    # Aggregate: count checks per CDE per dimension
                    agg = generated_df.groupby(
                        ['Critical Data Element', 'Domain', 'DQ Dimension']
                    ).size().reset_index(name='check_count')
                    for _, row in agg.iterrows():
                        checks_summary.append({
                            "cde": row["Critical Data Element"],
                            "domain": row["Domain"],
                            "dimension": row["DQ Dimension"],
                            "check_count": int(row["check_count"])
                        })
                else:
                    for _, row in generated_df.iterrows():
                        checks_summary.append({
                            "cde": row["Critical Data Element"],
                            "domain": row["Domain"],
                            "dimension": row["DQ Dimension"],
                            "check": row["Check name"]
                        })

                coverage_prompt = f"""You are a data quality expert. Assess the completeness and coverage of these generated DQ checks.

Generated Checks ({len(generated_df)} total across {generated_df['Critical Data Element'].nunique()} CDEs):
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
            _nav_to("pages/dq_marketplace.py")
