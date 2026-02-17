import streamlit as st
import pandas as pd
import json
import streamlit.components.v1 as components

# NOTE: No st.set_page_config() here — only allowed in main app file

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


st.title("✅ CDE Marketplace - Review & Customize Checks")
render_top_nav("dq_marketplace")

# ---------------------------------------------------------------------------
# Auto-detect SiS vs Local
# ---------------------------------------------------------------------------
IS_LOCAL = False
session = None

try:
    from snowflake.snowpark.context import get_active_session
    session = get_active_session()
except ImportError:
    IS_LOCAL = True
except Exception as e:
    if "1403" in str(e) or "No default Session" in str(e):
        IS_LOCAL = True
    else:
        st.error(f"Snowflake session error: {e}")

def get_output_table(config_key: str) -> str:
    """Build a fully-qualified output table name from session config."""
    cfg = st.session_state.get("dq_config", {})
    db = (cfg.get("OUTPUT_DATABASE") or "").strip()
    schema = (cfg.get("OUTPUT_SCHEMA") or "").strip()
    table = (cfg.get(config_key) or "").strip()
    if not table:
        defaults = {
            "GENERATED_CHECKS_TABLE": "DQ_GENERATED_CHECKS",
            "SELECTED_CHECKS_TABLE": "DQ_SELECTED_CHECKS",
            "APPROVED_CHECKS_TABLE": "DQ_APPROVED_CHECKS",
        }
        table = defaults.get(config_key, config_key)
    if db and schema:
        return f"{db}.{schema}.{table}"
    return table

# Initialize session state
if "selected_checks" not in st.session_state:
    st.session_state.selected_checks = {}

# Check for data
if not st.session_state.get('connected', False):
    st.error("⚠️ Not connected. Please go back.")
    if st.button("⬅️ Back to Upload Page"):
        _nav_to("collibra_upload.py")
    st.stop()

if st.session_state.get('df') is None:
    st.error("⚠️ No checks found. Please go back and generate checks first.")
    if st.button("⬅️ Back to Upload Page"):
        _nav_to("collibra_upload.py")
    st.stop()

# Sidebar
with st.sidebar:
    st.success("✅ Connected to Snowflake")
    st.caption(f"Model: {st.session_state.get('model', 'llama3.1-70b')}")
    st.divider()

    if st.button("🔌 Reset Session"):
        st.session_state.clear()
        st.rerun()


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


# Get checks dataframe
df = st.session_state.df

# Display data
st.header("📊 Generated Checks from Catalog")
st.caption(f"Total: {len(df)} checks across {df['Critical Data Element'].nunique()} CDEs")

# Summary metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Checks", len(df))
with col2:
    st.metric("CDEs", df['Critical Data Element'].nunique())
with col3:
    st.metric("Domains", df['Domain'].nunique())
with col4:
    dimensions = df['DQ Dimension'].nunique()
    st.metric("DQ Dimensions", dimensions)

# ---------------------------------------------------------------------------
# Helper — call Cortex LLM (dual-mode)
# ---------------------------------------------------------------------------
def call_cortex(prompt, model_name):
    """Call Snowflake Cortex — supports both Snowpark session and connector."""
    try:
        escaped_prompt = prompt.replace("'", "''")
        query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model_name}',
            '{escaped_prompt}'
        ) AS response
        """
        if session is not None:
            result = session.sql(query).collect()
            return result[0]["RESPONSE"] if result else None
        elif st.session_state.get("connection"):
            cur = st.session_state.connection.cursor()
            cur.execute(query)
            result = cur.fetchone()
            cur.close()
            return result[0] if result else None
        else:
            st.error("Not connected to Snowflake.")
            return None
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return None

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

st.divider()

# ---------------------------------------------------------------------------
# AI Quality Summary
# ---------------------------------------------------------------------------
if st.button("🤖 AI Quality Summary", key="ai_quality_summary_btn", use_container_width=True):
    with st.spinner("Cortex AI is analyzing your checks..."):
        checks_info = []
        for _, row in df.iterrows():
            checks_info.append({
                "cde": row["Critical Data Element"],
                "domain": row["Domain"],
                "dimension": row["DQ Dimension"],
                "check_name": row["Check name"],
                "description": row["Check Description"][:100]
            })

        summary_prompt = f"""You are a data quality expert. Provide a quality summary of these DQ checks.

Checks:
{json.dumps(checks_info, indent=2)}

Analyze and provide:
1. Overall quality assessment of the check suite
2. Strengths — what's well-covered
3. Weaknesses — what's missing or could be better
4. Priority actions — top 3 things to improve
5. Quality score (0-100)

Respond ONLY with valid JSON:
{{
    "quality_score": 80,
    "assessment": "Brief overall assessment...",
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "priority_actions": ["action1", "action2", "action3"]
}}

Output only JSON, no markdown."""

        response = call_cortex(summary_prompt, st.session_state.get('model', 'llama3.1-70b'))
        if response:
            try:
                summary = parse_llm_json(response)
                st.session_state.ai_quality_summary_data = summary
            except Exception as e:
                st.error(f"Failed to parse summary: {e}")
                with st.expander("Raw response"):
                    st.code(response)

if st.session_state.get("ai_quality_summary_data"):
    summary = st.session_state.ai_quality_summary_data
    with st.expander("🤖 AI Quality Summary", expanded=True):
        qs_col1, qs_col2 = st.columns([1, 3])
        with qs_col1:
            st.metric("Quality Score", f"{summary.get('quality_score', 0)}/100")
        with qs_col2:
            st.info(summary.get("assessment", ""))

        qs_left, qs_right = st.columns(2)
        with qs_left:
            st.markdown("**Strengths:**")
            for s in summary.get("strengths", []):
                st.markdown(f"✅ {s}")
        with qs_right:
            st.markdown("**Weaknesses:**")
            for w in summary.get("weaknesses", []):
                st.markdown(f"⚠️ {w}")

        st.markdown("**Priority Actions:**")
        for i, action in enumerate(summary.get("priority_actions", []), 1):
            st.markdown(f"**{i}.** {action}")

st.divider()

# Initialize selection state
if not st.session_state.selected_checks:
    for idx, row in df.iterrows():
        check_key = f"{row['Critical Data Element']}_{row['DQ Dimension']}_{idx}"
        st.session_state.selected_checks[check_key] = {
            'selected': True,
            'row': row.to_dict()
        }

# Group by CDE
cdes = df['Critical Data Element'].unique()

st.header("🛒 Select and Customize Checks")

# Three column layout
col_left, col_middle, col_right = st.columns([2, 2, 3])

with col_left:
    st.subheader("📋 Data Elements")

    for cde in cdes:
        cde_checks = df[df['Critical Data Element'] == cde]
        domain = cde_checks.iloc[0]['Domain']

        with st.expander(f"📦 {cde} ({domain})", expanded=True):
            st.caption(f"{len(cde_checks)} checks")

            for idx, row in cde_checks.iterrows():
                check_key = f"{row['Critical Data Element']}_{row['DQ Dimension']}_{idx}"

                is_selected = st.checkbox(
                    f"{row['DQ Dimension']} - {row['Check name']}",
                    value=st.session_state.selected_checks[check_key]['selected'],
                    key=f"select_{check_key}"
                )
                st.session_state.selected_checks[check_key]['selected'] = is_selected

                if is_selected:
                    st.caption(f"✓ {row['Check Description'][:80]}...")

with col_middle:
    st.subheader("✏️ Check Details")

    selected_count = sum(1 for v in st.session_state.selected_checks.values() if v['selected'])
    st.caption(f"✓ {selected_count} checks selected")

    for check_key, check_data in st.session_state.selected_checks.items():
        if check_data['selected']:
            row = check_data['row']
            st.markdown(f"**{row['Critical Data Element']} - {row['DQ Dimension']}**")

            new_desc = st.text_area(
                "Description",
                value=row['Check Description'],
                height=80,
                key=f"desc_{check_key}",
                label_visibility="collapsed"
            )
            st.session_state.selected_checks[check_key]['row']['Check Description'] = new_desc
            st.divider()

with col_right:
    st.subheader("💻 SODA CL Code")

    for check_key, check_data in st.session_state.selected_checks.items():
        if check_data['selected']:
            row = check_data['row']
            st.markdown(f"**{row['Check name']}**")

            new_soda = st.text_area(
                "SODA",
                value=row['SODACL Yaml Check Definition'],
                height=150,
                key=f"soda_{check_key}",
                label_visibility="collapsed"
            )
            st.session_state.selected_checks[check_key]['row']['SODACL Yaml Check Definition'] = new_soda
            st.divider()

st.divider()

# Action buttons
st.header("📥 Export")

col_btn1, col_btn2, col_btn3 = st.columns(3)

# Compile selected checks
selected_rows = []
for check_data in st.session_state.selected_checks.values():
    if check_data['selected']:
        selected_rows.append(check_data['row'])

if selected_rows:
    selected_df = pd.DataFrame(selected_rows)

    # Generate YAML
    yaml_lines = []
    for _, row in selected_df.iterrows():
        yaml_lines.append(row['SODACL Yaml Check Definition'])
        yaml_lines.append("")
    final_yaml = "\n".join(yaml_lines)

    with col_btn1:
        _sel_table_fq = get_output_table("SELECTED_CHECKS_TABLE")
        st.caption(f"Target: `{_sel_table_fq}`")
        if st.button("💾 Save to Snowflake Table", use_container_width=True):
            if save_to_snowflake(selected_df, _sel_table_fq):
                st.success(f"✅ Saved to `{_sel_table_fq}`")

    with col_btn2:
        # Download buttons for local, YAML preview for SiS
        if IS_LOCAL:
            csv_data = selected_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv_data,
                file_name="selected_checks.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            if st.button("📄 View YAML", use_container_width=True, type="primary"):
                st.session_state['show_yaml_preview'] = True

    with col_btn3:
        if st.button("➡️ Next: DQ Review", use_container_width=True, type="primary"):
            _nav_to("pages/dq_review_page.py")

    # YAML preview / download
    if IS_LOCAL:
        st.download_button(
            "📥 Download YAML",
            data=final_yaml,
            file_name="dq_checks.yml",
            mime="text/yaml",
            use_container_width=True,
            type="primary"
        )
    elif st.session_state.get('show_yaml_preview', False):
        st.subheader("📄 SODACL YAML Preview")
        st.code(final_yaml, language="yaml")
        if st.button("Close Preview"):
            st.session_state['show_yaml_preview'] = False
            st.rerun()
else:
    st.warning("No checks selected")
