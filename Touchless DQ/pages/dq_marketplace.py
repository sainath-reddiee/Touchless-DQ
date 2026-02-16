import streamlit as st
import pandas as pd

# NOTE: No st.set_page_config() here — only allowed in main app file

# Hide navigation
st.markdown("""
<style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("✅ CDE Marketplace - Review & Customize Checks")

# ---------------------------------------------------------------------------
# Auto-detect SiS vs Local
# ---------------------------------------------------------------------------
IS_LOCAL = False
session = None

try:
    from snowflake.snowpark.context import get_active_session
    session = get_active_session()
except Exception:
    IS_LOCAL = True

# Initialize session state
if "selected_checks" not in st.session_state:
    st.session_state.selected_checks = {}

# Check for data
if not st.session_state.get('connected', False):
    st.error("⚠️ Not connected. Please go back.")
    st.stop()

if st.session_state.get('df') is None:
    st.error("⚠️ No checks found. Please go back and generate checks first.")
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
        # Save to Snowflake table
        if st.button("💾 Save to Snowflake Table", use_container_width=True):
            if save_to_snowflake(selected_df, "DQ_SELECTED_CHECKS"):
                st.success("✅ Saved to `DQ_SELECTED_CHECKS`")

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
        st.page_link("pages/dq_review_page.py", label="➡️ Review One-by-One", icon="📊", use_container_width=True)

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
