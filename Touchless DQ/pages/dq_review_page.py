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


st.title("📊 Review & Approve Data Quality Checks")
render_top_nav("dq_review_page")

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


# ---------------------------------------------------------------------------
# Helper — call Cortex LLM (dual-mode)
# ---------------------------------------------------------------------------
def call_cortex(prompt: str, model_name: str):
    """Call Snowflake Cortex — supports both Snowpark session and connector."""
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


# Initialize session state
if "current_check_index" not in st.session_state:
    st.session_state.current_check_index = 0

# Check if we have checks from marketplace
if not st.session_state.get('selected_checks'):
    st.error("⚠️ No checks found! Please go back to the marketplace and select some checks first.")
    if st.button("⬅️ Back to Marketplace"):
        _nav_to("pages/dq_marketplace.py")
    st.stop()

# Build list of selected checks
all_checks = []
for check_key, check_data in st.session_state.selected_checks.items():
    if check_data['selected']:
        all_checks.append({
            'key': check_key,
            'data': check_data['row']
        })

if not all_checks:
    st.warning("No checks selected. Please go back to the marketplace and select some checks.")
    if st.button("⬅️ Back to Marketplace", key="nav_back_mp_empty"):
        _nav_to("pages/dq_marketplace.py")
    st.stop()

# Sidebar navigation
with st.sidebar:
    st.header("📋 All Checks")
    st.caption(f"{len(all_checks)} checks to review")
    st.divider()

    for idx, check in enumerate(all_checks):
        row = check['data']
        is_current = (idx == st.session_state.current_check_index)

        label = f"{'✓' if is_current else '○'} {row['Critical Data Element']}"

        if st.button(
            label,
            key=f"nav_{idx}",
            use_container_width=True,
            type="primary" if is_current else "secondary"
        ):
            st.session_state.current_check_index = idx
            st.rerun()

    st.divider()

    if st.button("⬅️ Back to Marketplace", use_container_width=True):
        _nav_to("pages/dq_marketplace.py")

# Progress indicator
st.progress((st.session_state.current_check_index + 1) / len(all_checks))
st.caption(f"Check {st.session_state.current_check_index + 1} of {len(all_checks)}")

# Get current check
current_check = all_checks[st.session_state.current_check_index]
current_row = current_check['data']

# Create layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader(f"📋 {current_row['Critical Data Element']}")
    st.caption(f"*{current_row['Domain']} - {current_row['DQ Dimension']}*")
    st.divider()

    # CDE Name (read-only)
    st.markdown("**Critical Data Element:**")
    st.text_input(
        "CDE",
        value=current_row['Critical Data Element'],
        disabled=True,
        label_visibility="collapsed",
        key="cde_display"
    )

    # Domain (read-only)
    st.markdown("**Domain:**")
    st.text_input(
        "Domain",
        value=current_row['Domain'],
        disabled=True,
        label_visibility="collapsed",
        key="domain_display"
    )

    # DQ Dimension (read-only)
    st.markdown("**DQ Dimension:**")
    st.text_input(
        "Dimension",
        value=current_row['DQ Dimension'],
        disabled=True,
        label_visibility="collapsed",
        key="dimension_display"
    )

    # Check Name (read-only)
    st.markdown("**Check Name:**")
    st.text_input(
        "Check Name",
        value=current_row['Check name'],
        disabled=True,
        label_visibility="collapsed",
        key="checkname_display"
    )

    # Description (editable)
    st.markdown("**Check Description:**")
    new_description = st.text_area(
        "Description",
        value=current_row['Check Description'],
        height=120,
        label_visibility="collapsed",
        key=f"review_desc_{st.session_state.current_check_index}"
    )
    st.session_state.selected_checks[current_check['key']]['row']['Check Description'] = new_description

    # Short Description (read-only)
    st.markdown("**Short Description of CDE:**")
    st.text_area(
        "Short Desc",
        value=current_row['Short description of Critical Data Element'],
        height=80,
        disabled=True,
        label_visibility="collapsed",
        key="short_desc_display"
    )

with col_right:
    st.subheader("💻 SODA CL Check Definition")

    new_soda_check = st.text_area(
        "SODA CL",
        value=current_row['SODACL Yaml Check Definition'],
        height=400,
        label_visibility="collapsed",
        key=f"review_soda_{st.session_state.current_check_index}"
    )
    st.session_state.selected_checks[current_check['key']]['row']['SODACL Yaml Check Definition'] = new_soda_check

    # AI-powered check improvement
    col_improve, col_explain = st.columns(2)
    with col_improve:
        if st.button("🤖 AI Improve Check", key=f"ai_improve_{st.session_state.current_check_index}",
                     use_container_width=True):
            with st.spinner("Cortex AI is improving your check..."):
                improve_prompt = f"""You are a SODA CL data quality expert. Improve this check definition.

Column: {current_row['Critical Data Element']}
Domain: {current_row['Domain']}
DQ Dimension: {current_row['DQ Dimension']}
Description: {current_row['Check Description']}

Current SODA CL Check:
{new_soda_check}

Improve the check by:
1. Adding better thresholds or ranges where applicable
2. Handling edge cases (nulls, empty strings, out-of-range)
3. Using best-practice SODA CL syntax
4. Adding complementary sub-checks if beneficial

Return ONLY the improved SODA CL YAML. No explanation, no markdown fences."""

                improved = call_cortex(improve_prompt, st.session_state.get('model', 'llama3.1-70b'))
                if improved:
                    st.session_state[f'ai_improved_{st.session_state.current_check_index}'] = improved.strip()

    with col_explain:
        if st.button("💡 AI Explain Check", key=f"ai_explain_{st.session_state.current_check_index}",
                     use_container_width=True):
            with st.spinner("Cortex AI is explaining this check..."):
                explain_prompt = f"""Explain this SODA CL data quality check in simple terms.

Column: {current_row['Critical Data Element']}
DQ Dimension: {current_row['DQ Dimension']}

SODA CL Check:
{new_soda_check}

Provide:
1. What this check does in plain English
2. What values would PASS
3. What values would FAIL
4. Any limitations or edge cases

Keep it concise (3-5 sentences per point)."""

                explanation = call_cortex(explain_prompt, st.session_state.get('model', 'llama3.1-70b'))
                if explanation:
                    st.session_state[f'ai_explanation_{st.session_state.current_check_index}'] = explanation.strip()

    # Show AI-improved check
    if st.session_state.get(f'ai_improved_{st.session_state.current_check_index}'):
        st.markdown("**AI-Improved Check:**")
        st.code(st.session_state[f'ai_improved_{st.session_state.current_check_index}'], language="yaml")
        if st.button("✅ Accept Improvement", key=f"accept_improve_{st.session_state.current_check_index}",
                     use_container_width=True):
            st.session_state.selected_checks[current_check['key']]['row']['SODACL Yaml Check Definition'] = \
                st.session_state[f'ai_improved_{st.session_state.current_check_index}']
            del st.session_state[f'ai_improved_{st.session_state.current_check_index}']
            st.rerun()

    # Show AI explanation
    if st.session_state.get(f'ai_explanation_{st.session_state.current_check_index}'):
        with st.expander("💡 AI Explanation", expanded=True):
            st.markdown(st.session_state[f'ai_explanation_{st.session_state.current_check_index}'])

st.divider()

# Navigation buttons
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    if st.button("⬅️ Previous", use_container_width=True, disabled=st.session_state.current_check_index == 0):
        st.session_state.current_check_index -= 1
        st.rerun()

with col2:
    if st.button("➡️ Next", use_container_width=True, disabled=st.session_state.current_check_index >= len(all_checks) - 1):
        st.session_state.current_check_index += 1
        st.rerun()

with col3:
    if st.button("🧪 Test Check", use_container_width=True, type="secondary"):
        st.session_state[f'show_test_{st.session_state.current_check_index}'] = True

with col4:
    if st.button("✅ Approve", use_container_width=True, type="primary"):
        st.success(f"✅ Approved: {current_row['Check name']}")
        if st.session_state.current_check_index < len(all_checks) - 1:
            st.session_state.current_check_index += 1
            st.rerun()
        else:
            st.balloons()
            st.info("🎉 All checks reviewed!")




# ===========================================================================
# Test functionality section
# ===========================================================================
if st.session_state.get(f'show_test_{st.session_state.current_check_index}', False):
    st.divider()
    st.subheader("🧪 Test Check with Sample Values")

    col_test_left, col_test_right = st.columns([1, 1])

    with col_test_left:
        st.markdown("**Enter test values (one per line):**")
        st.caption(f"Testing: {current_row['Critical Data Element']}")

        cde_name = current_row['Critical Data Element']
        has_synthetic = (
            st.session_state.get('synthetic_values') and
            cde_name in st.session_state.synthetic_values and
            len(st.session_state.synthetic_values[cde_name]) > 0
        )

        if has_synthetic:
            st.info(f"✨ {len(st.session_state.synthetic_values[cde_name])} AI-generated values available")

            use_synthetic = st.checkbox(
                "Use AI-Generated Synthetic Values",
                value=True,
                key=f"use_synthetic_{st.session_state.current_check_index}"
            )

            if use_synthetic:
                with st.expander("🔍 View Synthetic Values", expanded=True):
                    synth_vals = st.session_state.synthetic_values[cde_name]
                    st.caption("AI-generated test values (includes valid and invalid cases):")
                    for idx, sv in enumerate(synth_vals):
                        value = sv.get('value', '')
                        st.info(f"**Value {idx+1}:** `{value}`")

                synthetic_text = '\n'.join([
                    str(sv.get('value', ''))
                    for sv in st.session_state.synthetic_values[cde_name]
                    if sv.get('value') is not None
                ])
                sample_values_input = st.text_area(
                    "Sample Values",
                    value=synthetic_text,
                    height=200,
                    key=f"test_values_{st.session_state.current_check_index}",
                    label_visibility="collapsed"
                )
            else:
                sample_values_input = st.text_area(
                    "Sample Values",
                    placeholder="Enter values to test, one per line:\n32.95847500\n33.74893200\n-91.5\n0.0\nabc123",
                    height=200,
                    key=f"test_values_manual_{st.session_state.current_check_index}",
                    label_visibility="collapsed"
                )
        else:
            sample_values_input = st.text_area(
                "Sample Values",
                placeholder="Enter values to test, one per line:\n32.95847500\n33.74893200\n-91.5\n0.0\nabc123",
                height=200,
                key=f"test_values_{st.session_state.current_check_index}",
                label_visibility="collapsed"
            )

        col_test_btn, col_close_btn = st.columns([1, 1])

        with col_test_btn:
            test_button = st.button(
                "▶️ Run Test",
                type="primary",
                use_container_width=True,
                key=f"run_test_{st.session_state.current_check_index}"
            )

        with col_close_btn:
            if st.button(
                "✖️ Close Test",
                use_container_width=True,
                key=f"close_test_{st.session_state.current_check_index}"
            ):
                st.session_state[f'show_test_{st.session_state.current_check_index}'] = False
                st.rerun()

    with col_test_right:
        st.markdown("**Test Results:**")

        if test_button and sample_values_input.strip():
            test_values = [v.strip() for v in sample_values_input.strip().split('\n') if v.strip()]

            with st.spinner("🤖 AI is testing your values..."):
                soda_check = current_row['SODACL Yaml Check Definition']
                dimension = current_row['DQ Dimension']
                cde_name = current_row['Critical Data Element']

                test_prompt = f"""You are a data quality validator. Test if the provided sample values pass or fail the SODA CL check.

Column Name: {cde_name}
DQ Dimension: {dimension}

SODA CL Check:
{soda_check}

Sample Values to Test:
{chr(10).join(test_values)}

For each value, determine if it would PASS or FAIL the check and explain why.

Respond ONLY with valid JSON:
{{
  "results": [
    {{
      "value": "32.95847500",
      "status": "PASS",
      "reason": "Valid latitude in range [-90, 90]"
    }},
    {{
      "value": "abc123",
      "status": "FAIL",
      "reason": "Not a valid numeric format"
    }}
  ]
}}

Output only JSON, no markdown."""

                try:
                    response = call_cortex(test_prompt, st.session_state.model)

                    if response:
                        # Parse response
                        cleaned = response.strip()
                        if '```' in cleaned:
                            parts = cleaned.split('```')
                            if len(parts) >= 3:
                                cleaned = parts[1].replace('json', '', 1)

                        start = cleaned.find('{')
                        end = cleaned.rfind('}')
                        if start != -1 and end != -1:
                            cleaned = cleaned[start:end + 1]

                        test_results = json.loads(cleaned)

                        # Display results
                        for test_result in test_results.get('results', []):
                            value = test_result.get('value', '')
                            status = test_result.get('status', 'UNKNOWN')
                            reason = test_result.get('reason', '')

                            if status == "PASS":
                                st.success(f"✅ **PASS**: `{value}`")
                                st.caption(reason)
                            elif status == "FAIL":
                                st.error(f"❌ **FAIL**: `{value}`")
                                st.caption(reason)
                            else:
                                st.warning(f"⚠️ **UNKNOWN**: `{value}`")
                                st.caption(reason)

                        # Summary
                        pass_count = sum(1 for r in test_results.get('results', []) if r.get('status') == 'PASS')
                        fail_count = sum(1 for r in test_results.get('results', []) if r.get('status') == 'FAIL')

                        st.divider()
                        col_pass, col_fail = st.columns(2)
                        with col_pass:
                            st.metric("✅ Passed", pass_count)
                        with col_fail:
                            st.metric("❌ Failed", fail_count)
                    else:
                        st.error("No response from LLM")

                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse test results: {str(e)}")
                    with st.expander("Raw response"):
                        st.code(response)
                except Exception as e:
                    st.error(f"Test failed: {str(e)}")

        elif test_button and not sample_values_input.strip():
            st.warning("⚠️ Please enter some test values first")

st.divider()

# ===========================================================================
# Final export — dual mode: download locally, save to table in SiS
# ===========================================================================
st.subheader("📥 Export All Approved Checks")

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

    col_export1, col_export2 = st.columns(2)

    with col_export1:
        if IS_LOCAL:
            csv_data = selected_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv_data,
                file_name="approved_checks.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            _appr_table_fq = get_output_table("APPROVED_CHECKS_TABLE")
            st.caption(f"Target: `{_appr_table_fq}`")
            if st.button("💾 Save Approved Checks to Snowflake", use_container_width=True):
                try:
                    if session is not None:
                        snowpark_df = session.create_dataframe(selected_df)
                        snowpark_df.write.mode("overwrite").save_as_table(_appr_table_fq)
                    elif st.session_state.get("connection"):
                        from snowflake.connector.pandas_tools import write_pandas
                        write_pandas(
                            st.session_state.connection,
                            selected_df,
                            _appr_table_fq,
                            auto_create_table=True,
                            overwrite=True
                        )
                    else:
                        st.error("Not connected to Snowflake.")
                    st.success(f"✅ Saved to `{_appr_table_fq}`")
                except Exception as e:
                    st.error(f"Save failed: {e}")

    with col_export2:
        if IS_LOCAL:
            st.download_button(
                "📥 Download YAML",
                data=final_yaml,
                file_name="dq_checks_final.yml",
                mime="text/yaml",
                use_container_width=True,
                type="primary"
            )
        else:
            if st.button("📄 View Final YAML", use_container_width=True, type="primary"):
                st.session_state['show_final_yaml'] = True

    if not IS_LOCAL and st.session_state.get('show_final_yaml', False):
        st.subheader("📄 Final SODACL YAML")
        st.code(final_yaml, language="yaml")
        if st.button("Close YAML Preview"):
            st.session_state['show_final_yaml'] = False
            st.rerun()
else:
    st.warning("No checks selected")
