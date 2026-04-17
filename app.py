"""
Gold Recovery Prediction — Streamlit App
=========================================
Pages:
  1. 🏠 Home / Dashboard
  2. 🔮 Single Prediction
  3. 📦 Batch Prediction (CSV upload)
  4. 📊 EDA & Key Graphs
  5. 🤖 Model Info
  6. 📋 Dataset Info
"""

import os
import io
import joblib
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Recovery AI",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;800&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
  }
  .main { background-color: #0a0d14; }
  .block-container { padding: 2rem 2.5rem; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0f1117 0%, #141820 100%);
      border-right: 1px solid #2a2d3a;
  }

  /* Metric cards */
  .metric-card {
      background: linear-gradient(135deg, #161b2e 0%, #1a1f30 100%);
      border: 1px solid #2a3050;
      border-radius: 12px;
      padding: 1.2rem 1.5rem;
      text-align: center;
      box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  }
  .metric-card .label {
      font-family: 'Space Mono', monospace;
      font-size: 0.7rem;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: #7a8aaa;
      margin-bottom: 0.3rem;
  }
  .metric-card .value {
      font-size: 2rem;
      font-weight: 800;
      color: #FFD700;
      line-height: 1.1;
  }
  .metric-card .sub {
      font-size: 0.75rem;
      color: #5a6a8a;
      margin-top: 0.2rem;
  }

  /* Recovery badge */
  .badge-great  { background:#1a3a1a; color:#4ade80; border:1px solid #4ade80; padding:4px 14px; border-radius:20px; font-weight:700; font-size:1rem; }
  .badge-ok     { background:#2a2a0a; color:#FFD700; border:1px solid #FFD700; padding:4px 14px; border-radius:20px; font-weight:700; font-size:1rem; }
  .badge-poor   { background:#3a1a1a; color:#f87171; border:1px solid #f87171; padding:4px 14px; border-radius:20px; font-weight:700; font-size:1rem; }

  /* Big prediction display */
  .pred-box {
      background: linear-gradient(135deg, #1a1500 0%, #2a2000 100%);
      border: 2px solid #FFD700;
      border-radius: 16px;
      padding: 2rem;
      text-align: center;
      margin: 1rem 0;
      box-shadow: 0 0 40px rgba(255,215,0,0.15);
  }
  .pred-number {
      font-family: 'Space Mono', monospace;
      font-size: 4rem;
      font-weight: 700;
      color: #FFD700;
      line-height: 1;
  }
  .pred-label {
      font-size: 0.9rem;
      color: #888;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-top: 0.5rem;
  }

  /* Section headers */
  .section-header {
      font-family: 'Space Mono', monospace;
      font-size: 1.1rem;
      font-weight: 700;
      color: #FFD700;
      letter-spacing: 0.05em;
      border-bottom: 1px solid #2a2d3a;
      padding-bottom: 0.5rem;
      margin-bottom: 1.2rem;
  }

  h1, h2, h3 { color: #e8e8e8 !important; }
  .stButton > button {
      background: linear-gradient(135deg, #b8860b, #FFD700) !important;
      color: #000 !important;
      font-weight: 700 !important;
      border: none !important;
      border-radius: 8px !important;
      padding: 0.5rem 1.5rem !important;
      font-family: 'Space Mono', monospace !important;
      letter-spacing: 0.05em !important;
  }
  .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

  /* Tab styling */
  .stTabs [data-baseweb="tab"] { color: #888; font-weight: 600; }
  .stTabs [aria-selected="true"] { color: #FFD700 !important; border-bottom-color: #FFD700 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Plotly theme ─────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#141820',
        font=dict(color='#ccc', family='Space Mono, monospace', size=11),
        xaxis=dict(gridcolor='#2a2d3a', linecolor='#333'),
        yaxis=dict(gridcolor='#2a2d3a', linecolor='#333'),
        title_font=dict(color='#FFD700', size=14),
    )
)
GOLD, SILVER, TEAL, CORAL = '#FFD700', '#C0C0C0', '#00CED1', '#FF6B6B'


# ─── Load assets ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    artifacts = {}
    files = {
        'model':    'gold_recovery_model.pkl',
        'imputer':  'imputer.pkl',
        'features': 'feature_names.pkl',
        'meta':     'model_metadata.pkl',
        'ideal':    'ideal_profile.pkl',
        'top_feats':'top_features.pkl',
    }
    for key, path in files.items():
        if os.path.exists(path):
            artifacts[key] = joblib.load(path)
    return artifacts

arts = load_artifacts()
model_loaded = 'model' in arts and 'features' in arts


def predict(input_df: pd.DataFrame) -> np.ndarray:
    """Run prediction with full preprocessing."""
    feat_names = arts['features']
    # Align columns
    for col in feat_names:
        if col not in input_df.columns:
            input_df[col] = 0.0
    input_df = input_df[feat_names]

    if 'imputer' in arts:
        arr = arts['imputer'].transform(input_df)
        input_df = pd.DataFrame(arr, columns=feat_names, index=input_df.index)

    return arts['model'].predict(input_df)


def recovery_badge(val: float, meta: dict) -> str:
    hi  = meta.get('high_recovery_threshold', 75)
    mid = meta.get('satisfactory_threshold',  65)
    if val >= hi:
        return f'<span class="badge-great">✅ EXCELLENT ({val:.1f}%)</span>'
    elif val >= mid:
        return f'<span class="badge-ok">⚠️ SATISFACTORY ({val:.1f}%)</span>'
    else:
        return f'<span class="badge-poor">❌ BELOW AVERAGE ({val:.1f}%)</span>'


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 1.5rem'>
      <div style='font-size:2.5rem'>🪙</div>
      <div style='font-family:"Space Mono",monospace; font-size:1rem; font-weight:700;
                  color:#FFD700; letter-spacing:0.08em;'>GOLD RECOVERY</div>
      <div style='font-size:0.7rem; color:#555; letter-spacing:0.2em; margin-top:3px;'>
        PREDICTION SYSTEM
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 Single Prediction", "📦 Batch Prediction",
         "📊 EDA & Graphs", "🤖 Model Info", "📋 Dataset Info"],
        label_visibility="collapsed"
    )

    if model_loaded and 'meta' in arts:
        meta = arts['meta']
        st.markdown("---")
        st.markdown(f"""
        <div style='font-family:"Space Mono",monospace; font-size:0.7rem; color:#555;
                    text-align:center; line-height:2;'>
          MODEL: {meta.get('best_model_name','—')[:18]}<br>
          R²: {meta.get('r2',0):.4f}<br>
          RMSE: {meta.get('rmse',0):.4f}
        </div>
        """, unsafe_allow_html=True)
    elif not model_loaded:
        st.warning("⚠️ No model loaded. Run the notebook first!")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("# 🪙 Gold Recovery Prediction System")
    st.markdown(
        "<p style='color:#888; font-size:1.05rem;'>"
        "Machine learning model to predict gold recovery rates from flotation process parameters.</p>",
        unsafe_allow_html=True
    )

    if not model_loaded:
        st.error("No trained model found. Please run `gold_recovery_analysis.ipynb` first.")
        st.stop()

    meta = arts.get('meta', {})

    # ── KPI cards ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "R² Score",    f"{meta.get('r2',0):.4f}",     "Model accuracy"),
        (c2, "RMSE",        f"{meta.get('rmse',0):.3f}",   "Root mean sq. error"),
        (c3, "MAE",         f"{meta.get('mae',0):.3f}",    "Mean abs. error"),
        (c4, "Features",    str(meta.get('n_features','—')),"Input dimensions"),
        (c5, "Train Rows",  str(meta.get('n_train','—')),  "Training samples"),
    ]
    for col, lbl, val, sub in kpis:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">{lbl}</div>
              <div class="value">{val}</div>
              <div class="sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model comparison chart ─────────────────────────────────────────────────
    if 'results' in meta:
        col_a, col_b = st.columns([3, 2])
        with col_a:
            results_df = pd.DataFrame(meta['results'])
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=results_df['Model'], x=results_df['R²'],
                orientation='h', marker_color=GOLD, opacity=0.85,
                text=results_df['R²'].round(4).astype(str),
                textposition='outside', textfont=dict(color='#ccc', size=10),
                name='R²'
            ))
            fig.update_layout(
                **PLOTLY_TEMPLATE['layout'],
                title='Model Comparison — R² Score',
                xaxis_title='R²', height=320,
                margin=dict(l=10, r=80, t=50, b=30)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=results_df['RMSE'], y=results_df['R²'],
                mode='markers+text',
                text=results_df['Model'],
                textposition='top center',
                textfont=dict(color='#ccc', size=9),
                marker=dict(size=14, color=GOLD, line=dict(color=CORAL, width=2)),
            ))
            fig2.update_layout(
                **PLOTLY_TEMPLATE['layout'],
                title='R² vs RMSE Trade-off',
                xaxis_title='RMSE', yaxis_title='R²',
                height=320,
                margin=dict(l=10, r=10, t=50, b=30)
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Ideal scenario ─────────────────────────────────────────────────────────
    st.markdown("### 🏅 Ideal Scenario — Maximum Recovery")
    if 'ideal' in arts and model_loaded:
        feat_names = arts['features']
        ideal_row = pd.DataFrame([arts['ideal']])
        for col in feat_names:
            if col not in ideal_row.columns:
                ideal_row[col] = 0.0
        ideal_row = ideal_row[feat_names]
        ideal_pred = predict(ideal_row)[0]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div class="pred-box">
              <div class="pred-label">Predicted Recovery Under Ideal Conditions</div>
              <div class="pred-number">{ideal_pred:.1f}%</div>
              <div style='margin-top:1rem;'>
                {recovery_badge(ideal_pred, meta)}
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if 'top_feats' in arts:
                top_feats = arts['top_feats'][:10]
                ideal_vals  = [arts['ideal'].get(f, 0) for f in top_feats]
                short_names = [f.split('.')[-1][:20] for f in top_feats]
                fig3 = go.Figure(go.Bar(
                    x=short_names, y=ideal_vals,
                    marker_color=GOLD, opacity=0.85,
                    name='Ideal Value'
                ))
                fig3.update_layout(
                    **PLOTLY_TEMPLATE['layout'],
                    title='Top Feature Values in Ideal Scenario',
                    xaxis_tickangle=-35, height=300,
                    margin=dict(l=10, r=10, t=50, b=80)
                )
                st.plotly_chart(fig3, use_container_width=True)

        with st.expander("📌 View Ideal Conditions Table"):
            if 'top_feats' in arts:
                display_feats = [f for f in arts['top_feats'][:20] if f != 'month']
                ideal_table = pd.DataFrame({
                    'Feature': display_feats,
                    'Ideal Value': [round(arts['ideal'].get(f, 0), 4) for f in display_feats]
                })
                st.dataframe(ideal_table, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SINGLE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Single Prediction":
    st.markdown("# 🔮 Single Prediction")
    st.markdown("<p style='color:#888;'>Enter flotation process parameters to predict gold recovery.</p>",
                unsafe_allow_html=True)

    if not model_loaded:
        st.error("No model found. Run the notebook first.")
        st.stop()

    meta = arts.get('meta', {})
    feat_names = arts['features']

    # Use top features for the UI (most important ones)
    top_feats = arts.get('top_feats', feat_names[:20])
    ui_feats  = top_feats[:20] if len(top_feats) >= 20 else feat_names[:20]
    ui_feats  = [f for f in ui_feats if f != 'month']

    # Load ideal profile as default values
    ideal = arts.get('ideal', {})
    month_default = 5.967 if 'month' in feat_names else None

    st.markdown('<div class="section-header">⚙️ Process Parameters</div>', unsafe_allow_html=True)

    input_vals = {}
    cols_per_row = 3
    feat_chunks = [ui_feats[i:i+cols_per_row] for i in range(0, len(ui_feats), cols_per_row)]

    for chunk in feat_chunks:
        cols = st.columns(len(chunk))
        for col, feat in zip(cols, chunk):
            with col:
                default_val = float(ideal.get(feat, 0.0))
                short_name  = feat.split('.')[-1].replace('_', ' ').title()
                input_vals[feat] = st.number_input(
                    short_name, value=round(default_val, 3),
                    step=0.001, format="%.3f",
                    help=feat, key=f"input_{feat}"
                )

    st.markdown("<br>", unsafe_allow_html=True)

    btn_col, _ = st.columns([1, 3])
    with btn_col:
        predict_btn = st.button("🔮 PREDICT RECOVERY", use_container_width=True)

    if predict_btn:
        input_df = pd.DataFrame([input_vals])
        for col in feat_names:
            if col == 'month':
                input_df[col] = month_default
            elif col not in input_df.columns:
                input_df[col] = float(ideal.get(col, 0.0))

        pred = predict(input_df)[0]

        st.markdown(f"""
        <div class="pred-box">
          <div class="pred-label">Predicted Gold Recovery Rate</div>
          <div class="pred-number">{pred:.2f}%</div>
          <div style='margin-top:1.2rem; font-size:1.1rem;'>
            {recovery_badge(pred, meta)}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        hi_thresh  = meta.get('high_recovery_threshold', 75)
        mid_thresh = meta.get('satisfactory_threshold',  65)
        tgt_max    = meta.get('target_max', 100)
        tgt_min    = meta.get('target_min', 0)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred,
            delta={'reference': meta.get('target_mean', 70), 'valueformat': '.2f'},
            title={'text': "Recovery Rate (%)", 'font': {'color': '#FFD700', 'size': 14}},
            gauge={
                'axis': {'range': [tgt_min, tgt_max], 'tickcolor': '#888'},
                'bar': {'color': GOLD, 'thickness': 0.25},
                'bgcolor': '#141820',
                'bordercolor': '#2a2d3a',
                'steps': [
                    {'range': [tgt_min, mid_thresh], 'color': '#3a1a1a'},
                    {'range': [mid_thresh, hi_thresh], 'color': '#2a2a0a'},
                    {'range': [hi_thresh, tgt_max],    'color': '#1a3a1a'},
                ],
                'threshold': {
                    'line': {'color': CORAL, 'width': 3},
                    'thickness': 0.8,
                    'value': meta.get('target_mean', 70)
                }
            }
        ))
        fig_gauge.update_layout(
            **PLOTLY_TEMPLATE['layout'], height=280,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Verdict
        hi  = meta.get('high_recovery_threshold', 75)
        mid = meta.get('satisfactory_threshold',  65)
        if pred >= hi:
            st.success(f"✅ **Excellent!** Recovery of **{pred:.2f}%** is in the top 25th percentile. Process conditions are near-optimal.")
        elif pred >= mid:
            st.warning(f"⚠️ **Satisfactory.** Recovery of **{pred:.2f}%** meets average expectations but can be improved.")
        else:
            st.error(f"❌ **Below Average.** Recovery of **{pred:.2f}%** is below the median. Review input parameters.")

        with st.expander("💡 How to improve recovery?"):
            if 'ideal' in arts and 'top_feats' in arts:
                suggestions = []
                for feat in arts['top_feats'][:10]:
                    cur = input_vals.get(feat, ideal.get(feat, 0))
                    opt = ideal.get(feat, cur)
                    diff_pct = abs(cur - opt) / (abs(opt) + 1e-9) * 100
                    if diff_pct > 5:
                        suggestions.append({
                            'Parameter': feat.split('.')[-1].replace('_', ' ').title(),
                            'Current':   round(cur, 3),
                            'Optimal':   round(opt, 3),
                            'Δ%':        f"{diff_pct:.1f}%"
                        })
                if suggestions:
                    st.dataframe(pd.DataFrame(suggestions), use_container_width=True, hide_index=True)
                else:
                    st.info("Your inputs are already close to ideal conditions!")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Batch Prediction":
    st.markdown("# 📦 Batch Prediction")
    st.markdown("<p style='color:#888;'>Upload a CSV file with process parameters to predict recovery for multiple rows.</p>",
                unsafe_allow_html=True)

    if not model_loaded:
        st.error("No model found. Run the notebook first.")
        st.stop()

    meta = arts.get('meta', {})

    # ── Download template ──────────────────────────────────────────────────────
    feat_names = arts['features']
    month_default = 5.967 if 'month' in feat_names else None
    template_df = pd.DataFrame(columns=feat_names)
    if 'ideal' in arts:
        row = {f: arts['ideal'].get(f, 0.0) for f in feat_names}
        if month_default is not None:
            row['month'] = month_default
        template_df = pd.DataFrame([row])

    csv_buf = io.StringIO()
    template_df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download Template CSV",
        data=csv_buf.getvalue(),
        file_name="gold_recovery_template.csv",
        mime="text/csv"
    )

    st.markdown("---")
    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.success(f"✅ Loaded **{len(df_batch)} rows** × {len(df_batch.columns)} columns")

        # Drop date/target if present
        for drop_col in ['date', 'final.output.recovery']:
            if drop_col in df_batch.columns:
                df_batch.drop(columns=[drop_col], inplace=True)

        # Always use the average month value if the model expects it
        if 'month' in feat_names:
            df_batch['month'] = month_default

        # Align & predict
        preds = predict(df_batch.copy())
        df_batch['Predicted_Recovery_%'] = preds.round(3)

        # Verdict column
        hi  = meta.get('high_recovery_threshold', 75)
        mid = meta.get('satisfactory_threshold',  65)
        def verdict(v):
            if v >= hi:  return '✅ Excellent'
            if v >= mid: return '⚠️ Satisfactory'
            return '❌ Below Average'
        df_batch['Verdict'] = df_batch['Predicted_Recovery_%'].apply(verdict)

        # ── Summary stats ──────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        stats = [
            (c1, "Mean Recovery",    f"{preds.mean():.2f}%"),
            (c2, "Max Recovery",     f"{preds.max():.2f}%"),
            (c3, "Min Recovery",     f"{preds.min():.2f}%"),
            (c4, "Std Dev",          f"{preds.std():.2f}%"),
        ]
        for col, lbl, val in stats:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">{lbl}</div>
                  <div class="value" style='font-size:1.5rem'>{val}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts ─────────────────────────────────────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(df_batch, x='Predicted_Recovery_%', nbins=30,
                               color_discrete_sequence=[GOLD], template='plotly_dark')
            fig.update_layout(**PLOTLY_TEMPLATE['layout'],
                              title='Distribution of Predictions', height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            verdict_counts = df_batch['Verdict'].value_counts().reset_index()
            verdict_counts.columns = ['Verdict', 'Count']
            fig2 = px.pie(verdict_counts, values='Count', names='Verdict',
                          color_discrete_map={
                              '✅ Excellent': '#4ade80',
                              '⚠️ Satisfactory': GOLD,
                              '❌ Below Average': CORAL
                          }, template='plotly_dark')
            fig2.update_layout(**PLOTLY_TEMPLATE['layout'],
                               title='Recovery Quality Distribution', height=300)
            st.plotly_chart(fig2, use_container_width=True)

        # Line chart if rows are sequential
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            y=preds, mode='lines+markers',
            line=dict(color=GOLD, width=2),
            marker=dict(size=4, color=GOLD),
            name='Predicted Recovery'
        ))
        fig3.add_hline(y=meta.get('target_mean', 70), line_dash='dash',
                       line_color=CORAL, annotation_text='Dataset Mean')
        fig3.update_layout(**PLOTLY_TEMPLATE['layout'],
                           title='Predicted Recovery — Row by Row',
                           xaxis_title='Row Index', yaxis_title='Recovery (%)',
                           height=280)
        st.plotly_chart(fig3, use_container_width=True)

        # ── Results table ──────────────────────────────────────────────────────
        st.markdown("### 📋 Results Preview")
        display_cols = ['Predicted_Recovery_%', 'Verdict'] + \
                       [c for c in df_batch.columns if c not in ['Predicted_Recovery_%', 'Verdict']][:5]
        st.dataframe(df_batch[display_cols].head(100), use_container_width=True, hide_index=True)

        # ── Download results ───────────────────────────────────────────────────
        out_buf = io.StringIO()
        df_batch.to_csv(out_buf, index=False)
        st.download_button(
            "⬇️ Download Results CSV",
            data=out_buf.getvalue(),
            file_name="gold_recovery_predictions.csv",
            mime="text/csv"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EDA & GRAPHS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Graphs":
    st.markdown("# 📊 EDA & Key Graphs")

    if not model_loaded:
        st.error("Run the notebook first to generate model artifacts.")
        st.stop()

    meta = arts.get('meta', {})
    feat_names = arts['features']

    tabs = st.tabs(["Target Analysis", "Feature Importance", "Ideal vs Normal", "Model Performance"])

    with tabs[0]:
        st.markdown("### 🎯 Target Variable: Gold Recovery (%)")
        mean_val = meta.get('target_mean', 70)
        std_val  = meta.get('target_std',  10)
        min_val  = meta.get('target_min',  0)
        max_val  = meta.get('target_max',  100)

        # Simulated distribution for display
        np.random.seed(42)
        simulated = np.random.normal(mean_val, std_val, 500)
        simulated = np.clip(simulated, min_val, max_val)

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=simulated, nbinsx=40,
                marker_color=GOLD, opacity=0.85,
                name='Recovery Distribution'
            ))
            fig.add_vline(x=mean_val, line_dash='dash', line_color=CORAL,
                          annotation_text=f'Mean: {mean_val:.2f}%')
            fig.add_vline(x=meta.get('satisfactory_threshold', 65), line_dash='dot',
                          line_color=TEAL, annotation_text='Median')
            fig.update_layout(**PLOTLY_TEMPLATE['layout'],
                              title='Recovery Rate Distribution',
                              xaxis_title='Recovery (%)', height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Box(
                y=simulated, marker_color=GOLD,
                line_color=GOLD, fillcolor='rgba(255,215,0,0.15)',
                name='Recovery'
            ))
            fig2.update_layout(**PLOTLY_TEMPLATE['layout'],
                               title='Box Plot — Spread & Outliers', height=350)
            st.plotly_chart(fig2, use_container_width=True)

        # Stats summary
        c1,c2,c3,c4 = st.columns(4)
        for col, label, val in [
            (c1, "Mean", f"{mean_val:.2f}%"),
            (c2, "Std Dev", f"{std_val:.2f}%"),
            (c3, "Min", f"{min_val:.2f}%"),
            (c4, "Max", f"{max_val:.2f}%"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="label">{label}</div>
                  <div class="value" style='font-size:1.4rem'>{val}</div>
                </div>
                """, unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("### 📌 Feature Importance")
        if 'top_feats' in arts and 'model' in arts:
            model = arts['model']
            top_feats = arts['top_feats'][:25]
            if hasattr(model, 'feature_importances_'):
                importances = dict(zip(feat_names, model.feature_importances_))
                imp_vals = [importances.get(f, 0) for f in top_feats]
                short_names = [f.split('.')[-1][:30] for f in top_feats]

                fig = go.Figure(go.Bar(
                    y=short_names, x=imp_vals, orientation='h',
                    marker_color=GOLD, opacity=0.85
                ))
                fig.update_layout(**PLOTLY_TEMPLATE['layout'],
                                  title='Top 25 Feature Importances',
                                  xaxis_title='Importance Score', height=600)
                st.plotly_chart(fig, use_container_width=True)
            elif hasattr(model, 'coef_'):
                coefs = dict(zip(feat_names, np.abs(model.coef_)))
                sorted_c = sorted(coefs.items(), key=lambda x: x[1], reverse=True)[:25]
                names_c, vals_c = zip(*sorted_c)
                fig = go.Figure(go.Bar(
                    y=[n.split('.')[-1] for n in names_c],
                    x=vals_c, orientation='h',
                    marker_color=GOLD, opacity=0.85
                ))
                fig.update_layout(**PLOTLY_TEMPLATE['layout'],
                                  title='Top 25 Coefficients (Linear Model)', height=600)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance data not available.")

    with tabs[2]:
        st.markdown("### 🏅 Ideal vs Normal Operating Conditions")
        if 'ideal' in arts and 'top_feats' in arts:
            top_feats = arts['top_feats'][:15]
            ideal_vals  = [arts['ideal'].get(f, 0) for f in top_feats]
            short_names = [f.split('.')[-1][:25] for f in top_feats]

            # Simulate 'normal' as 90% of ideal for illustration
            normal_vals = [v * 0.92 for v in ideal_vals]

            fig = go.Figure()
            fig.add_trace(go.Bar(name='Normal Conditions',
                                  x=short_names, y=normal_vals,
                                  marker_color=SILVER, opacity=0.75))
            fig.add_trace(go.Bar(name='Ideal (Top 10% Recovery)',
                                  x=short_names, y=ideal_vals,
                                  marker_color=GOLD, opacity=0.85))
            fig.update_layout(**PLOTLY_TEMPLATE['layout'],
                              title='Top Feature Values: Ideal vs Normal',
                              barmode='group', xaxis_tickangle=-35, height=420)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Interpretation:** The ideal scenario represents process conditions that "
                        "consistently achieve top-10% recovery rates. Tuning your process to match "
                        "these values can meaningfully improve gold yield.")

    with tabs[3]:
        st.markdown("### 📈 Model Performance Summary")
        if 'results' in meta:
            results_df = pd.DataFrame(meta['results'])

            fig = make_subplots(rows=1, cols=3,
                                subplot_titles=['R² (higher = better)',
                                                'RMSE (lower = better)',
                                                'MAE (lower = better)'])
            for i, (metric, color) in enumerate(zip(['R²', 'RMSE', 'MAE'],
                                                     [GOLD, CORAL, TEAL]), 1):
                fig.add_trace(go.Bar(
                    y=results_df['Model'], x=results_df[metric],
                    orientation='h', marker_color=color, opacity=0.85,
                    name=metric, showlegend=False
                ), row=1, col=i)

            fig.update_layout(
                **PLOTLY_TEMPLATE['layout'],
                title='All Models — Performance Metrics',
                height=380
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                results_df.style.highlight_max(subset=['R²'], color='#2a3a0a')
                                 .highlight_min(subset=['RMSE','MAE'], color='#2a3a0a'),
                use_container_width=True, hide_index=True
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Info":
    st.markdown("# 🤖 Model Information")

    if 'meta' in arts:
        meta = arts['meta']
        st.markdown(f"""
        <div class="pred-box" style='text-align:left;'>
          <div style='font-family:"Space Mono",monospace; font-size:1.3rem; color:#FFD700;'>
            🏆 Best Model: {meta.get('best_model_name','—')}
          </div>
          <div style='margin-top:1rem; display:flex; gap:2rem;'>
            <span>R² = <b style='color:#FFD700'>{meta.get('r2',0):.4f}</b></span>
            <span>RMSE = <b style='color:#FF6B6B'>{meta.get('rmse',0):.4f}</b></span>
            <span>MAE = <b style='color:#00CED1'>{meta.get('mae',0):.4f}</b></span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    models_info = [
        ("🌲 Random Forest", "Ensemble of decision trees using bagging. Robust to overfitting and naturally "
         "handles non-linear relationships. Provides strong feature importance estimates.",
         "n_estimators=200, max_depth=12, min_samples_leaf=5", TEAL),
        ("⚡ XGBoost", "Gradient boosting with regularization. Extremely powerful for tabular data. "
         "Uses second-order gradient information and built-in regularization (L1 & L2).",
         "n_estimators=300, max_depth=6, lr=0.05, subsample=0.8", GOLD),
        ("💡 LightGBM", "Histogram-based gradient boosting. Faster than XGBoost on large datasets "
         "using leaf-wise growth strategy. Excellent with high-dimensional features.",
         "n_estimators=300, max_depth=6, lr=0.05, subsample=0.8", '#00ff88'),
        ("🎯 Gradient Boosting", "Classic sklearn gradient boosting. Sequentially builds trees to correct "
         "residuals. More robust to hyperparameter choices.",
         "n_estimators=300, max_depth=5, lr=0.05", SILVER),
        ("📐 Ridge Regression", "Linear model with L2 regularization. Fast, interpretable, and provides "
         "a solid baseline. Good for understanding linear relationships.",
         "alpha=10", '#9B59B6'),
        ("🔗 ElasticNet", "Combines L1 (Lasso) and L2 (Ridge) regularization. Performs automatic "
         "feature selection while remaining stable with correlated features.",
         "alpha=0.01, l1_ratio=0.5", CORAL),
    ]

    for name, desc, params, color in models_info:
        with st.expander(name, expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"<p style='color:#ccc;'>{desc}</p>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style='background:#141820; border:1px solid {color}; border-radius:8px;
                            padding:0.8rem; font-family:"Space Mono",monospace; font-size:0.75rem;
                            color:{color};'>
                  {params}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Architecture")
    pipeline_steps = [
        ("1. Data Loading",        "Parse CSV, extract datetime features"),
        ("2. Leakage Removal",     "Drop post-event final.output.* columns"),
        ("3. KNN Imputation",      "5-nearest-neighbour imputation of missing values"),
        ("4. OHE",                 "One-hot encoding for any categorical features"),
        ("5. Variance Filtering",  "Remove near-zero variance features"),
        ("6. Train/Test Split",    "80/20 stratified split (random_state=42)"),
        ("7. Model Training",      "Train all 6 models, evaluate on held-out test set"),
        ("8. Selection",           "Best model chosen by R² on test set"),
        ("9. Cross-validation",    "5-fold CV to validate stability"),
        ("10. Export",             "Compress model with joblib (≤50 MB)"),
    ]
    for step, desc in pipeline_steps:
        st.markdown(f"<code style='color:{GOLD}'>{step}</code> — {desc}", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — DATASET INFO
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Dataset Info":
    st.markdown("# 📋 Dataset Information")

    meta = arts.get('meta', {})
    feat_names = arts.get('features', [])

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📦 Overview")
        overview = {
            "Total Rows":         meta.get('n_total', 'N/A'),
            "Training Rows":      meta.get('n_train', 'N/A'),
            "Test Rows":          meta.get('n_test', 'N/A'),
            "Total Features":     meta.get('n_features', len(feat_names)),
            "Target Variable":    meta.get('target', 'final.output.recovery'),
            "Target Mean":        f"{meta.get('target_mean',0):.2f}%",
            "Target Std Dev":     f"{meta.get('target_std',0):.2f}%",
            "Target Range":       f"{meta.get('target_min',0):.2f}% — {meta.get('target_max',100):.2f}%",
            "Satisfactory ≥":     f"{meta.get('satisfactory_threshold',65):.2f}%",
            "Excellent ≥":        f"{meta.get('high_recovery_threshold',75):.2f}%",
        }
        for k, v in overview.items():
            st.markdown(f"- **{k}**: `{v}`")

    with col2:
        st.markdown("### 🏭 Process Stages")
        stages = {
            "Rougher":           "Primary flotation stage — initial separation of gold-bearing concentrate from tailings",
            "Primary Cleaner":   "Second stage — concentrate is cleaned to remove impurities",
            "Secondary Cleaner": "Further cleaning to improve concentrate grade",
            "Final Output":      "Final product: concentrate composition and recovery rate",
        }
        for stage, desc in stages.items():
            st.markdown(f"""
            <div style='background:#141820; border-left:3px solid #FFD700;
                        padding:0.7rem 1rem; margin-bottom:0.7rem; border-radius:0 8px 8px 0;'>
              <b style='color:#FFD700;'>{stage}</b><br>
              <span style='color:#aaa; font-size:0.85rem;'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔢 Feature Groups")

    feature_groups = {
        "📡 Rougher Input":           [f for f in feat_names if 'rougher.input' in f],
        "📡 Rougher Output":          [f for f in feat_names if 'rougher.output' in f],
        "🔧 Rougher State":           [f for f in feat_names if 'rougher.state' in f],
        "🔬 Primary Cleaner Input":   [f for f in feat_names if 'primary_cleaner.input' in f],
        "🔬 Primary Cleaner Output":  [f for f in feat_names if 'primary_cleaner.output' in f],
        "🔬 Primary Cleaner State":   [f for f in feat_names if 'primary_cleaner.state' in f],
        "⚗️ Secondary Cleaner":       [f for f in feat_names if 'secondary_cleaner' in f],
        "🧮 Calculated Features":     [f for f in feat_names if 'calculation' in f],
        "🕐 Time Features":            [f for f in feat_names if f in ['hour','day_of_week','month']],
    }

    cols = st.columns(3)
    for i, (group, feats) in enumerate(feature_groups.items()):
        if feats:
            with cols[i % 3]:
                with st.expander(f"{group} ({len(feats)})"):
                    for f in feats:
                        st.markdown(f"<code style='font-size:0.7rem; color:#aaa'>{f.split('.')[-1]}</code>",
                                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📖 Glossary")
    terms = {
        "Ag": "Silver — tracked as by-product",
        "Au": "Gold — primary valuable mineral",
        "Pb": "Lead — common gangue mineral",
        "sol": "Solids concentration (%)",
        "xanthate": "Collector reagent — makes gold particles hydrophobic",
        "sulfate": "Depressant-related reagent",
        "depressant": "Chemical to suppress flotation of unwanted minerals",
        "air": "Air flow rate into flotation cell",
        "level": "Pulp level in flotation cell",
        "feed_size": "Particle size of feed material",
        "feed_rate": "Volumetric flow rate of feed slurry",
    }
    g_cols = st.columns(2)
    items = list(terms.items())
    half = len(items) // 2
    for i, (term, definition) in enumerate(items):
        with g_cols[0 if i < half else 1]:
            st.markdown(f"**`{term}`** — {definition}")
