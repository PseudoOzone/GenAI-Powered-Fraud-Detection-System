"""
Streamlit UI for GenAI Fraud Detection System
Displays pipeline results and model information
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import torch


st.set_page_config(
    page_title="GenAI Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- COMPLETE CSS FOR FINTECH REDESIGN V2 ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

    /* --- Base & Typography --- */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        color: #E2E8F0;
    }
    body {
        background-image: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        background-attachment: fixed;
    }
    h1, h2, h3 { text-shadow: 0px 2px 4px rgba(0,0,0,0.2); }
    h1 { font-size: 3rem; font-weight: 800; }
    p, .st-write, .st-markdown { font-size: 1rem; line-height: 1.6; }

    /* --- Header & Branding --- */
    .stApp > header { display: none; }
    div[data-testid="stVerticalBlock"]:has(> div > div > h1) {
        background: linear-gradient(180deg, rgba(30,58,138,0.3) 0%, rgba(26,22,60,0) 100%);
        text-align: center;
        padding: 3rem 1rem 2rem 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    div[data-testid="stVerticalBlock"]:has(> div > div > h1) h1 span:first-of-type {
        font-size: 3rem;
    }

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #3b21a8 100%) !important;
        padding: 1rem;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] label {
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px;
        transition: all 0.3s ease;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid transparent;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background-color: rgba(255,255,255,0.1);
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.5);
        border-color: rgba(139, 92, 246, 0.5);
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] input:checked + div {
        background-color: #1e3a8a;
        border-color: #8B5CF6;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] .st-emotion-cache-1we0j1p {
        color: #E2E8F0;
        font-size: 1.1rem;
    }

    /* --- Main Page Metric Cards --- */
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease;
    }
    .stMetric:hover { transform: translateY(-8px); }
    
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) .stMetric {
        background: linear-gradient(135deg, #1E40AF, #3B82F6);
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stMetric {
        background: linear-gradient(135deg, #d97706, #f59e0b);
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) .stMetric {
        background: linear-gradient(135deg, #059669, #10B981);
    }

    div[data-testid="stMetricValue"] {
        font-size: 4rem;
        font-weight: 900;
        color: #ffffff;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.7);
    }

    /* --- System Status (Sidebar) --- */
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); } 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); } }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    [data-testid="stSidebar"] div[data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }
    [data-testid="stSidebar"] div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
    }
    [data-testid="stSidebar"] div[data-testid="stMetric"]:nth-of-type(2) {
        border: 2px solid #10b981;
        box-shadow: 0 0 20px #10b981;
        animation: pulse 2s infinite;
    }

    /* --- Key Feature Pills & Overview Card --- */
    .overview-card, .features-card {
        background: rgba(139, 92, 246, 0.1);
        border: 2px solid #8B5CF6;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 3rem;
    }
    .features-card ul {
        display: flex; flex-wrap: wrap; justify-content: center; gap: 1rem; list-style-type: none; padding: 0;
    }
    .features-card li {
        background: linear-gradient(135deg, #8B5CF6, #6D28D9);
        padding: 0.75rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .features-card li:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.7);
    }
    
    /* --- General Layout --- */
    hr {
        background: linear-gradient(to right, transparent, #8B5CF6, transparent);
        border: none; height: 1px; margin: 3rem 0;
    }

    /* --- Data Quality Analysis Table --- */
    .data-quality-table table {
        width: 100%;
        font-size: 0.9rem;
    }
    .data-quality-table table td, .data-quality-table table th {
        padding: 0.4rem 0.5rem;
    }

    /* --- Footer --- */
    .footer { text-align: center; padding: 2rem; border-top: 1px solid rgba(139, 92, 246, 0.3); }
    .footer p { margin: 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline_data():
    """Load generated pipeline data"""
    notebooks_dir = Path(__file__).parent.resolve()
    project_root = notebooks_dir.parent.resolve()
    generated_dir = project_root / 'generated'
    data = {'embeddings': None, 'narratives': None, 'combined_data': None, 'labels': None}
    
    embeddings_file = generated_dir / 'fraud_embeddings.pkl'
    if embeddings_file.exists():
        try:
            with open(embeddings_file, 'rb') as f:
                embeddings_data = pickle.load(f)
                data.update(embeddings_data)
        except Exception as e: 
            st.error(f"Error loading embeddings: {e}")
    
    data_file = generated_dir / 'fraud_data_combined_clean.csv'
    if data_file.exists():
        try: 
            data['combined_data'] = pd.read_csv(data_file, nrows=1000)
        except Exception as e: 
            st.error(f"Error loading data: {e}")
    return data


def fix_plotly_dtypes(df):
    """Convert pandas nullable dtypes to standard NumPy types for Plotly compatibility"""
    df_fixed = df.copy()
    for col in df_fixed.columns:
        if pd.api.types.is_integer_dtype(df_fixed[col]):
            if df_fixed[col].isnull().any():
                df_fixed[col] = df_fixed[col].astype(np.float64)
            else:
                df_fixed[col] = df_fixed[col].astype(np.int64)
        elif pd.api.types.is_float_dtype(df_fixed[col]):
            df_fixed[col] = df_fixed[col].astype(np.float64)
        elif pd.api.types.is_bool_dtype(df_fixed[col]):
            df_fixed[col] = df_fixed[col].astype(bool)
    return df_fixed


def show_home():
    """Home page"""
    st.title("🔍 GenAI-Powered Fraud Detection System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="overview-card">
                <h3>📊 System Overview</h3>
                <p>This advanced fraud detection system uses:</p>
                <ul>
                    <li><strong>DistilBERT</strong> for narrative embeddings</li>
                    <li><strong>GPT-2 LoRA</strong> for pattern generation</li>
                    <li><strong>RTX 3050</strong> GPU acceleration</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="features-card">
                <h3>Key Features</h3>
                <ul>
                    <li>🛡️ Real-time fraud scoring</li>
                    <li>⚡ Narrative analysis</li>
                    <li>🎯 Transaction monitoring</li>
                    <li>📊 Pattern detection</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


# Replace the ENTIRE `show_data_analysis()` function with this fixed version:

def show_data_analysis():
    """Data analysis page with visualizations"""
    st.header("📈 Data Analysis Dashboard")
    
    with st.spinner("Loading and analyzing data..."):
        pipeline_data = load_pipeline_data()
    
    if pipeline_data['combined_data'] is not None:
        import plotly.express as px
        import plotly.graph_objects as go
        
        df = fix_plotly_dtypes(pipeline_data['combined_data'])
        
        # --- DATASET OVERVIEW ---
        st.subheader("📝 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.divider()
        
        # --- TRANSACTION AMOUNT DISTRIBUTION ---
        st.subheader("💰 Transaction Amount Distribution")
        col_viz, col_table = st.columns([2, 1])
        
        with col_viz:
            try:
                numeric_cols = df.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns.tolist()
                if numeric_cols:
                    amount_col = [c for c in numeric_cols if 'amount' in c.lower() or 'value' in c.lower()]
                    if not amount_col:
                        amount_col = [numeric_cols[0]]
                    
                    fig = px.histogram(df, x=amount_col[0], nbins=30, 
                                     title="Transaction Amount Histogram",
                                     labels={amount_col[0]: "Amount ($)"},
                                     color_discrete_sequence=['#1E40AF'])
                    fig.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#E2E8F0")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create histogram: {e}")
        
        with col_table:
            st.markdown("<h4 style='color:#E2E8F0;'>Statistics</h4>", unsafe_allow_html=True)
            numeric_cols = df.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns.tolist()
            if numeric_cols:
                amount_col = [c for c in numeric_cols if 'amount' in c.lower() or 'value' in c.lower()]
                if not amount_col:
                    amount_col = [numeric_cols[0]]
                
                stats_data = {
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"${df[amount_col[0]].mean():.2f}",
                        f"${df[amount_col[0]].median():.2f}",
                        f"${df[amount_col[0]].std():.2f}",
                        f"${df[amount_col[0]].min():.2f}",
                        f"${df[amount_col[0]].max():.2f}"
                    ]
                }
                st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)
        
        st.divider()
        
        # --- FRAUD DISTRIBUTION ---
        # REPLACE Fraud Distribution with DUAL charts:
        st.subheader("⚠️ Transaction & Fraud Analysis")
        col1, col2 = st.columns(2)

        with col1:
            # LEFT: Payment Types (keep your colorful pie)
            if 'payment_type' in df.columns:
                pt_counts = df['payment_type'].value_counts()
                fig_pt = px.pie(values=pt_counts.values, names=pt_counts.index,
                            title="Payment Types", color_discrete_sequence=['#F59E0B','#3B82F6','#10B981','#8B5CF6'])
                fig_pt.update_layout(height=350, font_size=12)
                st.plotly_chart(fig_pt, use_container_width=True)

        with col2:
            # RIGHT: ACTUAL FRAUD (your goal)
            if 'fraud_bool' in df.columns:
                fraud_counts = df['fraud_bool'].value_counts()
                labels = ['Fraud' if x == 1 else 'Legitimate' for x in fraud_counts.index]
                fig_fraud = px.pie(values=fraud_counts.values, names=labels,
                                title="Fraud Rate", color_discrete_sequence=['#DC2626','#059669'])
                fig_fraud.update_layout(height=350, font_size=12)
                st.plotly_chart(fig_fraud, use_container_width=True)

        # BELOW: Key metrics table
        if 'fraud_bool' in df.columns:
            fraud_rate = df['fraud_bool'].mean() * 100
            st.metric("🚨 CRITICAL: Fraud Rate", f"{fraud_rate:.2f}%", "1.3%")

        
        # --- FIXED COLUMN STATISTICS & DATA TYPES ---
        st.subheader("📊 Column Statistics & Data Types")
        col_table, col_viz = st.columns([1, 2])
        
        with col_table:
            # FIXED EXPANDER - No HTML, plain text only
            with st.expander("🔍 Summary Statistics", expanded=False):
                stats_df = df.describe(include='all').fillna(0).round(4)
                st.dataframe(stats_df, use_container_width=True)
        
        with col_viz:
            st.markdown("<h4 style='color:#E2E8F0;'>Data Type Distribution</h4>", unsafe_allow_html=True)
            try:
                # ULTRA-SAFE: Convert ALL dtypes to strings FIRST
                dtype_list = df.dtypes.astype(str).tolist()
                dtype_counts = pd.Series(dtype_list).value_counts()
                
                # Pure Python lists - NO pandas objects for Plotly
                data_types = dtype_counts.index.tolist()
                counts = dtype_counts.values.tolist()
                
                fig = px.bar(
                    x=data_types,
                    y=counts,
                    labels={'x': 'Data Type', 'y': 'Number of Columns'},
                    color=counts,
                    color_continuous_scale='Teal',
                    title="Column Data Types Distribution"
                )
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    font=dict(size=14, color="#E2E8F0"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {str(e)}")
                # SAFE FALLBACK
                st.write("**Data Type Breakdown:**")
                for dtype, count in pd.Series(df.dtypes.astype(str).tolist()).value_counts().items():
                    st.write(f"• **{dtype}**: {count} columns")
        
        st.divider()
        
        # --- DATA SAMPLE ---
        st.subheader("👀 Data Sample")
        cols_to_display = st.multiselect(
            "Display columns", df.columns.tolist(), default=df.columns.tolist()
        )
        search_term = st.text_input("Search table...")
        
        df_display = df[cols_to_display]
        if search_term:
            df_display = df_display[
                df_display.astype(str).apply(lambda row: row.str.contains(search_term, case=False).any(), axis=1)
            ]
        st.dataframe(df_display, use_container_width=True, height=400)
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv().encode('utf-8')
        
        csv = convert_df_to_csv(df_display)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='filtered_data.csv',
            mime='text/csv',
        )
        
        st.divider()
        
        # --- FIXED DATA QUALITY (NO Int64DType ERROR) ---
        st.subheader("🔍 Data Quality Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h4 style='color:#E2E8F0;'>Missing Values (%)</h4>", unsafe_allow_html=True)
            missing_df = df.isnull().sum().reset_index()
            missing_df.columns = ['Column', 'Missing Count']
            missing_df['Missing %'] = (missing_df['Missing Count'] / len(df)) * 100
            
            def format_missing_percentage(value):
                if value == 0:
                    icon, color = "✅", "#059669"
                elif value < 5:
                    icon, color = "⚠️", "#F59E0B"
                else:
                    icon, color = "❌", "#DC2626"
                progress_bar = f'<div style="background:#334155;border-radius:5px;height:15px;width:100%;"><div style="background:{color};width:{value}%;height:100%;border-radius:5px;text-align:center;color:white;font-weight:bold;line-height:15px;font-size:0.8rem;">{value:.1f}%</div></div>'
                return f"{icon} {progress_bar}"
            
            missing_df['Status'] = missing_df['Missing %'].apply(format_missing_percentage)
            st.markdown(f'<div class="data-quality-table">{missing_df[["Column", "Status"]].to_html(escape=False, index=False)}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("<h4 style='color:#E2E8F0;'>Data Type Composition</h4>", unsafe_allow_html=True)
            try:
                # ULTRA-SAFE PIE CHART: Pure Python lists
                dtype_list = df.dtypes.astype(str).tolist()
                dtype_counts = pd.Series(dtype_list).value_counts()
                
                fig = go.Figure(data=[go.Pie(
                    labels=dtype_counts.index.tolist(),
                    values=dtype_counts.values.tolist(),
                    hole=0.4,
                    marker_colors=px.colors.sequential.Tealgrn
                )])
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    title="Data Type Distribution",
                    font=dict(size=14, color="#E2E8F0"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create pie chart: {str(e)}")
                # SAFE TABLE FALLBACK
                dtype_summary = pd.Series(df.dtypes.astype(str).tolist()).value_counts().reset_index()
                dtype_summary.columns = ['Data Type', 'Count']
                st.dataframe(dtype_summary, hide_index=True)
                
    else:
        st.warning("⚠️ No data available - pipeline may not have completed")




def show_embeddings():
    st.header("🧠 Model Embeddings")

    with st.spinner("Loading embeddings..."):
        pipeline_data = load_pipeline_data()

    if pipeline_data['embeddings'] is not None and pipeline_data['labels'] is not None and pipeline_data['narratives'] is not None:
        embeddings = pipeline_data['embeddings']
        labels = np.array(pipeline_data['labels'])
        # Ensure narratives are in a simple list or array format
        narratives = np.array(pipeline_data['narratives']).flatten()

        # --- METRICS ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Embedding Shape", f"{embeddings.shape}")
        with col2:
            st.metric("Embedding Dimension", embeddings.shape[1])
        with col3:
            st.metric("Total Narratives", embeddings.shape[0])

        st.subheader("Embedding Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{embeddings.mean():.4f}")
        with col2:
            st.metric("Std Dev", f"{embeddings.std():.4f}")
        with col3:
            st.metric("Min", f"{embeddings.min():.4f}")
        with col4:
            st.metric("Max", f"{embeddings.max():.4f}")
            
        st.divider()

        # --- VISUALIZATION CONTROLS ---
        st.subheader("Principal Component Analysis (PCA) Projection")
        
        # Performance sampling option
        use_sampling = st.checkbox("Sample 1000 points for performance", value=True)
        
        # View selection
        view_mode = st.radio("Select View", ['2D View', '3D View'], index=1, horizontal=True)

        try:
            from sklearn.decomposition import PCA
            import plotly.express as px
            
            # --- DATA SAMPLING ---
            if use_sampling and embeddings.shape[0] > 1000:
                st.info("Displaying a random sample of 1000 data points.")
                sample_indices = np.random.choice(embeddings.shape[0], 1000, replace=False)
                embeddings_subset = embeddings[sample_indices]
                labels_subset = labels[sample_indices]
                narratives_subset = narratives[sample_indices]
            else:
                embeddings_subset = embeddings
                labels_subset = labels
                narratives_subset = narratives

            # --- PCA COMPUTATION ---
            n_components = 3 if view_mode == '3D View' else 2
            with st.spinner(f"Computing {n_components}D PCA projection..."):
                pca = PCA(n_components=n_components)
                projected_embeddings = pca.fit_transform(embeddings_subset)

            # --- PLOT DATAFRAME CREATION ---
            df_plot = pd.DataFrame()
            df_plot['PC1'] = projected_embeddings[:, 0]
            df_plot['PC2'] = projected_embeddings[:, 1]
            if n_components == 3:
                df_plot['PC3'] = projected_embeddings[:, 2]

            df_plot['Type'] = ['Fraud' if l == 1 else 'Legitimate' for l in labels_subset]
            df_plot['narrative'] = narratives_subset

            # --- PLOTTING ---
            fig = None
            color_map = {'Fraud': '#DC2626', 'Legitimate': '#059669'}
            
            if view_mode == '3D View':
                title_3d = f'3D PCA Projection (Total Variance: {pca.explained_variance_ratio_.sum():.2%})'
                labels_3d = {
                    'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                    'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
                    'PC3': f'PC3 ({pca.explained_variance_ratio_[2]:.2%})'
                }
                fig = px.scatter_3d(
                    df_plot,
                    x='PC1', y='PC2', z='PC3',
                    color='Type',
                    hover_data={'narrative': True, 'PC1': False, 'PC2': False, 'PC3': False},
                    title=title_3d,
                    labels=labels_3d,
                    color_discrete_map=color_map,
                    height=700
                )
                fig.update_traces(marker=dict(size=4, opacity=0.8))
            
            else: # 2D View
                title_2d = f'2D PCA Projection (Total Variance: {pca.explained_variance_ratio_.sum():.2%})'
                labels_2d = {
                    'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                    'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'
                }
                fig = px.scatter(
                    df_plot,
                    x='PC1', y='PC2',
                    color='Type',
                    hover_data={'narrative': True, 'PC1': False, 'PC2': False},
                    title=title_2d,
                    labels=labels_2d,
                    color_discrete_map=color_map,
                    height=600
                )
            
            if fig:
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    font_color="#E2E8F0",
                    legend_title_text=''
                )
                st.plotly_chart(fig, use_container_width=True)

            # --- VARIANCE STATS ---
            st.subheader("Explained Variance")
            for i, ratio in enumerate(pca.explained_variance_ratio_):
                st.write(f"**Variance Explained by PC{i+1}:** {ratio:.2%}")
            st.write(f"**Total Variance Explained:** {pca.explained_variance_ratio_.sum():.2%}")

        except ImportError as e:
            st.warning(f"Visualization libraries not available: {e}. Please run `pip install scikit-learn`.")
        except Exception as e:
            st.error(f"Error creating visualization: {e}")

    else:
        st.warning("⚠️ No embeddings, labels, or narratives available - pipeline may not have completed")


def show_model_info():
    """Model information"""
    st.header("ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("DistilBERT Embedding Model")
        st.write("""
        - **Architecture**: DistilBERT (distilbert-base-uncased)
        - **Input**: Fraud narratives (max 256 tokens)
        - **Output**: 768-dimensional embeddings
        - **Training**: 3 epochs on RTX 3050 GPU
        - **Optimization**: GPU-accelerated training
        
        **Performance Metrics**:
        - Epoch 1 Avg Loss: 0.0217
        - Epoch 2 Avg Loss: 0.0001
        - Epoch 3 Avg Loss: 0.0001
        """)
    
    with col2:
        st.subheader("GPT-2 LoRA Fine-tuning")
        st.write("""
        - **Architecture**: GPT-2 with LoRA adapters
        - **LoRA Rank**: 8
        - **LoRA Alpha**: 32
        - **Target Modules**: c_attn (attention layers)
        - **Training**: 3 epochs on RTX 3050 GPU
        
        **Performance Metrics**:
        - Epoch 1 Avg Loss: 1.1946
        - Epoch 2 Avg Loss: 0.1380
        - Epoch 3 Avg Loss: 0.0954
        """)
    
    st.subheader("🖥️ Hardware Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("GPU", "RTX 3050 Laptop")
    with col2:
        st.metric("VRAM", "4.29 GB")
    with col3:
        st.metric("PyTorch", "2.6.0+cu124")


def show_pipeline_summary():
    """Pipeline execution summary"""
    st.header("📋 Pipeline Execution Summary")
    
    pipeline_info = {
        "Step": ["1: PII Cleaning", "2: Narrative Generation", "3: Embedding Model", "4: GPT-2 LoRA"],
        "Status": ["✅ SUCCESS", "✅ SUCCESS", "✅ SUCCESS", "✅ SUCCESS"],
        "Duration": ["~15 sec", "~2 sec", "~30 min", "~50 min"],
        "GPU": ["CPU", "CPU", "✅ GPU", "✅ GPU"]
    }
    
    df_pipeline = pd.DataFrame(pipeline_info)
    st.dataframe(df_pipeline, use_container_width=True)
    
    st.metric("Total Execution Time", "1 hour 20 minutes 52 seconds")
    
    st.subheader("Output Files Generated")
    output_files = {
        "File": [
            "fraud_data_combined_clean.csv",
            "fraud_narratives_combined.csv",
            "fraud_embeddings.pkl",
            "fraud_embedding_model.pt",
            "embedding_tokenizer/",
            "fraud_pattern_generator_lora/",
            "gpt2_tokenizer/"
        ],
        "Size": ["65.43 MB", "888 KB", "9.19 MB", "~100 MB", "~1 MB", "~10 MB", "~1 MB"],
        "Type": ["Data", "Data", "Embeddings", "Model", "Tokenizer", "Model", "Tokenizer"]
    }
    
    df_outputs = pd.DataFrame(output_files)
    st.dataframe(df_outputs, use_container_width=True)


# Main app
def main():
    with st.sidebar:
        st.sidebar.header("Navigation")
        page = st.sidebar.radio(
            "Go to",
            ["🏠 Home", "📊 Data Analysis", "🧠 Embeddings", "ℹ️ Model Info", "📋 Pipeline Summary"],
            label_visibility="collapsed"
        )
        st.sidebar.divider()
        st.sidebar.subheader("System Status")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("GPU", "RTX 3050")
        with c2:
            st.metric("Status", "Active")
    
    # Main content metrics first
    st.header("🚀 Quick Stats")
    pipeline_data = load_pipeline_data()
    labels = np.array(pipeline_data.get('labels', []))
    fraud_count = np.sum(labels)
    total = len(labels) if len(labels) > 0 else 1
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Narratives", total)
    with m2:
        st.metric("Fraud Cases", fraud_count, f"{(fraud_count/total*100):.1f}%")
    with m3:
        st.metric("Legitimate", total - fraud_count, f"{((total-fraud_count)/total*100):.1f}%")

    st.divider()
    
    page_functions = {
        "🏠 Home": show_home,
        "📊 Data Analysis": show_data_analysis,
        "🧠 Embeddings": show_embeddings,
        "ℹ️ Model Info": show_model_info,
        "📋 Pipeline Summary": show_pipeline_summary
    }
    page_functions[page]()
    
    # --- FOOTER ---
    st.markdown("""
    <div class="footer">
        <p>🐍 Python | 🎈 Streamlit | 🔥 PyTorch | 🤗 HuggingFace</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
