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

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline_data():
    """Load generated pipeline data"""
    notebooks_dir = Path(__file__).parent.resolve()
    project_root = notebooks_dir.parent.resolve()
    generated_dir = project_root / 'generated'
    
    data = {
        'embeddings': None,
        'narratives': None,
        'combined_data': None,
        'labels': None
    }
    
    # Load embeddings
    embeddings_file = generated_dir / 'fraud_embeddings.pkl'
    if embeddings_file.exists():
        try:
            with open(embeddings_file, 'rb') as f:
                embeddings_data = pickle.load(f)
                data['embeddings'] = embeddings_data.get('embeddings')
                data['narratives'] = embeddings_data.get('narratives')
                data['labels'] = embeddings_data.get('labels')
        except Exception as e:
            st.error(f"Error loading embeddings: {e}")
    
    # Load combined data
    data_file = generated_dir / 'fraud_data_combined_clean.csv'
    if data_file.exists():
        try:
            data['combined_data'] = pd.read_csv(data_file, nrows=1000)
        except Exception as e:
            st.error(f"Error loading data: {e}")
    
    return data

def show_home():
    """Home page"""
    st.title("🔍 GenAI-Powered Fraud Detection System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📊 System Overview")
        st.write("""
        This advanced fraud detection system uses:
        - **DistilBERT** for narrative embeddings
        - **GPT-2 LoRA** for pattern generation
        - **RTX 3050** GPU acceleration
        
        ### Key Features
        ✅ Real-time fraud scoring
        ✅ Narrative analysis
        ✅ Transaction monitoring
        ✅ Pattern detection
        """)
    
    with col2:
        st.header("🚀 Quick Stats")
        with st.spinner("Loading statistics..."):
            pipeline_data = load_pipeline_data()
            
            if pipeline_data['labels'] is not None:
                labels = np.array(pipeline_data['labels'])
                fraud_count = np.sum(labels)
                total = len(labels)
                
                st.metric("Total Narratives", total)
                st.metric("Fraud Cases", f"{fraud_count} ({fraud_count/total*100:.1f}%)")
                st.metric("Legitimate Cases", f"{total - fraud_count} ({(total-fraud_count)/total*100:.1f}%)")
            else:
                st.info("Loading statistics...")

def show_data_analysis():
    """Data analysis page"""
    st.header("📈 Data Analysis")
    
    with st.spinner("Loading data..."):
        pipeline_data = load_pipeline_data()
    
    if pipeline_data['combined_data'] is not None:
        df = pipeline_data['combined_data'].copy()
        
        # Convert all int64/int32 to standard Python int for PyArrow compatibility
        for col in df.columns:
            if df[col].dtype in ['int64', 'int32', 'int16', 'int8']:
                try:
                    df[col] = df[col].astype('float64').astype('int')
                except:
                    pass
        
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.subheader("Data Sample")
        st.dataframe(df.head(10), width='stretch')
        
        st.subheader("Column Statistics")
        stats_df = df.describe().astype(float)
        st.dataframe(stats_df, width='stretch')
        
        st.subheader("Data Types")
        st.write(df.dtypes)
    else:
        st.warning("⚠️ No data available - pipeline may not have completed")

def show_embeddings():
    """Embeddings visualization"""
    st.header("🧠 Model Embeddings")
    
    with st.spinner("Loading embeddings..."):
        pipeline_data = load_pipeline_data()
    
    if pipeline_data['embeddings'] is not None:
        embeddings = pipeline_data['embeddings']
        
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
        
        # PCA visualization
        st.subheader("2D Projection (PCA)")
        try:
            from sklearn.decomposition import PCA
            import plotly.express as px
            
            with st.spinner("Computing PCA projection..."):
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(embeddings)
            
            # Create DataFrame for plotting
            df_plot = pd.DataFrame({
                'PC1': embeddings_2d[:, 0],
                'PC2': embeddings_2d[:, 1],
            })
            
            # Add labels if available
            if pipeline_data['labels'] is not None:
                labels = np.array(pipeline_data['labels'])
                df_plot['Type'] = ['Fraud' if l == 1 else 'Legitimate' for l in labels]
                
                fig = px.scatter(
                    df_plot,
                    x='PC1',
                    y='PC2',
                    color='Type',
                    title=f'PCA Projection (Variance explained: {pca.explained_variance_ratio_.sum():.2%})',
                    labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', 
                            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'},
                    height=600
                )
            else:
                # No labels, just plot all points
                fig = px.scatter(
                    df_plot,
                    x='PC1',
                    y='PC2',
                    title=f'PCA Projection (Variance explained: {pca.explained_variance_ratio_.sum():.2%})',
                    labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', 
                            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'},
                    height=600
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show variance explained
            st.write(f"**Variance Explained by PC1:** {pca.explained_variance_ratio_[0]:.2%}")
            st.write(f"**Variance Explained by PC2:** {pca.explained_variance_ratio_[1]:.2%}")
            st.write(f"**Total Variance Explained:** {pca.explained_variance_ratio_.sum():.2%}")
                
        except ImportError as e:
            st.warning(f"Visualization libraries not available: {e}")
        except Exception as e:
            st.error(f"Error creating visualization: {e}")
            
    else:
        st.warning("⚠️ No embeddings available - pipeline may not have completed")

def show_model_info():
    """Model information"""
    st.header("🤖 Model Information")
    
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
    st.dataframe(df_pipeline, width=1200, hide_index=True)
    
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
    st.dataframe(df_outputs, width=1200, hide_index=True)

# Main app
def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Section",
        ["Home", "Data Analysis", "Embeddings", "Model Info", "Pipeline Summary"]
    )
    
    st.sidebar.divider()
    
    st.sidebar.subheader("System Status")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.sidebar.metric("GPU", "RTX 3050")
    with col2:
        st.sidebar.metric("Status", "🟢 Active")
    
    # Render selected page
    if page == "Home":
        show_home()
    elif page == "Data Analysis":
        show_data_analysis()
    elif page == "Embeddings":
        show_embeddings()
    elif page == "Model Info":
        show_model_info()
    elif page == "Pipeline Summary":
        show_pipeline_summary()
    
    # Footer
    st.divider()
    st.caption("GenAI Fraud Detection System | Powered by DistilBERT + GPT-2 LoRA | RTX 3050 GPU Optimized")

if __name__ == "__main__":
    main()
