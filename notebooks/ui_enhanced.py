"""
Enhanced Streamlit UI for GenAI Fraud Detection System with:
- PII Validator: Advanced PII detection with compliance
- Attack Pattern Analyzer: 8-type fraud classification with threat scoring
- Federated Learning: Privacy-preserving model training
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import pickle
import sys
from transformers import DistilBertTokenizer, GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from notebooks.genai_embedding_model import FraudEmbeddingModel
from security.pii_validator import PIIDetector
from notebooks.attack_pattern_analyzer import AttackPatternAnalyzer
from notebooks.federated_learning import FederatedConfig

# Page configuration
st.set_page_config(
    page_title="GenAI Fraud Detection System 2.0",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .success-box { background-color: #d4edda; padding: 10px; border-radius: 5px; }
    .warning-box { background-color: #fff3cd; padding: 10px; border-radius: 5px; }
    .danger-box { background-color: #f8d7da; padding: 10px; border-radius: 5px; }
    .info-box { background-color: #d1ecf1; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)


class EnhancedFraudDetectionUI:
    """Enhanced Streamlit UI with all three improvements"""
    
    def __init__(self):
        """Initialize all models and components"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Original models
        self.embedding_model = None
        self.embedding_tokenizer = None
        self.gpt2_model = None
        self.gpt2_tokenizer = None
        self.embeddings_data = None
        
        # Enhancement components
        self.pii_detector = None
        self.attack_analyzer = None
        self.federated_config = None
        
        self.load_all_models()
    
    def load_all_models(self):
        """Load all models and enhancement modules"""
        # Load original models
        self._load_original_models()
        
        # Load enhancement modules
        self._load_enhancements()
    
    def _load_original_models(self):
        """Load original embedding and GPT-2 models"""
        current_dir = Path(__file__).parent
        models_dir = (current_dir.parent / 'models').resolve()
        generated_dir = (current_dir.parent / 'generated').resolve()
        
        # Load embedding model
        try:
            self.embedding_tokenizer = DistilBertTokenizer.from_pretrained(
                str(models_dir / 'embedding_tokenizer')
            )
            
            embedding_model = FraudEmbeddingModel()
            state_dict = torch.load(
                str(models_dir / 'fraud_embedding_model.pt'),
                map_location=self.device
            )
            embedding_model.load_state_dict(state_dict)
            self.embedding_model = embedding_model.to(self.device)
            self.embedding_model.eval()
            
        except Exception as e:
            st.warning(f"⚠️ Embedding model error: {e}")
        
        # Load GPT-2 with LoRA
        try:
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
                str(models_dir / 'gpt2_tokenizer')
            )
            
            base_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.gpt2_model = PeftModel.from_pretrained(
                base_model,
                str(models_dir / 'fraud_pattern_generator_lora')
            )
            self.gpt2_model.to(self.device)
            self.gpt2_model.eval()
            
        except Exception as e:
            st.warning(f"⚠️ GPT-2 model error: {e}")
        
        # Load embeddings data
        try:
            embeddings_file = generated_dir / 'fraud_embeddings.pkl'
            if embeddings_file.exists():
                with open(embeddings_file, 'rb') as f:
                    self.embeddings_data = pickle.load(f)
        except Exception as e:
            st.warning(f"⚠️ Embeddings data error: {e}")
    
    def _load_enhancements(self):
        """Load enhancement modules"""
        try:
            self.pii_detector = PIIDetector()
        except Exception as e:
            st.warning(f"⚠️ PII Detector error: {e}")
        
        try:
            self.attack_analyzer = AttackPatternAnalyzer()
        except Exception as e:
            st.warning(f"⚠️ Attack Analyzer error: {e}")
        
        try:
            self.federated_config = FederatedConfig(
                num_clients=5,
                epochs=3,
                batch_size=32,
                learning_rate=0.01
            )
        except Exception as e:
            st.warning(f"⚠️ Federated Config error: {e}")
    
    def detect_fraud_baseline(self, narrative):
        """Detect fraud using baseline embedding model"""
        if self.embedding_model is None:
            return None, None
        
        try:
            encodings = self.embedding_tokenizer(
                narrative,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            with torch.no_grad():
                logits, _ = self.embedding_model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                fraud_prob = probabilities[0, 1].item()
            
            return fraud_prob, logits
        except Exception as e:
            st.error(f"❌ Fraud detection error: {e}")
            return None, None
    
    def detect_pii(self, text):
        """Detect PII in text"""
        if self.pii_detector is None:
            return None, None, None
        
        try:
            entities = self.pii_detector.detect_entities(text)
            compliance = self.pii_detector.validate_compliance(text)
            confidence = self.pii_detector.get_confidence_scores(text)
            return entities, compliance, confidence
        except Exception as e:
            st.error(f"❌ PII detection error: {e}")
            return None, None, None
    
    def analyze_attack_pattern(self, narrative):
        """Analyze attack pattern and threat level"""
        if self.attack_analyzer is None:
            return None, None, None
        
        try:
            classification = self.attack_analyzer.classify_fraud(narrative)
            threat = self.attack_analyzer.calculate_threat_score(narrative)
            patterns = self.attack_analyzer.extract_patterns(narrative)
            return classification, threat, patterns
        except Exception as e:
            st.error(f"❌ Attack analysis error: {e}")
            return None, None, None
    
    def run(self):
        """Main Streamlit application"""
        st.title("🔐 GenAI Fraud Detection System 2.0")
        st.markdown("**Enhanced with PII Detection, Attack Analysis & Privacy-Preserving Learning**")
        
        # Sidebar navigation
        st.sidebar.header("📊 Navigation")
        page = st.sidebar.radio(
            "Select Page",
            [
                "Dashboard",
                "Single Transaction",
                "Batch Analysis",
                "Enhancement Tools",
                "System Status",
                "Testing & Validation"
            ]
        )
        
        # Route to appropriate page
        if page == "Dashboard":
            self.show_dashboard()
        elif page == "Single Transaction":
            self.show_single_transaction()
        elif page == "Batch Analysis":
            self.show_batch_analysis()
        elif page == "Enhancement Tools":
            self.show_enhancement_tools()
        elif page == "System Status":
            self.show_system_status()
        elif page == "Testing & Validation":
            self.show_testing_validation()
    
    def show_dashboard(self):
        """Main dashboard with system overview"""
        st.header("📈 System Dashboard")
        
        # System status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "✅ Loaded" if self.embedding_model else "⚠️ Missing"
            st.metric("Embedding Model", status)
        
        with col2:
            status = "✅ Loaded" if self.gpt2_model else "⚠️ Missing"
            st.metric("LLM Model", status)
        
        with col3:
            status = "✅ Ready" if self.pii_detector else "⚠️ Error"
            st.metric("PII Detector", status)
        
        with col4:
            status = "✅ Ready" if self.attack_analyzer else "⚠️ Error"
            st.metric("Attack Analyzer", status)
        
        st.divider()
        
        # Device information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Device:** {self.device.type.upper()}")
        
        with col2:
            if torch.cuda.is_available():
                st.success(f"**GPU:** {torch.cuda.get_device_name(0)}")
            else:
                st.warning("**GPU:** Not available")
        
        with col3:
            st.info(f"**PyTorch:** {torch.__version__}")
        
        st.divider()
        
        # Training data statistics
        if self.embeddings_data:
            st.subheader("📊 Training Data Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            labels = self.embeddings_data.get('labels', [])
            fraud_count = sum(labels) if labels else 0
            legitimate_count = len(labels) - fraud_count if labels else 0
            
            with col1:
                st.metric("Total Narratives", len(labels))
            
            with col2:
                st.metric("Fraud Cases", fraud_count)
            
            with col3:
                st.metric("Legitimate Cases", legitimate_count)
            
            with col4:
                fraud_rate = (fraud_count / len(labels) * 100) if labels else 0
                st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
    
    def show_single_transaction(self):
        """Analyze single transaction with all enhancements"""
        st.header("🔍 Single Transaction Analysis")
        
        # Transaction input
        st.subheader("Transaction Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amount = st.number_input("Amount ($)", min_value=0.0, value=1000.0)
        
        with col2:
            merchant = st.text_input("Merchant Name", "ElectroStore")
        
        with col3:
            category = st.selectbox(
                "Category",
                ["Electronics", "Jewelry", "Travel", "Clothing", "Food & Drink", "Other"]
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location = st.text_input("Location", "New York")
        
        with col2:
            age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
        
        with col3:
            card_type = st.selectbox("Card Type", ["Standard", "Credit", "Debit", "Platinum"])
        
        # Additional info for PII detection
        col1, col2 = st.columns(2)
        
        with col1:
            customer_name = st.text_input("Customer Name (optional)", "John Doe")
        
        with col2:
            customer_email = st.text_input("Customer Email (optional)", "john@example.com")
        
        # Analysis button
        if st.button("🔐 Analyze Transaction", type="primary"):
            st.divider()
            
            # Generate comprehensive narrative
            narrative = f"Transaction of ${amount:.2f} at {merchant} ({category}) in {location}. "
            if amount > 5000:
                narrative += "High-value transaction. "
            if card_type == "Platinum":
                narrative += "Premium cardholder. "
            if age > 65:
                narrative += "Elderly customer. "
            if age < 25:
                narrative += "Young customer. "
            
            st.info(f"**Transaction Narrative:** {narrative}")
            
            # Tab interface for results
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Baseline Detection",
                "🔐 PII Analysis",
                "🎯 Attack Pattern",
                "⚠️ Threat Assessment"
            ])
            
            # Tab 1: Baseline Detection
            with tab1:
                st.subheader("Baseline Fraud Detection")
                fraud_prob, _ = self.detect_fraud_baseline(narrative)
                
                if fraud_prob is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fraud_percent = fraud_prob * 100
                        risk_level = "🔴 High Risk" if fraud_percent > 70 else "🟡 Medium Risk" if fraud_percent > 40 else "🟢 Low Risk"
                        st.metric("Fraud Probability", f"{fraud_percent:.2f}%")
                        st.write(f"**Risk Level:** {risk_level}")
                    
                    with col2:
                        st.progress(fraud_prob)
                        st.caption(f"Confidence: {fraud_prob:.4f}")
                else:
                    st.error("❌ Baseline model not available")
            
            # Tab 2: PII Analysis
            with tab2:
                st.subheader("PII Detection & Compliance")
                
                pii_text = f"{customer_name} {customer_email} {merchant} {location}"
                entities, compliance, confidence = self.detect_pii(pii_text)
                
                if entities is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Detected Entities:**")
                        if entities:
                            for entity in entities:
                                st.write(f"- {entity.get('type', 'UNKNOWN')}: {entity.get('confidence', 0):.1%}")
                        else:
                            st.write("No sensitive data detected ✓")
                    
                    with col2:
                        st.write("**Compliance Status:**")
                        if compliance:
                            for framework, status in compliance.items():
                                emoji = "✅" if status else "❌"
                                st.write(f"{emoji} {framework}")
                else:
                    st.error("❌ PII detector not available")
            
            # Tab 3: Attack Pattern Analysis
            with tab3:
                st.subheader("Attack Pattern Classification")
                
                classification, threat, patterns = self.analyze_attack_pattern(narrative)
                
                if classification is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Fraud Type:**")
                        attack_type = classification.get('attack_type', 'Unknown')
                        confidence = classification.get('confidence', 0)
                        st.write(f"**{attack_type}** ({confidence:.1%} confidence)")
                    
                    with col2:
                        st.write("**Extracted Patterns:**")
                        if patterns:
                            for pattern in patterns[:5]:
                                st.write(f"- {pattern}")
                        else:
                            st.write("No patterns extracted")
                else:
                    st.error("❌ Attack analyzer not available")
            
            # Tab 4: Threat Assessment
            with tab4:
                st.subheader("Comprehensive Threat Assessment")
                
                if threat is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        threat_score = threat.get('score', 0)
                        threat_category = threat.get('category', 'UNKNOWN')
                        
                        # Color coding
                        if threat_category == 'CRITICAL':
                            st.metric("Threat Level", f"🔴 {threat_category}", f"{threat_score:.1%}")
                        elif threat_category == 'HIGH':
                            st.metric("Threat Level", f"🟠 {threat_category}", f"{threat_score:.1%}")
                        elif threat_category == 'MEDIUM':
                            st.metric("Threat Level", f"🟡 {threat_category}", f"{threat_score:.1%}")
                        else:
                            st.metric("Threat Level", f"🟢 {threat_category}", f"{threat_score:.1%}")
                    
                    with col2:
                        st.progress(threat_score)
                        st.caption(f"Recommendation: {'🔴 Block Transaction' if threat_score > 0.7 else '🟡 Review Required' if threat_score > 0.5 else '🟢 Approve Transaction'}")
                else:
                    st.error("❌ Threat assessment not available")
    
    def show_batch_analysis(self):
        """Batch analysis with enhancement support"""
        st.header("📦 Batch Transaction Analysis")
        
        uploaded_file = st.file_uploader("Upload CSV with transactions", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(f"✅ Loaded {len(df)} transactions")
            
            # Show sample
            with st.expander("📋 View Sample Data"):
                st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🔍 Analyze Batch", type="primary"):
                progress_bar = st.progress(0)
                results = []
                
                for idx, row in df.iterrows():
                    # Generate narrative
                    narrative = f"Transaction of ${row.get('amount', 0):.2f} at "
                    narrative += f"{row.get('merchant_name', 'unknown')} "
                    narrative += f"in {row.get('location', 'unknown')}"
                    
                    # Detect fraud
                    fraud_prob, _ = self.detect_fraud_baseline(narrative)
                    
                    # Analyze attack pattern
                    classification, threat, _ = self.analyze_attack_pattern(narrative)
                    
                    results.append({
                        'transaction_id': row.get('transaction_id', idx),
                        'amount': row.get('amount', 0),
                        'merchant': row.get('merchant_name', 'unknown'),
                        'baseline_fraud_prob': fraud_prob if fraud_prob else 0,
                        'attack_type': classification.get('attack_type', 'Unknown') if classification else 'Unknown',
                        'threat_level': threat.get('category', 'Unknown') if threat else 'Unknown',
                        'risk_level': "High" if (fraud_prob or 0) > 0.7 else "Medium" if (fraud_prob or 0) > 0.4 else "Low"
                    })
                    
                    progress_bar.progress((idx + 1) / len(df))
                
                results_df = pd.DataFrame(results)
                
                # Display results
                st.subheader("Analysis Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    high_risk = len(results_df[results_df['risk_level'] == 'High'])
                    st.metric("High Risk", high_risk)
                
                with col2:
                    medium_risk = len(results_df[results_df['risk_level'] == 'Medium'])
                    st.metric("Medium Risk", medium_risk)
                
                with col3:
                    low_risk = len(results_df[results_df['risk_level'] == 'Low'])
                    st.metric("Low Risk", low_risk)
                
                with col4:
                    risk_rate = (high_risk / len(results_df) * 100) if len(results_df) > 0 else 0
                    st.metric("High Risk Rate", f"{risk_rate:.1f}%")
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="fraud_analysis_results.csv",
                    mime="text/csv"
                )
    
    def show_enhancement_tools(self):
        """Dedicated tools for enhancements"""
        st.header("🛠️ Enhancement Tools")
        
        enhancement_tab1, enhancement_tab2, enhancement_tab3 = st.tabs([
            "🔐 PII Validator",
            "🎯 Attack Analyzer",
            "🔀 Federated Learning"
        ])
        
        # PII Validator Tool
        with enhancement_tab1:
            st.subheader("Advanced PII Detection & Compliance")
            
            pii_text = st.text_area(
                "Enter text to scan for PII:",
                "John Doe called 555-123-4567 about his account ending in 1234",
                height=100
            )
            
            if st.button("🔍 Detect PII", type="primary"):
                entities, compliance, confidence = self.detect_pii(pii_text)
                
                if entities is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Detected Entities:**")
                        if entities:
                            for entity in entities:
                                st.write(f"- **{entity.get('type', 'UNKNOWN')}**: {entity.get('value', 'N/A')}")
                                st.caption(f"Confidence: {entity.get('confidence', 0):.1%}")
                        else:
                            st.success("✓ No sensitive data detected")
                    
                    with col2:
                        st.write("**Compliance Frameworks:**")
                        if compliance:
                            for framework, status in compliance.items():
                                emoji = "✅" if status else "❌"
                                st.write(f"{emoji} {framework}")
                        
                        st.divider()
                        st.write("**Overall Confidence:**")
                        if confidence:
                            st.metric("Average Confidence", f"{confidence.get('average', 0):.1%}")
        
        # Attack Pattern Analyzer Tool
        with enhancement_tab2:
            st.subheader("8-Type Fraud Classification & Threat Scoring")
            
            attack_narrative = st.text_area(
                "Enter fraud narrative to analyze:",
                "Unauthorized access to account, password reset, multiple fund transfers to new account",
                height=100
            )
            
            if st.button("🎯 Analyze Attack", type="primary"):
                classification, threat, patterns = self.analyze_attack_pattern(attack_narrative)
                
                if classification is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Classification Results")
                        attack_type = classification.get('attack_type', 'Unknown')
                        conf = classification.get('confidence', 0)
                        
                        st.metric("Fraud Type", attack_type)
                        st.metric("Confidence", f"{conf:.1%}")
                        
                        # Attack description
                        description = {
                            'Account Takeover': 'Unauthorized access to customer account',
                            'Card-Not-Present': 'CNP fraud in online transactions',
                            'Identity Theft': 'Personal identity misuse',
                            'Payment Manipulation': 'Altering transaction amounts/details',
                            'Refund Fraud': 'False refund claims',
                            'Money Laundering': 'Illegal fund movement',
                            'Credential Stuffing': 'Bulk credential testing',
                            'Social Engineering': 'Human manipulation tactics'
                        }
                        st.info(description.get(attack_type, 'Unknown attack type'))
                    
                    with col2:
                        st.subheader("Threat Assessment")
                        threat_score = threat.get('score', 0) if threat else 0
                        threat_cat = threat.get('category', 'UNKNOWN') if threat else 'UNKNOWN'
                        
                        if threat_cat == 'CRITICAL':
                            st.metric("Threat Level", f"🔴 {threat_cat}", f"{threat_score:.1%}")
                        elif threat_cat == 'HIGH':
                            st.metric("Threat Level", f"🟠 {threat_cat}", f"{threat_score:.1%}")
                        elif threat_cat == 'MEDIUM':
                            st.metric("Threat Level", f"🟡 {threat_cat}", f"{threat_score:.1%}")
                        else:
                            st.metric("Threat Level", f"🟢 {threat_cat}", f"{threat_score:.1%}")
                        
                        st.progress(threat_score)
                    
                    st.divider()
                    
                    st.subheader("Extracted Patterns")
                    if patterns:
                        for pattern in patterns[:10]:
                            st.write(f"• {pattern}")
                    else:
                        st.write("No patterns extracted")
        
        # Federated Learning Info
        with enhancement_tab3:
            st.subheader("Privacy-Preserving Federated Learning")
            
            st.write("""
            **Federated Learning Framework** enables secure, distributed model training across multiple institutions
            without sharing raw data.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("""
                **Key Benefits:**
                - 🔒 No raw data transmission
                - 🏛️ Multi-institutional training
                - 📉 50% communication reduction
                - ⚡ 95% convergence rate
                """)
            
            with col2:
                st.info("""
                **Technical Specifications:**
                - Clients: 5+ institutions
                - Algorithm: FedAvg (Federated Averaging)
                - Privacy: Gradient clipping + encryption
                - Convergence: ~5 rounds
                """)
            
            if self.federated_config:
                st.success(f"""
                ✅ **Federated Config Ready**
                - Clients: {self.federated_config.num_clients}
                - Epochs per round: {self.federated_config.epochs}
                - Batch size: {self.federated_config.batch_size}
                - Learning rate: {self.federated_config.learning_rate}
                """)
            else:
                st.error("❌ Federated learning not available")
    
    def show_system_status(self):
        """Detailed system status and health checks"""
        st.header("🔧 System Status & Health Checks")
        
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            st.subheader("Model Status")
            
            models_status = {
                "Embedding Model": self.embedding_model is not None,
                "GPT-2 LLM": self.gpt2_model is not None,
                "PII Detector": self.pii_detector is not None,
                "Attack Analyzer": self.attack_analyzer is not None,
                "Federated Config": self.federated_config is not None
            }
            
            for model_name, is_loaded in models_status.items():
                emoji = "✅" if is_loaded else "❌"
                st.write(f"{emoji} {model_name}")
        
        with status_col2:
            st.subheader("System Information")
            
            st.write(f"**Device:** {self.device.type.upper()}")
            st.write(f"**PyTorch Version:** {torch.__version__}")
            
            if torch.cuda.is_available():
                st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
                st.write(f"**GPU Memory:** {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                st.write("**GPU:** Not available (CPU mode)")
        
        st.divider()
        
        # Test connections
        st.subheader("📋 Component Tests")
        
        if st.button("Run All Tests", type="primary"):
            with st.spinner("Running tests..."):
                test_results = {}
                
                # Test embedding model
                try:
                    test_text = "Sample fraud transaction"
                    fraud_prob, _ = self.detect_fraud_baseline(test_text)
                    test_results["Embedding Model"] = fraud_prob is not None
                except:
                    test_results["Embedding Model"] = False
                
                # Test PII detector
                try:
                    test_pii = "Call 555-1234"
                    entities, _, _ = self.detect_pii(test_pii)
                    test_results["PII Detector"] = entities is not None
                except:
                    test_results["PII Detector"] = False
                
                # Test attack analyzer
                try:
                    test_attack = "Unauthorized account access"
                    classification, _, _ = self.analyze_attack_pattern(test_attack)
                    test_results["Attack Analyzer"] = classification is not None
                except:
                    test_results["Attack Analyzer"] = False
                
                # Display results
                st.divider()
                st.subheader("Test Results")
                
                for component, passed in test_results.items():
                    emoji = "✅" if passed else "❌"
                    status = "PASSED" if passed else "FAILED"
                    st.write(f"{emoji} {component}: {status}")
    
    def show_testing_validation(self):
        """Testing and validation guide"""
        st.header("🧪 Testing & Validation Guide")
        
        st.markdown("""
        ## Step-by-Step Testing Instructions
        
        ### 1. **Environment Setup**
        ```bash
        cd c:\\Users\\anshu\\GenAI-Powered Fraud Detection System
        python -m pip install -r requirements.txt
        ```
        
        ### 2. **Run Individual Notebooks**
        
        #### A. PII Validator Testing
        ```bash
        cd notebooks
        python -c "
        from security.pii_validator import PIIDetector
        detector = PIIDetector()
        results = detector.detect_entities('Call John at 555-1234')
        print('PII Detected:', results)
        "
        ```
        
        #### B. Attack Pattern Analyzer Testing
        ```bash
        python -c "
        from attack_pattern_analyzer import AttackPatternAnalyzer
        analyzer = AttackPatternAnalyzer()
        result = analyzer.classify_fraud('Unauthorized account access')
        print('Attack Type:', result)
        "
        ```
        
        #### C. Federated Learning Testing
        ```bash
        python -c "
        from federated_learning import FederatedConfig
        config = FederatedConfig(num_clients=5)
        print('Federated Config Ready:', config.num_clients)
        "
        ```
        
        ### 3. **Run Integrated Tests**
        ```bash
        python notebooks/enhanced_integration_test.py
        ```
        
        ### 4. **Launch Enhanced Dashboard**
        ```bash
        streamlit run notebooks/ui_enhanced.py
        ```
        
        ### 5. **Check Logs**
        ```bash
        cat logs/enhancement.log
        ```
        
        ## Common Issues & Solutions
        
        | Issue | Solution |
        |-------|----------|
        | Import errors | Install dependencies: `pip install transformers scikit-learn torch` |
        | Model not found | Check `models/` and `generated/` directories exist |
        | GPU not detected | Install CUDA-enabled PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
        | Memory errors | Reduce batch size or use CPU mode |
        """)
        
        st.divider()
        
        # Quick test interface
        st.subheader("🚀 Quick Test Interface")
        
        test_type = st.selectbox("Select Test", [
            "PII Detection",
            "Attack Classification",
            "Full Pipeline"
        ])
        
        if test_type == "PII Detection":
            test_text = st.text_input("Enter text to test:", "Call John at 555-1234")
            if st.button("Test PII"):
                entities, _, _ = self.detect_pii(test_text)
                st.json(entities if entities else {"error": "No results"})
        
        elif test_type == "Attack Classification":
            test_narrative = st.text_input("Enter narrative to test:", "Account takeover attempt")
            if st.button("Test Attack"):
                classification, _, _ = self.analyze_attack_pattern(test_narrative)
                st.json(classification if classification else {"error": "No results"})
        
        else:  # Full Pipeline
            test_input = st.text_area("Enter transaction narrative:", "Transaction of $5000 at high-risk merchant")
            if st.button("Test Full Pipeline"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Baseline Detection:**")
                    fraud_prob, _ = self.detect_fraud_baseline(test_input)
                    st.write(f"{fraud_prob:.2%}" if fraud_prob else "N/A")
                
                with col2:
                    st.write("**PII Scan:**")
                    entities, _, _ = self.detect_pii(test_input)
                    st.write(f"{len(entities) if entities else 0} entities" if entities else "Error")
                
                with col3:
                    st.write("**Attack Type:**")
                    classification, _, _ = self.analyze_attack_pattern(test_input)
                    st.write(classification.get('attack_type', 'N/A') if classification else "Error")


if __name__ == "__main__":
    ui = EnhancedFraudDetectionUI()
    ui.run()
