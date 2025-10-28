import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_and_validate_data, get_conversion_summary
from src.statistical_tests import perform_ab_test
from src.visualization import plot_conversion_rates, plot_user_distribution
from src.power_analysis import calculate_sample_size, calculate_statistical_power

# Page configuration
st.set_page_config(
    page_title="A/B Testing Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for data persistence
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

# Main title and description
st.title("A/B Testing Analysis Dashboard")
st.markdown("**Upload your A/B test data to analyze statistical significance and receive data-driven business recommendations**")

# Sidebar navigation
st.sidebar.title("Navigation Panel")
st.sidebar.markdown("Configure your analysis settings and access test data")

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Upload", 
    "Statistical Analysis", 
    "Data Visualizations", 
    "Test Planning", 
    "Machine Learning"
])

# Tab 1: Data Upload and Validation
with tab1:
    st.header("Data Upload and Validation")
    
    uploaded_file = st.file_uploader(
        "Upload your A/B test data (CSV format)", 
        type=['csv'],
        help="Required columns: test_group and converted"
    )
    
    if uploaded_file is not None:
        df = load_and_validate_data(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            st.success(f"Data loaded successfully: {df.shape[0]:,} rows, {df.shape[1]} columns")
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Data structure information
            st.subheader("Data Structure Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Column Names:**")
                st.write(df.columns.tolist())
            with col2:
                st.write("**Data Types:**")
                st.write(df.dtypes.astype(str))

# Tab 2: Statistical Analysis Results
with tab2:
    st.header("Statistical Analysis Results")
    
    if st.session_state.data_loaded:
        df = st.session_state.df
        conversion_summary = get_conversion_summary(df)
        
        if conversion_summary is not None and len(conversion_summary) >= 2:
            # Display conversion metrics
            st.subheader("Conversion Performance Metrics")
            col1, col2, col3 = st.columns(3)
            
            groups = conversion_summary.index.tolist()
            group_a, group_b = groups[0], groups[1]
            
            with col1:
                st.metric(
                    f"Group {group_a} Conversion Rate",
                    f"{conversion_summary.loc[group_a, 'conversion_rate']:.3%}",
                    help=f"{conversion_summary.loc[group_a, 'conversions']:,} conversions out of {conversion_summary.loc[group_a, 'user_count']:,} users"
                )
            
            with col2:
                st.metric(
                    f"Group {group_b} Conversion Rate",
                    f"{conversion_summary.loc[group_b, 'conversion_rate']:.3%}",
                    help=f"{conversion_summary.loc[group_b, 'conversions']:,} conversions out of {conversion_summary.loc[group_b, 'user_count']:,} users"
                )
            
            with col3:
                diff = (conversion_summary.loc[group_b, 'conversion_rate'] - 
                       conversion_summary.loc[group_a, 'conversion_rate'])
                st.metric(
                    "Performance Difference",
                    f"{diff:.3%}",
                    delta=f"{diff:.3%}" if diff != 0 else None
                )
            
            # Statistical significance testing
            st.subheader("Statistical Significance Analysis")
            test_results = perform_ab_test(conversion_summary)
            
            if test_results:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Statistical Z-Score", f"{test_results['z_score']:.2f}")
                    st.metric("Statistical P-Value", f"{test_results['p_value']:.6f}")
                
                with col2:
                    significance = "Statistically Significant" if test_results['is_significant'] else "Not Statistically Significant"
                    significance_color = "normal" if test_results['is_significant'] else "off"
                    st.metric("Test Result", significance, delta=None, delta_color=significance_color)
                    
                    ci_lower, ci_upper = test_results['confidence_interval']
                    st.metric("95% Confidence Interval", 
                             f"[{ci_lower:.4f}, {ci_upper:.4f}]")
                
                # Business recommendation
                st.subheader("Business Implementation Recommendation")
                if test_results['is_significant'] and test_results['absolute_difference'] > 0:
                    st.success(f"**Implement Group {test_results['group_b']}** - Statistically significant improvement of {test_results['relative_improvement']:.1f}%")
                elif test_results['is_significant'] and test_results['absolute_difference'] < 0:
                    st.success(f"**Implement Group {test_results['group_a']}** - Statistically significant improvement detected")
                else:
                    st.warning("**No statistically significant winner identified** - consider extending test duration or increasing sample size")
                
                # PDF Export Section
                st.subheader("Export Report")
                
                if st.button("Generate PDF Report", type="primary"):
                    try:
                        from src.pdf_reporter import generate_ab_test_report
                        import tempfile
                        import os
                        
                        # Create temporary file for PDF
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            pdf_path = tmp_file.name
                        
                        # Generate PDF report
                        pdf_path = generate_ab_test_report(conversion_summary, test_results, pdf_path)
                        
                        # Read the PDF file for download
                        with open(pdf_path, "rb") as pdf_file:
                            pdf_bytes = pdf_file.read()
                        
                        # Create download button
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_bytes,
                            file_name="ab_test_analysis_report.pdf",
                            mime="application/pdf",
                            type="primary"
                        )
                        
                        # Clean up temporary file
                        os.unlink(pdf_path)
                        
                        st.success("PDF report generated successfully! Click the download button above to save the report.")
                        
                    except Exception as e:
                        st.error(f"Error generating PDF report: {str(e)}")
                        st.info("Make sure the pdf_reporter.py file exists in the src directory")
            
        else:
            st.error("Analysis requires at least two test groups for comparison")
    else:
        st.warning("Please upload your data in the Data Upload tab to view analysis results")

# Tab 3: Data Visualizations
with tab3:
    st.header("Data Visualizations")
    
    if st.session_state.data_loaded:
        df = st.session_state.df
        conversion_summary = get_conversion_summary(df)
        
        if conversion_summary is not None and len(conversion_summary) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Conversion Rate Comparison")
                fig1 = plot_conversion_rates(conversion_summary)
                st.pyplot(fig1)
            
            with col2:
                st.subheader("User Distribution Across Test Groups")
                fig2 = plot_user_distribution(conversion_summary)
                st.pyplot(fig2)
        else:
            st.error("Visualizations require at least two test groups for comparison")
    else:
        st.warning("Please upload your data in the Data Upload tab to view visualizations")

# Tab 4: Test Planning
with tab4:
    st.header("A/B Test Planning Calculator")
    st.markdown("Determine how many users you need for your A/B test to achieve statistically significant results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        baseline_rate = st.number_input(
            "Baseline Conversion Rate (%)",
            min_value=0.1,
            max_value=50.0,
            value=2.0,
            step=0.1,
            help="Current conversion rate before any changes"
        ) / 100
        
        min_effect = st.number_input(
            "Minimum Detectable Effect (%)",
            min_value=0.1,
            max_value=20.0,
            value=1.0,
            step=0.1,
            help="Minimum improvement you want to detect"
        ) / 100
    
    with col2:
        confidence_level = st.selectbox(
            "Confidence Level",
            [90, 95, 99],
            index=1,
            help="Statistical confidence level"
        )
        
        statistical_power = st.selectbox(
            "Statistical Power",
            [80, 85, 90, 95],
            index=0,
            help="Probability of detecting a true effect"
        )
    
    # Calculate sample size
    alpha = 1 - confidence_level/100
    power = statistical_power/100
    
    sample_size = calculate_sample_size(baseline_rate, min_effect, alpha, power)
    
    st.subheader("Sample Size Requirement")
    st.metric(
        "Required Sample Size Per Group",
        f"{sample_size:,} users",
        help=f"Each test group needs {sample_size:,} users to detect a {min_effect*100:.1f}% improvement with {confidence_level}% confidence and {statistical_power}% power"
    )
    
    # Power analysis for current test
    if st.session_state.data_loaded:
        st.subheader("Current Test Power Analysis")
        conversion_summary = get_conversion_summary(st.session_state.df)
        
        if conversion_summary is not None and len(conversion_summary) >= 2:
            achieved_power = calculate_statistical_power(conversion_summary)
            
            st.metric(
                "Achieved Statistical Power",
                f"{achieved_power:.1%}",
                help="Probability that your test correctly detected a true effect"
            )
            
            if achieved_power < 0.8:
                st.warning(f"Test underpowered ({achieved_power:.1%}). Consider increasing sample size for reliable results.")
            else:
                st.success(f"Test adequately powered ({achieved_power:.1%}) for reliable results")

# Tab 5: Machine Learning
with tab5:
    st.header("Machine Learning Insights")
    st.markdown("Use machine learning to predict conversion probability and understand key drivers")
    
    if st.session_state.data_loaded:
        df = st.session_state.df
        
        # Check if we have enough data for ML
        if len(df) < 100:
            st.warning("Need at least 100 records for reliable machine learning analysis")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                model_type = st.selectbox(
                    "Select Model Type",
                    ["Random Forest", "Logistic Regression"],
                    help="Random Forest: Better for complex patterns. Logistic Regression: More interpretable."
                )
            
            with col2:
                st.write("")
                st.write("")
                train_model = st.button("Train Prediction Model", type="primary")
            
            if train_model:
                with st.spinner("Training machine learning model..."):
                    try:
                        from src.ml_model import ConversionPredictor
                        
                        # Initialize and train model
                        predictor = ConversionPredictor()
                        model_type_lower = model_type.lower().replace(" ", "_")
                        results = predictor.train_model(df, model_type=model_type_lower)
                        
                        if results:
                            st.success("Model trained successfully!")
                            
                            # Display model performance
                            st.subheader("Model Performance")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("AUC Score", f"{results['auc_score']:.3f}")
                            
                            with col2:
                                st.metric("Training Samples", f"{results['train_size']:,}")
                            
                            with col3:
                                st.metric("Test Samples", f"{results['test_size']:,}")
                            
                            # Feature importance
                            st.subheader("Feature Importance")
                            importance_df = predictor.get_feature_importance_df(results['feature_importance'])
                            
                            # Display feature importance as bar chart
                            fig, ax = plt.subplots(figsize=(10, 6))
                            features = importance_df['feature']
                            importances = importance_df['importance']
                            
                            y_pos = np.arange(len(features))
                            bars = ax.barh(y_pos, importances, color='steelblue', alpha=0.7)
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(features)
                            ax.set_xlabel('Importance')
                            ax.set_title('Feature Importance for Conversion Prediction', fontweight='bold')
                            ax.grid(axis='x', alpha=0.3)
                            
                            # Add value labels on bars
                            for i, v in enumerate(importances):
                                ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')
                            
                            st.pyplot(fig)
                            
                            # Business insights
                            st.subheader("Key Insights")
                            top_feature = importance_df.iloc[0]
                            st.info(f"**Most Important Factor:** {top_feature['feature']} (importance: {top_feature['importance']:.3f})\n\n"
                                   f"This indicates that {top_feature['feature'].replace('_', ' ')} has the strongest influence on conversion predictions.")
                            
                            # Save predictor to session state
                            st.session_state.predictor = predictor
                            st.session_state.ml_results = results
                            
                        else:
                            st.error("Failed to train model. Check your data quality.")
                            
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
            
            # Prediction section
            if 'predictor' in st.session_state and st.session_state.predictor.is_trained:
                st.subheader("Conversion Predictions")
                
                if st.button("Predict Conversion Probabilities", type="primary"):
                    with st.spinner("Making predictions..."):
                        try:
                            probabilities = st.session_state.predictor.predict_conversion_probability(df)
                            
                            # Add predictions to dataframe
                            df_with_predictions = df.copy()
                            df_with_predictions['conversion_probability'] = probabilities
                            
                            # Show users with highest conversion probability
                            st.subheader("Top 10 Users Most Likely to Convert")
                            top_users = df_with_predictions.nlargest(10, 'conversion_probability')[
                                ['user_id', 'test_group', 'conversion_probability']
                            ]
                            st.dataframe(top_users.style.format({'conversion_probability': '{:.3f}'}))
                            
                            # Distribution of predictions
                            st.subheader("Prediction Distribution")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.hist(probabilities, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                            ax.set_xlabel('Conversion Probability')
                            ax.set_ylabel('Number of Users')
                            ax.set_title('Distribution of Predicted Conversion Probabilities', fontweight='bold')
                            ax.grid(alpha=0.3)
                            st.pyplot(fig)
                            
                            # Summary statistics
                            st.subheader("Prediction Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Average", f"{probabilities.mean():.3f}")
                            with col2:
                                st.metric("Maximum", f"{probabilities.max():.3f}")
                            with col3:
                                st.metric("Minimum", f"{probabilities.min():.3f}")
                            with col4:
                                st.metric("Standard Deviation", f"{probabilities.std():.3f}")
                            
                        except Exception as e:
                            st.error(f"Error making predictions: {str(e)}")
    
    else:
        st.warning("Please upload your data in the Data Upload tab to enable machine learning analysis")

# Sidebar functionality
st.sidebar.header("Test Data Options")
if st.sidebar.button("Load Sample Test Data", type="secondary"):
    sample_data = {
        'user_id': range(1000),
        'test_group': np.random.choice(['ad', 'psa'], 1000),
        'converted': np.random.choice([0, 1], 1000, p=[0.97, 0.03]),
        'total_ads': np.random.randint(1, 100, 1000)
    }
    sample_df = pd.DataFrame(sample_data)
    st.session_state.df = sample_df
    st.session_state.data_loaded = True
    st.rerun()

# System status
st.sidebar.header("System Status")
if st.session_state.data_loaded:
    st.sidebar.success("Data Successfully Loaded")
    st.sidebar.write(f"**Dataset Rows:** {st.session_state.df.shape[0]:,}")
    st.sidebar.write(f"**Dataset Columns:** {st.session_state.df.shape[1]}")
else:
    st.sidebar.info("Awaiting Data Upload")