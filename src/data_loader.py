import pandas as pd
import streamlit as st

def load_and_validate_data(uploaded_file):
    """
    Load and validate uploaded A/B test data
    
    Parameters:
    uploaded_file: Streamlit uploaded file object
    
    Returns:
    pandas.DataFrame: Cleaned and validated dataframe
    """
    try:
        # Load the data
        df = pd.read_csv(uploaded_file)
        
        # Clean column names - remove spaces and convert to lowercase
        df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
        
        # Remove unnecessary index columns (common in exported CSV files)
        columns_to_remove = ['unnamed:_0', 'unnamed: 0', 'unnamed_0', 'unnamed']
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(col, axis=1)
                st.info(f"Removed unnecessary index column: {col}")
        
        # Convert boolean conversion column to integer if needed
        if 'converted' in df.columns:
            if df['converted'].dtype == 'bool':
                df['converted'] = df['converted'].astype(int)
                st.info("Converted boolean 'converted' column to integer (0/1)")
        
        # Validate required columns
        required_columns = ['test_group', 'converted']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.info("Please ensure your CSV contains 'test_group' and 'converted' columns")
            return None
            
        st.success(f"Data validation successful: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_conversion_summary(df):
    """
    Calculate conversion summary statistics
    
    Parameters:
    df: pandas.DataFrame with test_group and converted columns
    
    Returns:
    pandas.DataFrame: Conversion summary statistics
    """
    if df is None or 'test_group' not in df.columns or 'converted' not in df.columns:
        return None
        
    try:
        summary = df.groupby('test_group').agg({
            'converted': ['count', 'sum', 'mean']
        })
        
        # Flatten column names
        summary.columns = ['user_count', 'conversions', 'conversion_rate']
        summary['conversion_percentage'] = (summary['conversion_rate'] * 100).round(2)
        
        return summary
        
    except Exception as e:
        st.error(f"Error calculating conversion summary: {str(e)}")
        return None