import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ConversionPredictor:
    def __init__(self):
        self.model = None
        self.feature_encoder = {}
        self.feature_names = []
        self.is_trained = False
        
    def prepare_features(self, df):
        """
        Prepare features for machine learning model
        
        Parameters:
        df: pandas.DataFrame with A/B test data
        
        Returns:
        pandas.DataFrame: Features ready for model training
        """
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Feature engineering
        if 'total_ads' in df_processed.columns:
            df_processed['ad_engagement'] = df_processed['total_ads'].apply(
                lambda x: 'low' if x < 10 else 'medium' if x < 50 else 'high'
            )
        
        # Encode categorical variables
        categorical_columns = ['test_group', 'most_ads_day', 'ad_engagement']
        
        for col in categorical_columns:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.feature_encoder[col] = le
        
        # Select features for model
        feature_columns = []
        for col in ['test_group', 'total_ads', 'most_ads_hour']:
            if col in df_processed.columns:
                feature_columns.append(col)
        
        # Add encoded categorical features
        for col in ['most_ads_day', 'ad_engagement']:
            if col in df_processed.columns:
                feature_columns.append(col)
        
        self.feature_names = feature_columns
        
        return df_processed[feature_columns]
    
    def train_model(self, df, target_column='converted', model_type='random_forest'):
        """
        Train machine learning model to predict conversions
        
        Parameters:
        df: pandas.DataFrame with features and target
        target_column: Name of the target column
        model_type: Type of model to train ('random_forest' or 'logistic')
        
        Returns:
        dict: Model performance metrics
        """
        try:
            # Prepare features
            X = self.prepare_features(df)
            y = df[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Train model
            if model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            else:  # logistic regression
                self.model = LogisticRegression(random_state=42, max_iter=1000)
            
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            else:
                # For logistic regression, use absolute coefficients
                feature_importance = dict(zip(self.feature_names, np.abs(self.model.coef_[0])))
            
            self.is_trained = True
            
            return {
                'auc_score': auc_score,
                'feature_importance': feature_importance,
                'test_size': len(X_test),
                'train_size': len(X_train),
                'model_type': model_type
            }
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return None
    
    def predict_conversion_probability(self, df):
        """
        Predict conversion probability for each user
        
        Parameters:
        df: pandas.DataFrame with user features
        
        Returns:
        numpy.array: Conversion probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X = self.prepare_features(df)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return probabilities
    
    def get_feature_importance_df(self, feature_importance):
        """
        Convert feature importance to DataFrame for display
        
        Parameters:
        feature_importance: Dictionary of feature importances
        
        Returns:
        pandas.DataFrame: Sorted feature importance
        """
        importance_df = pd.DataFrame({
            'feature': list(feature_importance.keys()),
            'importance': list(feature_importance.values())
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        return importance_df