# src/DS_Hydroponics.ipynb (save as .py file)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle
import warnings
import logging
from datetime import datetime
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hydroponics_ml.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HydroponicsMLPipeline:
    """
    Comprehensive ML Pipeline for Hydroponic System Optimization
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.imputers = {}
        self.feature_importance = {}
        self.outlier_detector = None
        self.performance_metrics = {}
        
        # Suppress warnings
        warnings.filterwarnings('ignore')
        
        logger.info("Initializing Hydroponics ML Pipeline")
    
    def load_and_validate_data(self, filepath):
        """
        Load data with comprehensive validation and error handling
        """
        try:
            logger.info(f"Loading data from {filepath}")
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Dataset file not found: {filepath}")
            
            # Load data
            df = pd.read_csv(filepath)
            
            # Basic validation
            if df.empty:
                raise ValueError("Dataset is empty")
            
            if len(df) < 100:
                raise ValueError(f"Dataset too small: {len(df)} rows. Need at least 100 rows.")
            
            # Convert timestamp if exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            logger.info(f"Data loaded successfully: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def comprehensive_data_cleaning(self, df):
        """
        Advanced data cleaning with multiple strategies
        """
        try:
            logger.info("Starting comprehensive data cleaning")
            
            # Make a copy
            df_cleaned = df.copy()
            
            # Remove timestamp for processing (if exists)
            timestamp_col = None
            if 'timestamp' in df_cleaned.columns:
                timestamp_col = df_cleaned['timestamp']
                df_cleaned = df_cleaned.drop('timestamp', axis=1)
            
            # Identify numeric columns
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns.tolist()
            
            logger.info(f"Numeric columns: {len(numeric_cols)}")
            logger.info(f"Categorical columns: {len(categorical_cols)}")
            
            # Handle missing values
            missing_info = df_cleaned.isnull().sum()
            if missing_info.sum() > 0:
                logger.info(f"Missing values found: {missing_info[missing_info > 0]}")
                
                # Use KNN imputation for numeric columns
                if numeric_cols:
                    knn_imputer = KNNImputer(n_neighbors=5)
                    df_cleaned[numeric_cols] = knn_imputer.fit_transform(df_cleaned[numeric_cols])
                    self.imputers['knn'] = knn_imputer
                
                # Use mode imputation for categorical columns
                if categorical_cols:
                    mode_imputer = SimpleImputer(strategy='most_frequent')
                    df_cleaned[categorical_cols] = mode_imputer.fit_transform(df_cleaned[categorical_cols])
                    self.imputers['mode'] = mode_imputer
            
            # Outlier detection and handling
            if numeric_cols:
                logger.info("Detecting outliers")
                isolation_forest = IsolationForest(contamination=0.05, random_state=self.random_state)
                outlier_labels = isolation_forest.fit_predict(df_cleaned[numeric_cols])
                self.outlier_detector = isolation_forest
                
                # Cap outliers instead of removing them
                outlier_indices = np.where(outlier_labels == -1)[0]
                logger.info(f"Found {len(outlier_indices)} outliers ({len(outlier_indices)/len(df_cleaned)*100:.2f}%)")
                
                for col in numeric_cols:
                    Q1 = df_cleaned[col].quantile(0.25)
                    Q3 = df_cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)
            
            # Re-add timestamp if it existed
            if timestamp_col is not None:
                df_cleaned['timestamp'] = timestamp_col
            
            logger.info("Data cleaning completed successfully")
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {str(e)}")
            raise
    
    def feature_engineering(self, df):
        """
        Create additional features for better model performance
        """
        try:
            logger.info("Starting feature engineering")
            
            df_features = df.copy()
            
            if 'timestamp' in df_features.columns:
                # Time-based features
                df_features['hour'] = df_features['timestamp'].dt.hour
                df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
                df_features['month'] = df_features['timestamp'].dt.month
                df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
                
                # Cyclical features
                df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
                df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
                df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
                df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
            
            # Environmental feature interactions
            if all(col in df_features.columns for col in ['temperature', 'humidity']):
                df_features['temp_humidity_ratio'] = df_features['temperature'] / (df_features['humidity'] + 1)
                df_features['comfort_index'] = (df_features['temperature'] * 0.7 + df_features['humidity'] * 0.3) / 100
            
            # Nutrient ratios
            nutrient_cols = ['nitrogen_ppm', 'phosphorus_ppm', 'potassium_ppm']
            if all(col in df_features.columns for col in nutrient_cols):
                df_features['total_nutrients'] = df_features[nutrient_cols].sum(axis=1)
                df_features['npk_ratio'] = df_features['nitrogen_ppm'] / (df_features['phosphorus_ppm'] + df_features['potassium_ppm'] + 1)
            
            # pH and EC interaction
            if all(col in df_features.columns for col in ['ph_level', 'ec_level']):
                df_features['ph_ec_interaction'] = df_features['ph_level'] * df_features['ec_level']
                df_features['ph_optimal'] = ((df_features['ph_level'] >= 5.5) & (df_features['ph_level'] <= 6.5)).astype(int)
            
            # System efficiency indicators
            system_cols = ['pump_status', 'fan_status', 'heater_status']
            if all(col in df_features.columns for col in system_cols):
                df_features['system_active_count'] = df_features[system_cols].sum(axis=1)
                df_features['system_efficiency'] = df_features['system_active_count'] / len(system_cols)
            
            logger.info(f"Feature engineering completed. New shape: {df_features.shape}")
            return df_features
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def prepare_training_data(self, df, target_columns):
        """
        Prepare data for training with proper splits and scaling
        """
        try:
            logger.info("Preparing training data")
            
            # Separate features and targets
            feature_cols = [col for col in df.columns if col not in target_columns + ['timestamp']]
            X = df[feature_cols].copy()
            
            # Prepare targets dictionary
            y_dict = {}
            for target in target_columns:
                if target in df.columns:
                    y_dict[target] = df[target].copy()
                else:
                    logger.warning(f"Target column {target} not found in dataset")
            
            # Handle categorical variables (if any)
            categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if categorical_cols:
                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            
            logger.info(f"Final feature matrix shape: {X.shape}")
            logger.info(f"Target variables: {list(y_dict.keys())}")
            
            return X, y_dict
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def train_models(self, X, y_dict, test_size=0.2):
        """
        Train multiple models for each target variable
        """
        try:
            logger.info("Starting model training")
            
            # Split data
            X_train, X_test, indices_train, indices_test = train_test_split(
                X, X.index, test_size=test_size, random_state=self.random_state
            )
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['main'] = scaler
            
            # Define models
            model_configs = {
                'random_forest': {
                    'model': RandomForestRegressor(random_state=self.random_state),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingRegressor(random_state=self.random_state),
                    'params': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.05, 0.1],
                        'max_depth': [3, 5, 7]
                    }
                },
                'ridge': {
                    'model': Ridge(),
                    'params': {
                        'alpha': [0.1, 1.0, 10.0, 100.0]
                    }
                }
            }
            
            # Train models for each target
            for target_name, y_values in y_dict.items():
                logger.info(f"Training models for {target_name}")
                
                y_train = y_values.loc[indices_train]
                y_test = y_values.loc[indices_test]
                
                self.models[target_name] = {}
                self.performance_metrics[target_name] = {}
                
                for model_name, config in model_configs.items():
                    try:
                        logger.info(f"Training {model_name} for {target_name}")
                        
                        # Grid search for best parameters
                        grid_search = GridSearchCV(
                            config['model'],
                            config['params'],
                            cv=5,
                            scoring='r2',
                            n_jobs=-1,
                            verbose=0
                        )
                        
                        # Use scaled data for linear models, original for tree-based
                        if model_name in ['ridge', 'lasso']:
                            grid_search.fit(X_train_scaled, y_train)
                            y_pred = grid_search.predict(X_test_scaled)
                        else:
                            grid_search.fit(X_train, y_train)
                            y_pred = grid_search.predict(X_test)
                        
                        # Store model
                        self.models[target_name][model_name] = grid_search.best_estimator_
                        
                        # Calculate metrics
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        self.performance_metrics[target_name][model_name] = {
                            'mse': mse,
                            'rmse': rmse,
                            'mae': mae,
                            'r2': r2,
                            'best_params': grid_search.best_params_
                        }
                        
                        logger.info(f"{model_name} - R²: {r2:.4f}, RMSE: {rmse:.4f}")
                        
                        # Feature importance for tree-based models
                        if hasattr(grid_search.best_estimator_, 'feature_importances_'):
                            importance_df = pd.DataFrame({
                                'feature': X.columns,
                                'importance': grid_search.best_estimator_.feature_importances_
                            }).sort_values('importance', ascending=False)
                            
                            self.feature_importance[f"{target_name}_{model_name}"] = importance_df
                        
                    except Exception as e:
                        logger.error(f"Error training {model_name} for {target_name}: {str(e)}")
                        continue
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
    
    def save_models(self, filepath_prefix='models/hydroponics_model'):
        """
        Save all models and components using pickle
        """
        try:
            logger.info("Saving models and components")
            
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Save complete pipeline
            pipeline_data = {
                'models': self.models,
                'scalers': self.scalers,
                'imputers': self.imputers,
                'outlier_detector': self.outlier_detector,
                'feature_importance': self.feature_importance,
                'performance_metrics': self.performance_metrics,
                'random_state': self.random_state
            }
            
            with open(f'{filepath_prefix}.pkl', 'wb') as f:
                pickle.dump(pipeline_data, f)
            
            logger.info(f"Models saved successfully to {filepath_prefix}.pkl")
            
            # Save performance report
            self.generate_performance_report()
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
    
    def load_models(self, filepath):
        """
        Load saved models and components
        """
        try:
            logger.info(f"Loading models from {filepath}")
            
            with open(filepath, 'rb') as f:
                pipeline_data = pickle.load(f)
            
            self.models = pipeline_data.get('models', {})
            self.scalers = pipeline_data.get('scalers', {})
            self.imputers = pipeline_data.get('imputers', {})
            self.outlier_detector = pipeline_data.get('outlier_detector', None)
            self.feature_importance = pipeline_data.get('feature_importance', {})
            self.performance_metrics = pipeline_data.get('performance_metrics', {})
            self.random_state = pipeline_data.get('random_state', 42)
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def predict(self, X_new, target_name, model_name='random_forest'):
        """
        Make predictions with error handling
        """
        try:
            if target_name not in self.models:
                raise ValueError(f"No models trained for target: {target_name}")
            
            if model_name not in self.models[target_name]:
                raise ValueError(f"Model {model_name} not available for {target_name}")
            
            model = self.models[target_name][model_name]
            
            # Apply same preprocessing
            if model_name in ['ridge', 'lasso'] and 'main' in self.scalers:
                X_processed = self.scalers['main'].transform(X_new)
            else:
                X_processed = X_new
            
            predictions = model.predict(X_processed)
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def generate_performance_report(self):
        """
        Generate comprehensive performance report
        """
        try:
            logger.info("Generating performance report")
            
            report_lines = [
                "=" * 80,
                "HYDROPONICS ML PIPELINE - PERFORMANCE REPORT",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "=" * 80,
                ""
            ]
            
            for target_name, models in self.performance_metrics.items():
                report_lines.extend([
                    f"TARGET: {target_name.upper()}",
                    "-" * 50
                ])
                
                for model_name, metrics in models.items():
                    report_lines.extend([
                        f"  {model_name.upper()}:",
                        f"    R² Score: {metrics['r2']:.4f}",
                        f"    RMSE: {metrics['rmse']:.4f}",
                        f"    MAE: {metrics['mae']:.4f}",
                        f"    Best Params: {metrics['best_params']}",
                        ""
                    ])
                
                report_lines.append("")
            
            # Feature importance summary
            if self.feature_importance:
                report_lines.extend([
                    "FEATURE IMPORTANCE SUMMARY",
                    "-" * 50
                ])
                
                for key, importance_df in self.feature_importance.items():
                    report_lines.extend([
                        f"{key.upper()}:",
                        f"Top 5 features:"
                    ])
                    for idx, row in importance_df.head().iterrows():
                        report_lines.append(f"  {row['feature']}: {row['importance']:.4f}")
                    report_lines.append("")
            
            report_content = "\n".join(report_lines)
            
            # Save to file
            os.makedirs('output', exist_ok=True)
            with open('output/performance_report.txt', 'w') as f:
                f.write(report_content)
            
            logger.info("Performance report saved to output/performance_report.txt")
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")

# Main execution
def main():
    """
    Main execution function
    """
    try:
        logger.info("Starting Hydroponics ML Pipeline")
        
        # Initialize pipeline
        pipeline = HydroponicsMLPipeline()
        
        # Generate dataset if it doesn't exist
        dataset_path = 'data/hydroponics_dataset.csv'
        if not os.path.exists(dataset_path):
            logger.info("Dataset not found. Generating new dataset...")
            exec(open('data/generate_hydroponics_dataset.py').read())
        
        # Load and process data
        df = pipeline.load_and_validate_data(dataset_path)
        df_cleaned = pipeline.comprehensive_data_cleaning(df)
        df_features = pipeline.feature_engineering(df_cleaned)
        
        # Define target variables
        target_columns = ['growth_rate', 'expected_yield', 'health_score', 'disease_risk']
        
        # Prepare training data
        X, y_dict = pipeline.prepare_training_data(df_features, target_columns)
        
        # Train models
        pipeline.train_models(X, y_dict)
        
        # Save everything
        pipeline.save_models()
        
        logger.info("Pipeline completed successfully!")
        
        # Print summary
        print("\n" + "="*80)
        print("HYDROPONICS ML PIPELINE - EXECUTION SUMMARY")
        print("="*80)
        print(f"Dataset size: {df.shape}")
        print(f"Features created: {X.shape[1]}")
        print(f"Target variables: {len(target_columns)}")
        print(f"Models trained: {sum(len(models) for models in pipeline.models.values())}")
        print("\nFiles created:")
        print("- data/hydroponics_dataset.csv (50,000 samples)")
        print("- models/hydroponics_model.pkl (All models and components)")
        print("- output/performance_report.txt (Detailed performance metrics)")
        print("- hydroponics_ml.log (Execution log)")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
