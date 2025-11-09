"""
Model training module with ensemble learning (LightGBM, CatBoost, XGBoost).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from config import (
    LIGHTGBM_PARAMS, CATBOOST_PARAMS, XGBOOST_PARAMS,
    LIGHTGBM_MODEL_PATH, CATBOOST_MODEL_PATH, XGBOOST_MODEL_PATH,
    ENSEMBLE_MODEL_PATH, TEST_SIZE, RANDOM_STATE, CROSS_VALIDATION_FOLDS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleModelTrainer:
    """
    Train and evaluate ensemble models for fake review detection.
    """
    
    def __init__(self):
        """Initialize the model trainer."""
        self.lightgbm_model = None
        self.catboost_model = None
        self.xgboost_model = None
        self.ensemble_model = None
        self.models_trained = False
    
    def create_models(self) -> Tuple[Any, Any, Any]:
        """
        Create individual models with configured hyperparameters.
        
        Returns:
            Tuple: (lightgbm, catboost, xgboost) models
        """
        logger.info("Creating models with optimized hyperparameters...")
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)
        
        # CatBoost
        cat_model = CatBoostClassifier(**CATBOOST_PARAMS)
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        
        logger.info("Models created successfully!")
        return lgb_model, cat_model, xgb_model
    
    def train_individual_model(self, model: Any, X_train: np.ndarray, 
                              y_train: np.ndarray, X_val: np.ndarray = None,
                              y_val: np.ndarray = None, model_name: str = "") -> Any:
        """
        Train an individual model.
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            model_name: Name of the model for logging
            
        Returns:
            Trained model
        """
        logger.info(f"Training {model_name}...")
        
        if X_val is not None and y_val is not None:
            # Train with validation set for early stopping
            if isinstance(model, lgb.LGBMClassifier):
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
            elif isinstance(model, CatBoostClassifier):
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False
                )
            elif isinstance(model, xgb.XGBClassifier):
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
        else:
            # Train without validation set
            model.fit(X_train, y_train)
        
        logger.info(f"{model_name} training completed!")
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train all models in the ensemble.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dict: Trained models
        """
        # Create models
        lgb_model, cat_model, xgb_model = self.create_models()
        
        # Train individual models
        self.lightgbm_model = self.train_individual_model(
            lgb_model, X_train, y_train, X_val, y_val, "LightGBM"
        )
        
        self.catboost_model = self.train_individual_model(
            cat_model, X_train, y_train, X_val, y_val, "CatBoost"
        )
        
        self.xgboost_model = self.train_individual_model(
            xgb_model, X_train, y_train, X_val, y_val, "XGBoost"
        )
        
        # Create voting ensemble
        logger.info("Creating voting ensemble...")
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('lightgbm', self.lightgbm_model),
                ('catboost', self.catboost_model),
                ('xgboost', self.xgboost_model)
            ],
            voting='soft',
            weights=[1, 1, 1]  # Equal weights
        )
        
        # Fit ensemble (it will use already trained models)
        self.ensemble_model.fit(X_train, y_train)
        
        self.models_trained = True
        logger.info("All models trained successfully!")
        
        return {
            'lightgbm': self.lightgbm_model,
            'catboost': self.catboost_model,
            'xgboost': self.xgboost_model,
            'ensemble': self.ensemble_model
        }
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                       model_name: str = "") -> Dict[str, float]:
        """
        Evaluate a single model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dict: Evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info(f"\n{model_name} Performance:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dict: Evaluation metrics for all models
        """
        if not self.models_trained:
            raise ValueError("Models must be trained before evaluation")
        
        logger.info("\n" + "=" * 60)
        logger.info("MODEL EVALUATION")
        logger.info("=" * 60)
        
        results = {}
        
        # Evaluate individual models
        results['lightgbm'] = self.evaluate_model(
            self.lightgbm_model, X_test, y_test, "LightGBM"
        )
        
        results['catboost'] = self.evaluate_model(
            self.catboost_model, X_test, y_test, "CatBoost"
        )
        
        results['xgboost'] = self.evaluate_model(
            self.xgboost_model, X_test, y_test, "XGBoost"
        )
        
        # Evaluate ensemble
        results['ensemble'] = self.evaluate_model(
            self.ensemble_model, X_test, y_test, "Ensemble"
        )
        
        logger.info("=" * 60)
        
        return results
    
    def plot_confusion_matrix(self, model: Any, X_test: np.ndarray, 
                             y_test: np.ndarray, model_name: str = "",
                             save_path: str = None):
        """
        Plot confusion matrix for a model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            save_path: Path to save the plot
        """
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Genuine', 'Fake'],
                    yticklabels=['Genuine', 'Fake'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(self, models_dict: Dict[str, Any], X_test: np.ndarray,
                       y_test: np.ndarray, save_path: str = None):
        """
        Plot ROC curves for all models.
        
        Args:
            models_dict: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models_dict.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name.upper()} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - All Models')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"ROC curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_models(self):
        """Save all trained models to disk."""
        if not self.models_trained:
            raise ValueError("Models must be trained before saving")
        
        logger.info("Saving models...")
        
        joblib.dump(self.lightgbm_model, LIGHTGBM_MODEL_PATH)
        logger.info(f"LightGBM saved to {LIGHTGBM_MODEL_PATH}")
        
        joblib.dump(self.catboost_model, CATBOOST_MODEL_PATH)
        logger.info(f"CatBoost saved to {CATBOOST_MODEL_PATH}")
        
        joblib.dump(self.xgboost_model, XGBOOST_MODEL_PATH)
        logger.info(f"XGBoost saved to {XGBOOST_MODEL_PATH}")
        
        joblib.dump(self.ensemble_model, ENSEMBLE_MODEL_PATH)
        logger.info(f"Ensemble saved to {ENSEMBLE_MODEL_PATH}")
        
        logger.info("All models saved successfully!")
    
    def get_feature_importance(self, feature_names: list, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained models.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if not self.models_trained:
            raise ValueError("Models must be trained first")
        
        # Get feature importance from each model
        lgb_importance = self.lightgbm_model.feature_importances_
        cat_importance = self.catboost_model.feature_importances_
        xgb_importance = self.xgboost_model.feature_importances_
        
        # Average importance
        avg_importance = (lgb_importance + cat_importance + xgb_importance) / 3
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'lightgbm': lgb_importance,
            'catboost': cat_importance,
            'xgboost': xgb_importance,
            'average': avg_importance
        })
        
        # Sort by average importance
        importance_df = importance_df.sort_values('average', ascending=False)
        
        return importance_df.head(top_n)


if __name__ == "__main__":
    logger.info("Model Trainer Module - Ready for training!")
