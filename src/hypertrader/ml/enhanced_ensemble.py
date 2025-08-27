"""
Enhanced ML Ensemble with LightGBM/XGBoost + Transformers
Replaces the current ML ensemble with proper financial ML structure.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
import warnings

# Core ML imports
try:
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
except ImportError as e:
    logging.warning(f"ML libraries not available: {e}")
    lgb = xgb = RandomForestClassifier = SVC = StandardScaler = None
    Pipeline = TimeSeriesSplit = None

# Transformer imports
try:
    from transformers import AutoModel, AutoConfig, AutoTokenizer
    from transformers import Trainer, TrainingArguments, TrainerCallback
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    logging.warning(f"Transformers not available: {e}")
    AutoModel = AutoConfig = AutoTokenizer = None
    Trainer = TrainingArguments = TrainerCallback = None
    torch = nn = Dataset = DataLoader = None

from .cross_validation import PurgedKFold, ModelEvaluator
from .labeling import TripleBarrierLabeler, create_ml_features

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Structured model prediction result."""
    prediction: float
    confidence: float
    model_name: str
    features_used: List[str]
    metadata: Optional[Dict[str, Any]] = None


class FinancialTimeSeriesDataset(Dataset):
    """PyTorch Dataset for financial time series data."""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, sequence_length: int = 60):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.sequences[idx],
            'labels': self.labels[idx]
        }


class FinancialTransformer(nn.Module):
    """Custom transformer for financial time series forecasting."""
    
    def __init__(
        self, 
        input_dim: int = 10,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._create_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_len: int, d_model: int):
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Project input to model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_encoding
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer
        transformer_output = self.transformer_encoder(x)
        
        # Global average pooling
        pooled = transformer_output.mean(dim=1)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(pooled)  # (batch_size, num_classes)
        
        return logits


class EnhancedMLEnsemble:
    """
    Enhanced ML ensemble combining tabular models (LightGBM/XGBoost) 
    with sequence models (Transformers) for financial forecasting.
    """
    
    def __init__(
        self,
        use_lightgbm: bool = True,
        use_xgboost: bool = True,
        use_random_forest: bool = True,
        use_transformer: bool = True,
        sequence_length: int = 60,
        ensemble_method: str = "weighted_average",
        cv_folds: int = 5
    ):
        self.use_lightgbm = use_lightgbm and lgb is not None
        self.use_xgboost = use_xgboost and xgb is not None
        self.use_random_forest = use_random_forest and RandomForestClassifier is not None
        self.use_transformer = use_transformer and torch is not None
        
        self.sequence_length = sequence_length
        self.ensemble_method = ensemble_method
        self.cv_folds = cv_folds
        
        # Initialize models
        self.models = {}
        self.model_weights = {}
        self.scalers = {}
        self.fitted = False
        
        # Performance tracking
        self.model_evaluator = ModelEvaluator()
        self.cv_scores = {}
        
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all ensemble models."""
        
        # LightGBM - Fast, memory efficient, great for tabular data
        if self.use_lightgbm:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                objective='binary',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=42,
                n_estimators=100
            )
            
        # XGBoost - Robust, handles missing values well
        if self.use_xgboost:
            self.models['xgboost'] = xgb.XGBClassifier(
                objective='binary:logistic',
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_estimators=100,
                eval_metric='logloss'
            )
            
        # Random Forest - Good baseline, handles non-linear relationships
        if self.use_random_forest:
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            
        # Financial Transformer - Captures sequential patterns
        if self.use_transformer:
            self.models['transformer'] = FinancialTransformer(
                input_dim=10,  # Will be updated based on features
                d_model=128,
                nhead=8,
                num_layers=4,
                dropout=0.1
            )
            
        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
        
    def fit(
        self, 
        X: pd.DataFrame,
        y: pd.Series,
        feature_columns: Optional[List[str]] = None,
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Fit ensemble models with proper cross-validation.
        
        Args:
            X: Feature matrix with datetime index
            y: Target labels (from triple-barrier labeling)
            feature_columns: Specific features to use
            sample_weights: Optional sample weights
            
        Returns:
            Cross-validation results for each model
        """
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
            
        # Prepare features
        if feature_columns is not None:
            X_features = X[feature_columns].copy()
        else:
            X_features = X.copy()
            
        # Remove non-numeric columns and handle NaN
        numeric_columns = X_features.select_dtypes(include=[np.number]).columns
        X_numeric = X_features[numeric_columns].fillna(0)
        
        if X_numeric.empty:
            raise ValueError("No numeric features available for training")
            
        logger.info(f"Training on {len(X_numeric)} samples with {len(X_numeric.columns)} features")
        
        # Initialize cross-validation
        cv_splitter = PurgedKFold(n_splits=self.cv_folds) if hasattr(X.index, 'to_pydatetime') else TimeSeriesSplit(n_splits=self.cv_folds)
        
        model_results = {}
        
        # Train tabular models
        for model_name in ['lightgbm', 'xgboost', 'random_forest']:
            if model_name not in self.models:
                continue
                
            try:
                results = self._fit_tabular_model(
                    model_name, X_numeric, y, cv_splitter, sample_weights
                )
                model_results[model_name] = results
                logger.info(f"{model_name} CV results: {results}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                
        # Train transformer model
        if 'transformer' in self.models and self.use_transformer:
            try:
                results = self._fit_transformer_model(X_numeric, y)
                model_results['transformer'] = results
                logger.info(f"Transformer results: {results}")
                
            except Exception as e:
                logger.error(f"Failed to train transformer: {e}")
                
        # Calculate ensemble weights based on performance
        self._calculate_ensemble_weights(model_results)
        
        self.fitted = True
        self.cv_scores = model_results
        
        logger.info(f"Ensemble training completed. Model weights: {self.model_weights}")
        return model_results
        
    def _fit_tabular_model(
        self, 
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        cv_splitter,
        sample_weights: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Fit tabular model with cross-validation."""
        model = self.models[model_name]
        
        # Scale features for some models
        if model_name in ['svm']:  # Add more if needed
            scaler = StandardScaler()
            self.scalers[model_name] = scaler
        else:
            scaler = None
            
        cv_scores = []
        cv_predictions = []
        cv_actuals = []
        
        # Cross-validation loop
        try:
            splits = list(cv_splitter.split(X, y))
        except:
            # Fallback to simple train/test split
            split_idx = int(len(X) * 0.8)
            train_idx = np.arange(split_idx)
            test_idx = np.arange(split_idx, len(X))
            splits = [(train_idx, test_idx)]
            
        for train_idx, test_idx in splits:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Apply scaling if needed
            if scaler is not None:
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index
                )
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )
            else:
                X_train_scaled, X_test_scaled = X_train, X_test
                
            # Fit model
            if sample_weights is not None:
                model.fit(X_train_scaled, y_train, sample_weight=sample_weights[train_idx])
            else:
                model.fit(X_train_scaled, y_train)
                
            # Predict
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = y_pred.copy()
                
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores.append(accuracy)
            cv_predictions.extend(y_pred_proba)
            cv_actuals.extend(y_test.values)
            
        # Final model fitting on all data
        X_final = X.copy()
        if scaler is not None:
            self.scalers[model_name] = StandardScaler()
            X_final = pd.DataFrame(
                self.scalers[model_name].fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
        if sample_weights is not None:
            model.fit(X_final, y, sample_weight=sample_weights)
        else:
            model.fit(X_final, y)
            
        # Calculate overall metrics
        cv_predictions = np.array(cv_predictions)
        cv_actuals = np.array(cv_actuals)
        
        try:
            auc = roc_auc_score(cv_actuals, cv_predictions)
        except:
            auc = 0.5
            
        precision, recall, f1, _ = precision_recall_fscore_support(
            cv_actuals, (cv_predictions > 0.5).astype(int), average='binary'
        )
        
        return {
            'cv_accuracy': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    def _fit_transformer_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Fit transformer model for sequence learning."""
        if torch is None:
            return {'error': 'PyTorch not available'}
            
        # Prepare sequence data
        sequences, labels = self._prepare_sequence_data(X.values, y.values)
        
        if len(sequences) < 10:
            return {'error': 'Insufficient sequence data'}
            
        # Update model input dimension
        self.models['transformer'].input_projection = nn.Linear(
            sequences.shape[-1], 
            self.models['transformer'].input_projection.out_features
        )
        
        # Create dataset and dataloader
        dataset = FinancialTimeSeriesDataset(sequences, labels, self.sequence_length)
        
        # Simple train/validation split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Training setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.models['transformer'].to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        train_losses = []
        
        for epoch in range(10):  # Quick training
            epoch_loss = 0.0
            for batch in train_loader:
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].long().to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
        # Validation
        model.eval()
        val_predictions = []
        val_actuals = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                
                val_predictions.extend(probabilities.cpu().numpy())
                val_actuals.extend(labels.cpu().numpy())
                
        # Calculate metrics
        val_predictions = np.array(val_predictions)
        val_actuals = np.array(val_actuals)
        
        try:
            auc = roc_auc_score(val_actuals, val_predictions)
            accuracy = accuracy_score(val_actuals, (val_predictions > 0.5).astype(int))
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_actuals, (val_predictions > 0.5).astype(int), average='binary'
            )
        except:
            auc = accuracy = precision = recall = f1 = 0.0
            
        return {
            'validation_accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'final_loss': train_losses[-1] if train_losses else 0.0
        }
        
    def _prepare_sequence_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for transformer training."""
        sequences = []
        labels = []
        
        for i in range(self.sequence_length, len(X)):
            # Get sequence of features
            seq = X[i-self.sequence_length:i]
            label = y[i]
            
            sequences.append(seq)
            labels.append(label)
            
        return np.array(sequences), np.array(labels)
        
    def _calculate_ensemble_weights(self, model_results: Dict[str, Dict[str, float]]):
        """Calculate ensemble weights based on cross-validation performance."""
        if not model_results:
            return
            
        # Use F1 score as primary metric for weighting
        f1_scores = {}
        for model_name, results in model_results.items():
            f1_score = results.get('f1', 0.0)
            f1_scores[model_name] = max(f1_score, 0.1)  # Minimum weight
            
        # Normalize weights
        total_score = sum(f1_scores.values())
        if total_score > 0:
            self.model_weights = {
                model: score / total_score 
                for model, score in f1_scores.items()
            }
        else:
            # Equal weights fallback
            n_models = len(model_results)
            self.model_weights = {
                model: 1.0 / n_models 
                for model in model_results.keys()
            }
            
    def predict(self, X: pd.DataFrame) -> List[ModelResult]:
        """
        Generate ensemble predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of ModelResult objects with predictions and metadata
        """
        if not self.fitted:
            raise ValueError("Ensemble not fitted yet")
            
        # Prepare features
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_columns].fillna(0)
        
        results = []
        
        # Tabular model predictions
        for model_name in ['lightgbm', 'xgboost', 'random_forest']:
            if model_name not in self.models:
                continue
                
            try:
                model = self.models[model_name]
                
                # Apply scaling if needed
                X_pred = X_numeric
                if model_name in self.scalers:
                    X_pred = pd.DataFrame(
                        self.scalers[model_name].transform(X_numeric),
                        columns=X_numeric.columns,
                        index=X_numeric.index
                    )
                    
                # Predict
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_pred)[:, 1]
                    predictions = (probabilities > 0.5).astype(float)
                    confidences = probabilities
                else:
                    predictions = model.predict(X_pred)
                    confidences = np.abs(predictions - 0.5) + 0.5  # Simple confidence measure
                    
                # Create results
                for i, (pred, conf) in enumerate(zip(predictions, confidences)):
                    result = ModelResult(
                        prediction=pred,
                        confidence=conf,
                        model_name=model_name,
                        features_used=list(X_numeric.columns),
                        metadata={'model_weight': self.model_weights.get(model_name, 0.0)}
                    )
                    results.append(result)
                    
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
                
        # Transformer predictions
        if 'transformer' in self.models and torch is not None:
            try:
                results.extend(self._predict_transformer(X_numeric))
            except Exception as e:
                logger.warning(f"Transformer prediction failed: {e}")
                
        return results
        
    def _predict_transformer(self, X: pd.DataFrame) -> List[ModelResult]:
        """Generate transformer predictions."""
        model = self.models['transformer']
        device = next(model.parameters()).device
        
        # Prepare sequence data
        sequences, _ = self._prepare_sequence_data(X.values, np.zeros(len(X)))
        
        if len(sequences) == 0:
            return []
            
        model.eval()
        results = []
        
        with torch.no_grad():
            for i, seq in enumerate(sequences):
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
                outputs = model(seq_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                pred = probabilities[0, 1].cpu().numpy()
                conf = float(torch.max(probabilities).cpu().numpy())
                
                result = ModelResult(
                    prediction=pred,
                    confidence=conf,
                    model_name='transformer',
                    features_used=list(X.columns),
                    metadata={'model_weight': self.model_weights.get('transformer', 0.0)}
                )
                results.append(result)
                
        return results
        
    def predict_ensemble(self, X: pd.DataFrame) -> Tuple[float, float, Dict[str, float]]:
        """
        Generate ensemble prediction combining all models.
        
        Args:
            X: Feature matrix
            
        Returns:
            (ensemble_prediction, ensemble_confidence, individual_predictions)
        """
        individual_results = self.predict(X)
        
        if not individual_results:
            return 0.5, 0.0, {}
            
        # Group results by model
        model_predictions = {}
        for result in individual_results:
            if result.model_name not in model_predictions:
                model_predictions[result.model_name] = []
            model_predictions[result.model_name].append(result.prediction)
            
        # Calculate weighted ensemble
        ensemble_pred = 0.0
        ensemble_conf = 0.0
        total_weight = 0.0
        
        individual_preds = {}
        
        for model_name, predictions in model_predictions.items():
            model_pred = np.mean(predictions)
            model_weight = self.model_weights.get(model_name, 0.0)
            
            ensemble_pred += model_weight * model_pred
            ensemble_conf += model_weight  # Simple confidence aggregation
            total_weight += model_weight
            
            individual_preds[model_name] = model_pred
            
        if total_weight > 0:
            ensemble_pred /= total_weight
            ensemble_conf /= total_weight
        else:
            ensemble_pred = 0.5
            ensemble_conf = 0.0
            
        return ensemble_pred, ensemble_conf, individual_preds
        
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from tree-based models."""
        importances = {}
        
        for model_name in ['lightgbm', 'xgboost', 'random_forest']:
            if model_name not in self.models or not self.fitted:
                continue
                
            model = self.models[model_name]
            
            try:
                if hasattr(model, 'feature_importances_'):
                    # Get feature names (assuming last fitted data structure)
                    feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
                    importances[model_name] = dict(zip(feature_names, model.feature_importances_))
                    
            except Exception as e:
                logger.warning(f"Failed to get feature importance for {model_name}: {e}")
                
        return importances
        
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance metrics."""
        return {
            'cv_scores': self.cv_scores,
            'model_weights': self.model_weights,
            'fitted': self.fitted,
            'models_available': list(self.models.keys())
        }
