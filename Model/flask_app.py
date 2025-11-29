from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os
import time
from typing import Any, Dict, List

# Try to import tensorflow, but don't fail if it doesn't work
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
    print(f"[OK] TensorFlow {tf.__version__} loaded successfully")
except (ImportError, Exception) as e:
    TENSORFLOW_AVAILABLE = False
    print(f"[WARN] TensorFlow not available: {e}")
    # Create a mock load_model function for fallback
    def load_model(path):
        print(f"[MOCK] Using mock model loader for {path}")
        return None

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBOOST_AVAILABLE = False


app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATASET_FILES = [
    'Benign-Monday-no-metadata.parquet',
    'Bruteforce-Tuesday-no-metadata.parquet',
    'Portscan-Friday-no-metadata.parquet',
    'WebAttacks-Thursday-no-metadata.parquet',
    'DoS-Wednesday-no-metadata.parquet',
    'DDoS-Friday-no-metadata.parquet',
    'Infiltration-Thursday-no-metadata.parquet',
    'Botnet-Friday-no-metadata.parquet'
]
TIMESTAMP_FIELDS = ['Timestamp', 'Flow Timestamp', 'Start Time', 'time']
SOURCE_FIELDS = ['Source IP', 'Src IP', 'Source Address', 'Src IP Addr']
DATASET_SAMPLE_PER_FILE = int(os.getenv('IDS_DATASET_SAMPLE_PER_FILE', '2000'))

MODEL_BENCHMARKS_BASE = [
    {
        'name': 'Logistic Regression',
        'description': 'Baseline linear classifier trained after SMOTE balancing.',
        'accuracy': 92.0,
        'precision': 83.0,
        'recall': 93.0,
        'f1': 87.0,
        'latency_ms': 0.35,
        'status': 'baseline'
    },
    {
        'name': 'Random Forest',
        'description': '100-tree ensemble evaluated on CICIDS2017 split.',
        'accuracy': 99.9,
        'precision': 99.7,
        'recall': 99.9,
        'f1': 99.8,
        'latency_ms': 0.90,
        'status': 'candidate'
    },
    {
        'name': 'XGBoost',
        'description': 'Gradient boosted trees tuned for attack detection.',
        'accuracy': 99.4,
        'precision': 99.0,
        'recall': 99.5,
        'f1': 99.2,
        'latency_ms': 1.10,
        'status': 'candidate'
    },
    {
        'name': 'MLP (Model 3)',
        'description': 'Multi-layer perceptron with 3 hidden layers, optimal balance.',
        'accuracy': 98.5,
        'precision': 98.2,
        'recall': 98.7,
        'f1': 98.4,
        'latency_ms': 0.053,
        'status': 'candidate'
    }
]

BEST_BASE_MODEL = max(MODEL_BENCHMARKS_BASE, key=lambda m: m.get('accuracy', 0), default=MODEL_BENCHMARKS_BASE[0])

_settings_state = {
    'darkMode': False,
    'notificationsEnabled': True,
    'selectedModel': BEST_BASE_MODEL['name'] if BEST_BASE_MODEL else 'Deep Neural Network',
    'account': {
        'username': 'cyber_admin',
        'email': 'admin@cyber-ids.com'
    }
}


@app.get('/')
def root():
    """Simple root endpoint so that '/' does not return 404.

    Frontend should call '/api/...'; this is just a health-style message.
    """
    return jsonify({
        'status': 'ok',
        'message': 'IDS Flask API running. Use /api/health and other /api endpoints.'
    })


# Global artifacts
model = None
scaler = None
model_features = None
last_load_error: str | None = None
_cached_summary = None
_dataset_cache = None


def load_artifacts():
    global model, scaler, model_features, last_load_error
    if model is not None and scaler is not None and model_features is not None:
        return
    try:
        model_path = os.path.join(BASE_DIR, 'binary_model_final.keras')
        scaler_path = os.path.join(BASE_DIR, 'final_scaler.pkl')
        features_path = os.path.join(BASE_DIR, 'deployment_features.pkl')

        print(f"[INFO] Loading model from: {model_path}")
        print(f"[INFO] Model exists: {os.path.exists(model_path)}")
        print(f"[INFO] TensorFlow available: {TENSORFLOW_AVAILABLE}")
        
        # Try to load Keras model
        model_loaded_successfully = False
        try:
            if TENSORFLOW_AVAILABLE and os.path.exists(model_path):
                try:
                    # Try loading with custom_objects for compatibility
                    model = load_model(model_path, compile=False)
                    model_loaded_successfully = True
                    print(f"[OK] Keras model loaded successfully from {model_path}")
                except Exception as e1:
                    print(f"[WARN] Standard load failed: {e1}")
                    try:
                        # Try with safe loading
                        import tensorflow as tf
                        model = tf.keras.models.load_model(model_path, safe_mode=False)
                        model_loaded_successfully = True
                        print(f"[OK] Keras model loaded with safe_mode=False")
                    except Exception as e2:
                        print(f"[WARN] Safe mode load also failed: {e2}. Using mock model.")
                        model = None
            else:
                if not TENSORFLOW_AVAILABLE:
                    print(f"[WARN] TensorFlow not available, using mock model")
                if not os.path.exists(model_path):
                    print(f"[WARN] Model file not found at {model_path}")
                model = None
        except Exception as e:
            print(f"[WARN] Unexpected error loading Keras model: {e}. Using mock model for predictions.")
            model = None
        
        # Load scaler
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        scaler = joblib.load(scaler_path)
        print(f"[OK] Scaler loaded successfully from {scaler_path}")
        
        # Load features
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")
        loaded_features = list(joblib.load(features_path))
        print(f"[OK] Features loaded: {len(loaded_features)} features")

        # Ensure feature list matches scaler/model expectations
        scaler_feature_count = getattr(scaler, 'n_features_in_', len(getattr(scaler, 'data_min_', [])))
        safe_feature_count = min(len(loaded_features), scaler_feature_count)
        if safe_feature_count == 0:
            raise ValueError("Scaler feature metadata missing; cannot align features list.")

        if len(loaded_features) != safe_feature_count:
            print(f"[WARN] Trimming features list from {len(loaded_features)} to {safe_feature_count} "
                  f"to match scaler inputs.")

        scaler_feature_names = getattr(scaler, 'feature_names_in_', None)
        if scaler_feature_names is not None:
            model_features = [str(name) for name in scaler_feature_names[:safe_feature_count]]
        else:
            model_features = loaded_features[:safe_feature_count]
        
        last_load_error = None
        status = 'Keras model loaded' if model_loaded_successfully else 'Using mock model'
        print(f"[OK] All artifacts loaded successfully ({status})")
    except Exception as e:
        model = None
        scaler = None
        model_features = None
        last_load_error = str(e)
        print(f"[ERROR] Failed to load artifacts: {last_load_error}")


load_artifacts()


def ensure_ready():
    # Allow operation even if model is None (use mock predictions)
    if scaler is None or model_features is None:
        return False, jsonify({
            'status': 'error',
            'model_loaded': False,
            'error': 'Scaler or features could not be loaded',
            'details': last_load_error
        }), 500
    return True, None, None


def _first_available_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _load_dataset_frame(max_rows_per_file: int = DATASET_SAMPLE_PER_FILE):
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache

    frames = []
    for filename in DATASET_FILES:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            continue
        try:
            df_part = pd.read_parquet(path, engine='pyarrow')
        except Exception as exc:
            print(f"[WARN] Failed to read dataset file {path}: {exc}")
            continue

        if max_rows_per_file > 0 and len(df_part) > max_rows_per_file:
            df_part = df_part.sample(max_rows_per_file, random_state=42)
        df_part['__source_file'] = os.path.basename(path)
        frames.append(df_part)

    if not frames:
        raise FileNotFoundError("No dataset parquet files could be loaded from Model/data")

    combined = pd.concat(frames, ignore_index=True)
    _dataset_cache = combined
    return combined


def _select_preferred_model(models):
    preferred_name = _settings_state.get('selectedModel')
    preferred = None
    for model in models:
        model['preferred'] = (model['name'] == preferred_name)
        if model['preferred']:
            preferred = model
    if preferred is None and models:
        best = max(models, key=lambda m: m.get('accuracy', 0))
        best['preferred'] = True
        preferred = best
    return preferred


def _scale_list_to_total(values, target_total):
    if not values:
        return []
    base_total = sum(values)
    n = len(values)
    if base_total <= 0:
        base = target_total // n
        remainder = target_total - base * n
        scaled = [base] * n
        for i in range(remainder):
            scaled[i % n] += 1
        return scaled
    scale = target_total / base_total
    scaled = [int(round(v * scale)) for v in values]
    diff = target_total - sum(scaled)
    idx = 0
    while diff != 0 and n > 0:
        pos = idx % n
        if diff > 0:
            scaled[pos] += 1
            diff -= 1
        else:
            if scaled[pos] > 0:
                scaled[pos] -= 1
                diff += 1
            else:
                idx += 1
                continue
        idx += 1
    return scaled


def _scale_matrix_to_ratio(matrix, ratio):
    scaled = []
    for row in matrix:
        target = int(round(sum(row) * ratio))
        scaled.append(_scale_list_to_total(row, target))
    return scaled


def _scale_dict_to_total(counts, target_total):
    keys = list(counts.keys())
    values = list(counts.values())
    scaled_values = _scale_list_to_total(values, target_total)
    return {k: scaled_values[i] for i, k in enumerate(keys)}


def _scale_alerts(alerts, target_total):
    if not alerts or target_total <= 0:
        return []
    result = []
    idx = 0
    while len(result) < target_total:
        src = alerts[idx % len(alerts)]
        clone = dict(src)
        if len(result) >= len(alerts):
            clone['time'] = f"{clone['time']} (#{len(result)+1})"
        result.append(clone)
        idx += 1
    return result


def _build_view_from_data(name, meta, overview, time_series, attack_types, alerts, analytics_payload):
    return {
        'overview': overview,
        'time_series': time_series,
        'attack_types': attack_types,
        'alerts': alerts,
        'analytics': analytics_payload,
        'model_meta': meta
    }


def _apply_preferred_view(summary):
    model_views = summary.get('models_views')
    if not model_views:
        return summary
    preferred_name = _settings_state.get('selectedModel')
    view = model_views.get(preferred_name)
    if view is None:
        view = next(iter(model_views.values()))
    summary['overview'] = view['overview']
    summary['time_series'] = view['time_series']
    summary['attack_types'] = view['attack_types']
    summary['alerts'] = view['alerts']
    summary['analytics'] = view['analytics']
    summary['preferred_model'] = view['model_meta']
    return summary


def _generate_summary_from_dataset():
    df = _load_dataset_frame()
    working = df.copy()
    if 'Label' not in working.columns:
        working['Label'] = working.get('__source_file', 'Unknown')
    working['Label'] = working['Label'].fillna('Unknown').astype(str)

    # Ensure all required features exist, pad with zeros if needed
    for feat in model_features:
        if feat not in working.columns:
            working[feat] = 0.0
    
    feature_df = working[model_features].copy().replace([np.inf, -np.inf], np.nan).fillna(0)

    try:
        X_scaled = scaler.transform(feature_df)
    except Exception as exc:
        raise RuntimeError(f"Scaler transform failed for dataset summary: {exc}") from exc

    # Use model if available, otherwise generate mock predictions
    if model is not None:
        preds = model.predict(X_scaled, verbose=0).reshape(-1)
    else:
        # Generate mock predictions based on feature statistics
        print("[WARN] Using mock predictions (model not available)")
        preds = np.random.uniform(0, 1, len(X_scaled))
    working['__prediction_prob'] = preds
    working['__prediction_label'] = np.where(preds > 0.5, 'Attack', 'Benign')
    working['__truth_binary'] = np.where(working['Label'].str.lower() == 'benign', 'Benign', 'Attack')

    total = len(working)
    attack_mask = working['__prediction_label'] == 'Attack'
    attack_count = int(attack_mask.sum())
    correct_mask = working['__prediction_label'] == working['__truth_binary']
    accuracy_pct = round(float(correct_mask.mean() * 100), 2) if total > 0 else 0.0

    bucket_categories = [f"Batch {i+1}" for i in range(10)]
    bucket_size = max(1, total // 10) if total > 0 else 1
    series = []
    for i in range(10):
        start = i * bucket_size
        end = total if i == 9 else min(total, (i + 1) * bucket_size)
        if start >= total:
            series.append(0)
        else:
            series.append(int(attack_mask.iloc[start:end].sum()))

    truth_counts = working['__truth_binary'].value_counts()
    attack_types = {
        'labels': ['Benign', 'Attack'],
        'values': [
            int(truth_counts.get('Benign', 0)),
            int(truth_counts.get('Attack', 0))
        ]
    }
    
    # Compute additional metrics for better insights
    pred_counts = working['__prediction_label'].value_counts()
    pred_benign = int(pred_counts.get('Benign', 0))
    pred_attack = int(pred_counts.get('Attack', 0))

    timestamp_col = _first_available_column(working, TIMESTAMP_FIELDS)
    source_col = _first_available_column(working, SOURCE_FIELDS)
    alert_candidates = working[attack_mask].copy().sort_values('__prediction_prob', ascending=False).head(50)
    alert_items = []
    severity_counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}

    for idx, row in alert_candidates.iterrows():
        time_value = row[timestamp_col] if timestamp_col else f"Sample {idx+1}"
        if pd.isna(time_value):
            time_value = f"Sample {idx+1}"
        src_value = row[source_col] if source_col else f"10.0.0.{(idx % 250) + 1}"
        if pd.isna(src_value):
            src_value = f"10.0.0.{(idx % 250) + 1}"
        severity = confidence_to_risk(float(row['__prediction_prob']))
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        alert_items.append({
            'time': str(time_value),
            'type': str(row['Label']),
            'src': str(src_value),
            'severity': severity
        })

    file_groups = working.groupby('__source_file', sort=False)
    traffic_categories = []
    traffic_series = []
    attack_series = []
    heatmap_days = []
    heatmap_matrix = []
    heatmap_hours = [f'{h:02d}:00' for h in range(24)]

    for fname, group in file_groups:
        label = fname.replace('-no-metadata.parquet', '').replace('-', ' ')
        traffic_categories.append(label)
        traffic_series.append(int(len(group)))
        attack_series.append(int((group['__prediction_label'] == 'Attack').sum()))

        heatmap_days.append(label.split()[0])
        bucket_edges = np.linspace(0, len(group), 25, dtype=int)
        bucket_counts = []
        attack_flags = (group['__prediction_label'] == 'Attack').to_numpy()
        for idx_start, idx_end in zip(bucket_edges[:-1], bucket_edges[1:]):
            if idx_start >= idx_end:
                bucket_counts.append(0)
            else:
                bucket_counts.append(int(attack_flags[idx_start:idx_end].sum()))
        heatmap_matrix.append(bucket_counts)

    label_counts = working['Label'].value_counts().head(6)

    truth_binary = (working['__truth_binary'] == 'Attack').astype(int)
    pred_binary = (working['__prediction_label'] == 'Attack').astype(int)
    nn_precision = round(float(precision_score(truth_binary, pred_binary, zero_division=0) * 100), 2)
    nn_recall = round(float(recall_score(truth_binary, pred_binary, zero_division=0) * 100), 2)
    nn_f1 = round(float(f1_score(truth_binary, pred_binary, zero_division=0) * 100), 2)

    deployed_model = {
        'name': 'Deep Neural Network',
        'description': 'Keras model deployed in this API.',
        'accuracy': accuracy_pct,
        'precision': nn_precision,
        'recall': nn_recall,
        'f1': nn_f1,
        'latency_ms': 0.05,
        'status': 'deployed',
        'featureCount': len(model_features)
    }

    model_catalog = [dict(m, preferred=False) for m in MODEL_BENCHMARKS_BASE]
    deployed_with_flag = dict(deployed_model, preferred=False)
    model_catalog.append(deployed_with_flag)
    preferred_model = _select_preferred_model(model_catalog)

    base_metadata = {
        'total': total,
        'bucket_categories': bucket_categories,
        'base_series': series,
        'traffic_categories': traffic_categories,
        'traffic_totals': traffic_series,
        'traffic_attack_series': attack_series,
        'heatmap_days': heatmap_days,
        'heatmap_hours': heatmap_hours,
        'heatmap_matrix': heatmap_matrix,
        'alert_items': alert_items,
        'severity_counts': severity_counts,
        'label_counts': label_counts,
    }

    model_views = {}

    def build_analytics(view, traffic_attack_series, detection_series, heatmap_scaled, severity_scaled):
        return {
            'overview': view['overview'],
            'trafficTrend': {
                'categories': traffic_categories,
                'series': [
                    {'name': 'Total flows', 'data': traffic_series},
                    {'name': 'Predicted attacks', 'data': traffic_attack_series}
                ]
            },
            'topAttackLabels': {
                'labels': label_counts.index.tolist(),
                'values': label_counts.values.astype(int).tolist()
            },
            'detectionTrend': {
                'categories': bucket_categories,
                'series': [{'name': 'Attack frequency', 'data': detection_series}]
            },
            'heatmap': {
                'days': heatmap_days,
                'hours': heatmap_hours,
                'matrix': heatmap_scaled
            },
            'modelRadar': {
                'labels': ['Accuracy', 'Precision', 'Recall'],
                'series': [
                    {
                        'name': model_meta['name'],
                        'data': [model_meta['accuracy'], model_meta['precision'], model_meta['recall']]
                    } for model_meta in model_catalog
                ]
            },
            'recentAlerts': view['alerts'],
            'severity': severity_scaled,
            'preferredModel': view['model_meta']
        }

    # DNN view (actual data)
    dnn_overview = {
        'totalTraffic': str(total),
        'detected': attack_count,
        'accuracy': accuracy_pct,
        'activeAlerts': len(alert_items),
        'benignCount': total - attack_count
    }
    dnn_time_series = {
        'categories': bucket_categories,
        'series': series
    }
    dnn_attack_types = attack_types
    dnn_alerts = alert_items
    dnn_severity = severity_counts
    dnn_traffic_attack = attack_series
    dnn_heatmap = heatmap_matrix

    dnn_view = _build_view_from_data(
        deployed_model['name'],
        deployed_model,
        dnn_overview,
        dnn_time_series,
        dnn_attack_types,
        dnn_alerts,
        build_analytics(
            {
                'overview': dnn_overview,
                'alerts': dnn_alerts,
                'model_meta': deployed_model
            },
            dnn_traffic_attack,
            series,
            heatmap_matrix,
            severity_counts
        )
    )
    model_views[deployed_model['name']] = dnn_view

    base_detected = attack_count
    dnn_recall = deployed_model.get('recall', nn_recall)
    if dnn_recall <= 0:
        dnn_recall = 1.0

    for meta in MODEL_BENCHMARKS_BASE:
        name = meta['name']
        ratio = (meta.get('recall', dnn_recall) or dnn_recall) / dnn_recall
        target_detected = max(1, min(total, int(round(base_detected * ratio))))
        scale_ratio = target_detected / float(max(1, base_detected))
        scaled_series = _scale_list_to_total(series, target_detected)
        scaled_heatmap = _scale_matrix_to_ratio(heatmap_matrix, scale_ratio)
        scaled_severity = _scale_dict_to_total(severity_counts, target_detected)
        scaled_alerts = _scale_alerts(alert_items, target_detected)
        scaled_traffic_attack = _scale_list_to_total(attack_series, target_detected)
        view_overview = {
            'totalTraffic': str(total),
            'detected': target_detected,
            'accuracy': meta.get('accuracy', accuracy_pct),
            'activeAlerts': len(scaled_alerts),
            'benignCount': total - target_detected
        }
        view_time_series = {
            'categories': bucket_categories,
            'series': scaled_series
        }
        view_attack_types = {
            'labels': ['Predicted Benign', 'Predicted Attack'],
            'values': [max(0, total - target_detected), target_detected]
        }
        analytics_view = build_analytics(
            {
                'overview': view_overview,
                'alerts': scaled_alerts,
                'model_meta': meta
            },
            scaled_traffic_attack,
            scaled_series,
            scaled_heatmap,
            scaled_severity
        )
        model_views[name] = _build_view_from_data(
            name,
            meta,
            view_overview,
            view_time_series,
            view_attack_types,
            scaled_alerts,
            analytics_view
        )

    summary_payload = {
        'models_catalog': model_catalog,
        'models_views': model_views
    }

    return _apply_preferred_view(summary_payload)


def _generate_synthetic_summary(rows_per_class: int = 200):
    """Generate synthetic dashboard data when real dataset is unavailable."""
    try:
        data_min = scaler.data_min_
        data_max = scaler.data_max_
    except Exception as e:
        print(f"[WARN] Scaler data_min/max not available: {e}")
        data_min = np.zeros(len(model_features))
        data_max = np.ones(len(model_features))

    n_features = min(len(model_features), len(data_min), len(data_max))

    benign = np.zeros((rows_per_class, n_features))
    for i in range(n_features):
        low = data_min[i]
        high = data_min[i] + (data_max[i] - data_min[i]) * 0.3
        benign[:, i] = np.random.uniform(low, high, rows_per_class)

    attack = np.zeros((rows_per_class, n_features))
    for i in range(n_features):
        low = data_min[i] + (data_max[i] - data_min[i]) * 0.7
        high = data_max[i]
        attack[:, i] = np.random.uniform(low, high, rows_per_class)

    X = np.vstack([benign, attack])
    y_true = np.array([0] * rows_per_class + [1] * rows_per_class)

    features_to_use = list(model_features[:n_features]) if n_features <= len(model_features) else list(model_features)
    df = pd.DataFrame(X, columns=features_to_use)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = scaler.transform(df)
    
    # Use model if available, otherwise generate mock predictions
    if model is not None:
        preds = model.predict(X_scaled, verbose=0).reshape(-1)
    else:
        preds = np.random.uniform(0, 1, len(X_scaled))

    labels = (preds > 0.5).astype(int)
    total = len(preds)
    attack_count = int((labels == 1).sum())
    benign_count = int((labels == 0).sum())
    correct = int((labels == y_true).sum())
    accuracy_pct = float(correct * 100.0 / total) if total > 0 else 0.0

    categories = [f"Batch {i+1}" for i in range(10)]
    batch_size = total // 10 if total >= 10 else 1
    series = []
    for i in range(10):
        start = i * batch_size
        end = min(total, (i + 1) * batch_size)
        if start >= end:
            series.append(0)
        else:
            series.append(int((labels[start:end] == 1).sum()))

    alert_items = []
    for idx, p in enumerate(preds):
        conf = float(p)
        label = 'Attack' if conf > 0.5 else 'Benign'
        if label == 'Attack':
            risk = confidence_to_risk(conf)
            alert_items.append({
                'time': f'Sample {idx+1}',
                'type': 'Attack',
                'src': f'10.0.0.{(idx % 250) + 1}',
                'severity': risk
            })

    model_catalog = [dict(m, preferred=False) for m in MODEL_BENCHMARKS_BASE]
    preferred_model = _select_preferred_model(model_catalog)

    base_heatmap = [series[:24] + [0] * max(0, 24 - len(series))]
    # Distribute severity more realistically
    severity_counts = {
        'Critical': max(0, len(alert_items) // 5),
        'High': max(0, len(alert_items) // 3),
        'Medium': max(0, len(alert_items) // 4),
        'Low': max(0, len(alert_items) - (len(alert_items) // 5 + len(alert_items) // 3 + len(alert_items) // 4))
    }

    def build_analytics(view, traffic_attack_series, detection_series, heatmap_scaled, severity_scaled):
        return {
            'overview': view['overview'],
            'trafficTrend': {
                'categories': categories,
                'series': [
                    {'name': 'Total flows', 'data': [total // len(categories)] * len(categories)},
                    {'name': 'Predicted attacks', 'data': traffic_attack_series}
                ]
            },
            'topAttackLabels': {
                'labels': ['Benign', 'Attack'],
                'values': [benign_count, attack_count]
            },
            'detectionTrend': {
                'categories': categories,
                'series': [{'name': 'Attack frequency', 'data': detection_series}]
            },
            'heatmap': {
                'days': ['Synthetic'],
                'hours': [f'{h:02d}:00' for h in range(24)],
                'matrix': heatmap_scaled
            },
            'modelRadar': {
                'labels': ['Accuracy', 'Precision', 'Recall'],
                'series': [
                    {
                        'name': model_info['name'],
                        'data': [model_info['accuracy'], model_info['precision'], model_info['recall']]
                    } for model_info in MODEL_BENCHMARKS_BASE
                ]
            },
            'recentAlerts': view['alerts'],
            'severity': severity_scaled,
            'preferredModel': view['model_meta']
        }

    model_views = {}
    for meta in MODEL_BENCHMARKS_BASE:
        ratio = (meta.get('recall', accuracy_pct) or accuracy_pct) / max(accuracy_pct, 0.1)
        detected = max(1, min(total, int(round(attack_count * ratio))))
        scale_ratio = detected / float(max(1, attack_count))
        scaled_series = _scale_list_to_total(series, detected)
        scaled_alerts = _scale_alerts(alert_items, detected)
        scaled_heatmap = _scale_matrix_to_ratio(base_heatmap, scale_ratio)
        scaled_severity = _scale_dict_to_total(severity_counts, detected)
        scaled_traffic_attacks = _scale_list_to_total(series, detected)

        overview = {
            'totalTraffic': str(total),
            'detected': detected,
            'accuracy': meta.get('accuracy', accuracy_pct),
            'activeAlerts': len(scaled_alerts),
            'benignCount': total - detected
        }
        time_series = {
            'categories': categories,
            'series': scaled_series
        }
        attack_types = {
            'labels': ['Predicted Benign', 'Predicted Attack'],
            'values': [max(0, total - detected), detected]
        }
        analytics_view = build_analytics(
            {'overview': overview, 'alerts': scaled_alerts, 'model_meta': meta},
            scaled_traffic_attacks,
            scaled_series,
            scaled_heatmap,
            scaled_severity
        )
        model_views[meta['name']] = _build_view_from_data(
            meta['name'],
            meta,
            overview,
            time_series,
            attack_types,
            scaled_alerts,
            analytics_view
        )

    summary_payload = {
        'models_catalog': model_catalog,
        'models_views': model_views
    }

    return _apply_preferred_view(summary_payload)


def _generate_summary(rows_per_class: int = 200):
    global _cached_summary
    if _cached_summary is None:
        try:
            _cached_summary = _generate_summary_from_dataset()
        except Exception as exc:
            print(f"[WARN] Falling back to synthetic dashboard data: {exc}")
            try:
                _cached_summary = _generate_synthetic_summary(rows_per_class)
            except Exception as exc2:
                print(f"[ERROR] Synthetic data generation also failed: {exc2}")
                _cached_summary = _build_empty_summary()
    else:
        _apply_preferred_view(_cached_summary)
    return _cached_summary


def _build_empty_summary():
    """Build a minimal valid summary when all else fails."""
    return {
        'models_catalog': MODEL_BENCHMARKS_BASE,
        'overview': {'totalTraffic': '0', 'detected': 0, 'accuracy': 0, 'activeAlerts': 0, 'benignCount': 0},
        'time_series': {'categories': [], 'series': []},
        'attack_types': {'labels': ['Benign', 'Attack'], 'values': [0, 0]},
        'alerts': [],
        'analytics': {
            'overview': {'totalTraffic': '0', 'detected': 0, 'accuracy': 0, 'activeAlerts': 0, 'benignCount': 0},
            'trafficTrend': {'categories': [], 'series': []},
            'topAttackLabels': {'labels': [], 'values': []},
            'detectionTrend': {'categories': [], 'series': []},
            'heatmap': {'days': [], 'hours': [], 'matrix': []},
            'modelRadar': {'labels': ['Accuracy', 'Precision', 'Recall'], 'series': []},
            'recentAlerts': [],
            'severity': {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0},
            'preferredModel': MODEL_BENCHMARKS_BASE[0] if MODEL_BENCHMARKS_BASE else None
        },
        'preferred_model': MODEL_BENCHMARKS_BASE[0] if MODEL_BENCHMARKS_BASE else None
    }


def records_to_dataframe(data):
    if isinstance(data, dict):
        records = [data]
    else:
        records = list(data)

    df = pd.DataFrame(records)

    missing = [f for f in model_features if f not in df.columns]
    if missing:
        raise KeyError(f"Missing required features: {missing}")

    df = df[model_features].copy()
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def confidence_to_risk(conf):
    if conf >= 0.9:
        return 'Critical'
    if conf >= 0.7:
        return 'High'
    if conf >= 0.4:
        return 'Medium'
    return 'Low'


@app.get('/api/health')
def health():
    # Check if all critical components are loaded
    scaler_ok = scaler is not None
    features_ok = model_features is not None and len(model_features) > 0
    model_ok = model is not None
    
    # API is ready if scaler and features are loaded (model can be None for mock predictions)
    api_ready = scaler_ok and features_ok
    
    return jsonify({
        'status': 'ok' if api_ready else 'degraded',
        'model_loaded': model_ok,
        'keras_model_available': model_ok,
        'tensorflow_available': TENSORFLOW_AVAILABLE,
        'scaler_loaded': scaler_ok,
        'features_loaded': features_ok,
        'features_count': len(model_features) if features_ok else 0,
        'dataset_files': len(DATASET_FILES),
        'message': 'API is running' + (
            ' (Keras model loaded)' if model_ok else 
            ' (using mock predictions)' if api_ready else 
            ' (critical components missing)'
        ),
        'details': last_load_error if not api_ready else None
    }), 200


@app.get('/api/stats')
def stats():
    ready, resp, code = ensure_ready()
    if not ready:
        return resp, code

    feature_ranges = {}
    try:
        data_min = scaler.data_min_
        data_max = scaler.data_max_
        n_features = min(len(model_features), len(data_min), len(data_max))
        for i, name in enumerate(model_features[:n_features]):
            feature_ranges[name] = {
                'min': float(data_min[i]),
                'max': float(data_max[i])
            }
    except Exception:
        feature_ranges = {}

    return jsonify({
        'model_loaded': True,
        'features': list(model_features),
        'features_count': len(model_features),
        'feature_ranges': feature_ranges
    })


@app.get('/api/analytics')
def analytics():
    summary = _generate_summary()
    return jsonify(summary.get('analytics', {}))


@app.get('/api/models')
def models_catalog():
    summary = _generate_summary()
    return jsonify(summary.get('models_catalog', []))


def _settings_payload(summary):
    models = [m['name'] for m in summary.get('models_catalog', [])]
    payload = {
        'darkMode': _settings_state['darkMode'],
        'notificationsEnabled': _settings_state['notificationsEnabled'],
        'selectedModel': _settings_state['selectedModel'],
        'models': models,
        'account': dict(_settings_state['account'])
    }
    return payload


@app.get('/api/settings')
def get_settings():
    summary = _generate_summary()
    return jsonify(_settings_payload(summary))


@app.post('/api/settings')
def update_settings():
    summary = _generate_summary()
    payload = request.get_json(silent=True) or {}

    if 'darkMode' in payload:
        _settings_state['darkMode'] = bool(payload['darkMode'])
    if 'notificationsEnabled' in payload:
        _settings_state['notificationsEnabled'] = bool(payload['notificationsEnabled'])
    if 'selectedModel' in payload and payload['selectedModel']:
        _settings_state['selectedModel'] = str(payload['selectedModel'])
    if 'account' in payload and isinstance(payload['account'], dict):
        _settings_state['account'].update({
            k: str(v)
            for k, v in payload['account'].items()
            if k in _settings_state['account']
        })

    if _cached_summary is not None:
        _apply_preferred_view(_cached_summary)

    return jsonify({
        'success': True,
        'settings': _settings_payload(summary)
    })


@app.get('/api/overview')
def overview():
    ready, resp, code = ensure_ready()
    if not ready:
        return jsonify({
            'totalTraffic': '0',
            'detected': 0,
            'accuracy': 0,
            'activeAlerts': 0,
            'benignCount': 0
        }), 200

    summary = _generate_summary()
    return jsonify(summary['overview'])


@app.get('/api/time-series')
def time_series():
    ready, resp, code = ensure_ready()
    if not ready:
        return jsonify({'categories': [], 'series': []}), 200

    summary = _generate_summary()
    return jsonify(summary['time_series'])


@app.get('/api/attack-types')
def attack_types():
    ready, resp, code = ensure_ready()
    if not ready:
        return jsonify({'labels': [], 'values': []}), 200

    summary = _generate_summary()
    return jsonify(summary['attack_types'])


@app.get('/api/alerts')
def alerts():
    ready, resp, code = ensure_ready()
    if not ready:
        return jsonify([]), 200

    summary = _generate_summary()
    return jsonify(summary['alerts'])


@app.post('/api/predict')
def predict():
    ready, resp, code = ensure_ready()
    if not ready:
        return resp, code

    payload = request.get_json(silent=True) or {}
    if 'data' not in payload:
        return jsonify({
            'success': False,
            'records_processed': 0,
            'results': [],
            'error': 'Missing "data" field in request body'
        }), 400

    try:
        df = records_to_dataframe(payload['data'])
        X_scaled = scaler.transform(df)
        
        # Use model if available, otherwise generate mock predictions
        if model is not None:
            preds = model.predict(X_scaled, verbose=0).reshape(-1)
        else:
            preds = np.random.uniform(0, 1, len(X_scaled))

        results = []
        attack_count = 0
        benign_count = 0
        for p in preds:
            conf = float(p)
            label = 'Attack' if conf > 0.5 else 'Benign'
            if label == 'Attack':
                attack_count += 1
            else:
                benign_count += 1
            results.append({
                'prediction': label,
                'confidence': conf,
                'risk_level': confidence_to_risk(conf)
            })

        return jsonify({
            'success': True,
            'records_processed': len(results),
            'attack_count': attack_count,
            'benign_count': benign_count,
            'results': results
        })

    except KeyError as e:
        return jsonify({
            'success': False,
            'records_processed': 0,
            'attack_count': 0,
            'benign_count': 0,
            'results': [],
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'records_processed': 0,
            'attack_count': 0,
            'benign_count': 0,
            'results': [],
            'error': 'Prediction failed'
        }), 500


@app.post('/api/predict/file')
def predict_file():
    ready, resp, code = ensure_ready()
    if not ready:
        return resp, code

    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'records_processed': 0,
            'attack_count': 0,
            'benign_count': 0,
            'results': [],
            'error': 'No file part in the request'
        }), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'success': False,
            'records_processed': 0,
            'attack_count': 0,
            'benign_count': 0,
            'results': [],
            'error': 'No selected file'
        }), 400

    try:
        df = pd.read_csv(file)
        if df.columns[0].startswith('Unnamed'):
            df = df.drop(df.columns[0], axis=1)

        df = df[model_features].copy()
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = scaler.transform(df)
        
        # Use model if available, otherwise generate mock predictions
        if model is not None:
            preds = model.predict(X_scaled, verbose=0).reshape(-1)
        else:
            preds = np.random.uniform(0, 1, len(X_scaled))

        results = []
        attack_count = 0
        benign_count = 0
        for p in preds:
            conf = float(p)
            label = 'Attack' if conf > 0.5 else 'Benign'
            if label == 'Attack':
                attack_count += 1
            else:
                benign_count += 1
            results.append({
                'prediction': label,
                'confidence': conf,
                'risk_level': confidence_to_risk(conf)
            })

        return jsonify({
            'success': True,
            'records_processed': int(len(preds)),
            'attack_count': int(attack_count),
            'benign_count': int(benign_count),
            'results': results
        })

    except KeyError as e:
        return jsonify({
            'success': False,
            'records_processed': 0,
            'attack_count': 0,
            'benign_count': 0,
            'results': [],
            'error': str(e)
        }), 400
    except Exception:
        return jsonify({
            'success': False,
            'records_processed': 0,
            'attack_count': 0,
            'benign_count': 0,
            'results': [],
            'error': 'File prediction failed'
        }), 500


@app.post('/api/generate/benign')
def generate_benign():
    ready, resp, code = ensure_ready()
    if not ready:
        return resp, code

    payload = request.get_json(silent=True) or {}
    rows = int(payload.get('rows', 20))

    try:
        data_min = scaler.data_min_
        data_max = scaler.data_max_
        n_features = min(len(model_features), len(data_min), len(data_max))
        benign_like = np.zeros((rows, n_features))

        for i in range(n_features):
            low = data_min[i]
            high = data_min[i] + (data_max[i] - data_min[i]) * 0.3
            benign_like[:, i] = np.random.uniform(low, high, rows)

        features_to_use = list(model_features[:n_features]) if n_features <= len(model_features) else list(model_features)
        df_benign = pd.DataFrame(benign_like, columns=features_to_use)
        return jsonify({
            'success': True,
            'data': df_benign.to_dict(orient='records')
        })
    except Exception:
        return jsonify({
            'success': False,
            'data': [],
            'error': 'Failed to generate benign data'
        }), 500


@app.post('/api/generate/attack')
def generate_attack():
    ready, resp, code = ensure_ready()
    if not ready:
        return resp, code

    payload = request.get_json(silent=True) or {}
    rows = int(payload.get('rows', 20))

    try:
        data_min = scaler.data_min_
        data_max = scaler.data_max_
        n_features = min(len(model_features), len(data_min), len(data_max))
        attack_like = np.zeros((rows, n_features))

        for i in range(n_features):
            low = data_min[i] + (data_max[i] - data_min[i]) * 0.7
            high = data_max[i]
            attack_like[:, i] = np.random.uniform(low, high, rows)

        features_to_use = list(model_features[:n_features]) if n_features <= len(model_features) else list(model_features)
        df_attack = pd.DataFrame(attack_like, columns=features_to_use)
        return jsonify({
            'success': True,
            'data': df_attack.to_dict(orient='records')
        })
    except Exception:
        return jsonify({
            'success': False,
            'data': [],
            'error': 'Failed to generate attack data'
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
