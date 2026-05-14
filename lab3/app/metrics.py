from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

REGISTRY = CollectorRegistry()

# 1. Лічильник прогнозів
PREDICTION_COUNTER = Counter(
    "ml_predictions_total",
    "Загальна кількість прогнозів моделі",
    labelnames=["class_name", "status"],
    registry=REGISTRY,
)

# 2. Гістограма latency інференсу
PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "Час обробки одного прогнозу (секунди)",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)

# 3. Гістограма впевненості моделі
PREDICTION_CONFIDENCE = Histogram(
    "ml_prediction_confidence",
    "Розподіл значень predict_proba для обраного класу",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    registry=REGISTRY,
)

# 4. Лічильник помилок
ERROR_COUNTER = Counter(
    "ml_errors_total",
    "Кількість помилкових запитів",
    labelnames=["error_type"],
    registry=REGISTRY,
)

# 5. Gauge — чи завантажена модель
MODEL_LOADED = Gauge(
    "ml_model_loaded",
    "Чи успішно завантажена модель ",
    registry=REGISTRY,
)

# 6. Лічильники drift
DRIFT_CHECKS = Counter(
    "ml_drift_checks_total",
    "Загальна кількість виконаних перевірок на drift",
    registry=REGISTRY,
)

DRIFT_DETECTED = Counter(
    "ml_drift_detected_total",
    "Кількість випадків, коли drift був виявлений",
    labelnames=["feature"],
    registry=REGISTRY,
)