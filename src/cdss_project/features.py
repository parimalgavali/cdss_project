IDENTIFIER_COLUMNS: tuple[str, ...] = ("stay_id", "hour_before")
LABEL_COLUMN = "sepsis"

STATIC_NUMERIC_COLUMNS: tuple[str, ...] = (
    "admission_age",
    "height",
    "weight",
    "diabetes",
)

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "gender",
    "ethnicity",
    "last_careunit",
)

DYNAMIC_NUMERIC_COLUMNS: tuple[str, ...] = (
    "platelet",
    "leukocytes",
    "po2",
    "fio2",
    "lactate",
    "creatinine",
    "bilirubin_total",
    "gcs",
    "crp",
    "sofa_24hours",
    "dbp",
    "glucose",
    "heart_rate",
    "mbp",
    "resp_rate",
    "sbp",
    "temperature",
    "spo2",
    "urineoutput",
    "ventilation_status",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    *STATIC_NUMERIC_COLUMNS,
    *CATEGORICAL_COLUMNS,
    *DYNAMIC_NUMERIC_COLUMNS,
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    *IDENTIFIER_COLUMNS,
    *FEATURE_COLUMNS,
    LABEL_COLUMN,
)

INTERPOLATED_FEATURE_COLUMNS: tuple[str, ...] = (
    *STATIC_NUMERIC_COLUMNS,
    *DYNAMIC_NUMERIC_COLUMNS,
)
