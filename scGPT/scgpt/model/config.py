# scgpt/model/config.py

NUM_TASKS = 4

TASK_NAME_TO_ID = {
    "expr": 0,
    "direction": 1,
    "go": 2,
    "moa_broad": 3,
}

TASK_ID_TO_NAME = {v: k for k, v in TASK_NAME_TO_ID.items()}

TASK_TYPES = {
    "expr": "regression",
    "direction": "ternary",
    "go": "multiclass",
    "moa_broad": "multiclass",
}
