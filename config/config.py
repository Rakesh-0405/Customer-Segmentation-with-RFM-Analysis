import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUTS_PATH = os.path.join(BASE_DIR, 'outputs')
MODELS_PATH = os.path.join(OUTPUTS_PATH, 'models')
PLOTS_PATH = os.path.join(OUTPUTS_PATH, 'plots')
REPORTS_PATH = os.path.join(OUTPUTS_PATH, 'reports')

for path in [DATA_RAW_PATH, DATA_PROCESSED_PATH, MODELS_PATH, PLOTS_PATH, REPORTS_PATH]:
    os.makedirs(path, exist_ok=True)

SNAPSHOT_DATE = datetime(2011, 12, 10)
N_CLUSTERS = 4
RANDOM_STATE = 42

DATA_FILE = 'Online_Retail.xlsx'
RAW_DATA_PATH = os.path.join(DATA_RAW_PATH, DATA_FILE)

print("Configuration loaded successfully!")
print(f"Data will be loaded from: {RAW_DATA_PATH}")