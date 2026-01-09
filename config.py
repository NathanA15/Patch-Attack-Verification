from pathlib import Path
import os
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[0]

# input_box_configs_dir = PROJECT_ROOT / "input_box_configs"
BOX_DIR = PROJECT_ROOT / "input_box_configs"
ERAN_ROOT = PROJECT_ROOT / "util" / "ERAN"
ERAN_DIR = ERAN_ROOT / "tf_verify"
PYTHON_BIN = "python3"
DATASET = "mnist"
NETNAME = ERAN_DIR / "models" / "mnist_convSmallRELU__PGDK.onnx"
LOG_DIR = PROJECT_ROOT / "logs"
CSV_DIR = PROJECT_ROOT / "csv"


LOG_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
daily_folder = LOG_DIR / datetime.now().strftime("%Y%m%d")
daily_folder.mkdir(parents=True, exist_ok=True)
LOG_FILE = daily_folder / f"{timestamp}_eran_run.log"

MNIST_DATA_PATH = ERAN_ROOT / "data" / "mnist_test.csv"


GRB_LICENSE_FILE = ERAN_ROOT / "gurobi912" / "linux64" / "gurobi.lic"
os.environ["GRB_LICENSE_FILE"] = str(GRB_LICENSE_FILE)
print(os.environ.get("GRB_LICENSE_FILE"))
