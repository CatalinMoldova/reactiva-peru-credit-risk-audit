from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_SAMPLE = ROOT / "data" / "sample" / "reactiva_peru_sample.csv"
DATA_FULL = ROOT / "data" / "raw" / "reactiva_peru.csv"
RESULTS_DIR = ROOT / "results"
ASSETS_DIR = ROOT / "assets"

RANDOM_STATE = 42
TEST_SIZE = 0.3
CSV_HEADER_ROW = 2          # for full raw file
SAMPLE_HEADER_ROW = 0       # for sample file

RISK_THRESHOLDS = (4890.2, 11760, 30079.7)

FEATURE_COLS = [
    "SECTOR ECONÓMICO",
    "NOMBRE DE ENTIDAD OTORGANTE DEL CRÉDITO",
    "DEPARTAMENTO",
    "MONTO COBERTURADO (S/)",
]

CAT_COLS = FEATURE_COLS[:3]
AMOUNT_COL = "MONTO COBERTURADO (S/)"
ENTITY_COL = "NOMBRE DE ENTIDAD OTORGANTE DEL CRÉDITO"

PAPER_ALPHAS = {"ridge": 0.00910, "lasso": 0.00038}

FEATURE_NAME_MAP = {
    "SECTOR ECONÓMICO": "Economic Sector",
    "NOMBRE DE ENTIDAD OTORGANTE DEL CRÉDITO": "Credit Granting Entity",
    "DEPARTAMENTO": "Department",
    "MONTO COBERTURADO (S/)": "Covered Amount (S/)",
}


def ensure_output_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)