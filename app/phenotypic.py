# phenotypic.py
# Load and query the ABIDE-I phenotypic CSV for a given subject

import pandas as pd
import os

PHENO_PATH = '5320_ABIDE_Phenotypics_20230908.csv'

# ABIDE-I site ranges (SubID → site name)
SITE_RANGES = [
    ('CALTECH',  51456, 51490), ('CMU',      51540, 51575),
    ('KKI',      50801, 50830), ('LEUVEN_1', 50501, 50535),
    ('LEUVEN_2', 50601, 50625), ('MAX_MUN',  51301, 51340),
    ('NYU',      50681, 50780), ('OHSU',     51101, 51160),
    ('OLIN',     51201, 51262), ('PITT',     51001, 51062),
    ('SBL',      51401, 51432), ('SDSU',     51501, 51538),
    ('STANFORD', 51601, 51610), ('TRINITY',  51701, 51762),
    ('UCLA_1',   50901, 50940), ('UCLA_2',   51801, 51840),
    ('UM_1',     50101, 50200), ('UM_2',     50201, 50280),
    ('USM',      50401, 50465), ('YALE',     50961, 51005),
]

# Site-level performance from your validation results
SITE_PERFORMANCE = {
    'NYU':     {'Sensitivity': 0.9517, 'Specificity': 0.9691, 'AUC': 0.9910, 'N_subjects': 73},
    'UM_1':    {'Sensitivity': 0.9846, 'Specificity': 0.9796, 'AUC': 0.9974, 'N_subjects': 81},
    'UM_2':    {'Sensitivity': 0.9024, 'Specificity': 0.9852, 'AUC': 0.9912, 'N_subjects': 61},
    'OLIN':    {'Sensitivity': 0.9795, 'Specificity': 0.9961, 'AUC': 0.9990, 'N_subjects': 54},
    'OHSU':    {'Sensitivity': 1.0000, 'Specificity': 0.9504, 'AUC': 1.0000, 'N_subjects': 55},
    'PITT':    {'Sensitivity': 0.8854, 'Specificity': 0.9561, 'AUC': 0.9761, 'N_subjects': 56},
    'USM':     {'Sensitivity': 0.9636, 'Specificity': 0.9922, 'AUC': 0.9985, 'N_subjects': 59},
}


def _sub_to_site(sub_id: int) -> str:
    for site, lo, hi in SITE_RANGES:
        if lo <= sub_id <= hi:
            return site
    return 'Unknown'


def load_phenotypic() -> pd.DataFrame | None:
    """Load and parse the phenotypic CSV. Returns None if file not found."""
    if not os.path.exists(PHENO_PATH):
        return None
    df = pd.read_csv(PHENO_PATH, skiprows=1)
    df['anon_num'] = df['ID'].str.replace('A', '', regex=False).astype(int, errors='ignore')
    df['site']     = df['ABIDE_01'].apply(_sub_to_site)
    return df


def get_subject_info(anon_num: int, pheno_df: pd.DataFrame) -> dict | None:
    """
    Look up phenotypic info for a subject by their anonymised number
    (= PNG folder name = Anonymized ID with 'A' stripped).

    Returns dict with cleaned fields, or None if not found.
    """
    row = pheno_df[pheno_df['anon_num'] == anon_num]
    if len(row) == 0:
        return None

    row = row.iloc[0]

    # Decode fields
    dx_group = int(row['ABIDE_02']) if pd.notna(row['ABIDE_02']) else None
    sex_code  = int(row['ABIDE_05']) if pd.notna(row['ABIDE_05']) else None
    age       = float(row['ABIDE_04']) if pd.notna(row['ABIDE_04']) else None
    sub_id    = int(row['ABIDE_01'])   if pd.notna(row['ABIDE_01']) else None
    site      = str(row['site'])
    sub_type  = str(row['SUB_TYPE'])   if pd.notna(row['SUB_TYPE']) else 'Unknown'

    true_label = {1: 'ASD', 2: 'TC'}.get(dx_group, 'Unknown')
    sex_label  = {1: 'Male', 2: 'Female'}.get(sex_code, 'Unknown')

    site_perf = SITE_PERFORMANCE.get(site, None)

    return {
        'anon_id'       : int(anon_num),
        'sub_id'        : sub_id,
        'true_label'    : true_label,
        'sex'           : sex_label,
        'age'           : round(age, 1) if age else None,
        'site'          : site,
        'sub_type'      : sub_type,
        'site_perf'     : site_perf,
    }


def extract_anon_num_from_filename(filename: str) -> int | None:
    """
    Try to extract the anonymised number from a NIfTI filename.
    e.g. '32016.nii' → 32016
         'subject_32016.nii.gz' → 32016
    """
    import re
    name = filename.replace('.nii.gz', '').replace('.nii', '')
    digits = re.findall(r'\d{4,}', name)   # 4+ digit numbers
    return int(digits[0]) if digits else None