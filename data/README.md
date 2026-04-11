# Data — ABIDE-I

This project uses the **Autism Brain Imaging Data Exchange I (ABIDE-I)** dataset.

**No data is included in this repository.** All raw NIfTI files and PNG slices must be
downloaded separately and are subject to the ABIDE-I data sharing agreement.

---

## Dataset Overview

| Field | Value |
|---|---|
| Source | https://fcon_1000.projects.nitrc.org/indi/abide/ |
| Original subjects | 1,112 (573 ASD, 539 TC) |
| After cleaning | 1,067 subjects |
| Sites | 17 international research centres |
| Modality | Structural MRI (sMRI / T1-weighted) |
| Format | NIfTI (.nii.gz) |
| Age range | 7–64 years (median 14.7) |
| Sex | ~85% male (948M / 164F after cleaning) |

---

## Access Instructions

1. Register at https://fcon_1000.projects.nitrc.org/indi/abide/
2. Access is free for research use — requires institutional email and agreement to terms
3. Download the sMRI data for ABIDE-I (total ~50–100GB compressed)
4. Download the phenotypic CSV: `5320_ABIDE_Phenotypics_20230908.csv`
   - Note: Row 0 is a legend row mapping display names to field codes — use `skiprows=1`

---

## Preprocessing Pipeline (B.Tech 2023)

The original pipeline used in the B.Tech work:

```
NIfTI (.nii.gz) → DICOM → PNG slices → label CSVs
```

This produced 100,510 usable PNG slices from 1,067 subjects after removing:
- 19 subjects with missing or unclear sMRI
- 26 subjects with Unknown label in phenotypic file
- Blank/malformed slices identified during preprocessing

### CSV Format

Label CSVs have 4 columns:

| Column | Name | Example |
|---|---|---|
| 0 | ID | 32016 |
| 1 | Image_path | E:\...\32016\32016_088.png |
| 2 | Image_name | 32016_088.png |
| 3 | LABEL | 0 = TC, 1 = ASD |

At runtime, replace the Windows path prefix with the Kaggle or local path:
```python
img_path = img_path.replace(
    "E:\\TARUN\\Projects\\Autism Detection\\Data\\data_png",
    "/kaggle/input/autism/"
).replace("\\", "/")
```

---

## New Pipeline (Graduate System 2026)

The graduate system accepts raw NIfTI files directly — no DICOM conversion required:

```
NIfTI (.nii / .nii.gz) → axial slice extraction (nibabel) → quality filter → inference
```

For the deployed Streamlit app, name NIfTI files with the anonymised subject ID
(e.g. `32016.nii`) to enable automatic phenotypic metadata lookup.

The anonymised ID is derived from the ABIDE-I `ID` field by stripping the 'A' prefix:
- ABIDE ID `A00032016` → file `32016.nii`
- Phenotypic CSV lookup: `df[df['anon_num'] == 32016]`

---

## Subject Cleaning Log

Subjects removed from original 1,112:

| Reason | Count |
|---|---|
| Missing or unclear sMRI | ~19 |
| Unknown label in phenotypic CSV | ~26 |
| **Final subjects** | **1,067** |

---

## Quality Filter Results (Graduate System)

After applying the 4-metric quality filter (see `src/quality_filter.py`):

| Split | Before | After | Removed |
|---|---|---|---|
| Train | 72,367 | 67,625 | 4,742 (6.6%) |
| Val | 8,041 | 7,507 | 534 (6.6%) |
| Test | 20,102 | 18,814 | 1,288 (6.4%) |

Consistent rejection rate across splits confirms the filter is not introducing split-specific bias.

---

## Phenotypic CSV Field Reference

Key fields used by this project (field names follow ABIDE-I coding):

| Field | Meaning | Values |
|---|---|---|
| ID | Anonymised subject ID | A00032016 |
| ABIDE_01 | SubjectID (numeric) | 50682 |
| ABIDE_02 | DX Group | 1=ASD, 2=TC |
| ABIDE_04 | Age at scan | float (years) |
| ABIDE_05 | Sex | 1=Male, 2=Female |
| SUB_TYPE | Subject type label | CONTROL, PATIENT-AUT |

Note: The CSV has a legend row at row 0 — always load with `pd.read_csv(path, skiprows=1)`.
