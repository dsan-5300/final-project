"""
  - DEMO_J.xpt: Demographics
  - OHXDEN_J.xpt: Oral Health Dentition Exam
  - OHQ_J.xpt: Oral Health Questionnaire
  - OHXREF_J.xpt: Oral Health Referral / Recommendation
"""

import os
import re
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────
# DOWNLOAD AND MERGE NHANES FILES
# ────────────────────────────────────────────────────────────────────────────

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"

FILES = {
    "demo":   "DEMO_J.xpt",
    "ohxden": "OHXDEN_J.xpt",
    "ohq":    "OHQ_J.xpt",
    "ohxref": "OHXREF_J.xpt",
}


def download_nhanes() -> pd.DataFrame:
    """Download all four NHANES files and left-join on SEQN."""
    dfs = {}
    for name, filename in FILES.items():
        url = BASE_URL + filename
        print(f"  Downloading {filename}...")
        dfs[name] = pd.read_sas(url, format="xport", encoding="utf-8")
        print(f"    -> {dfs[name].shape[0]:,} rows, {dfs[name].shape[1]} cols")

    # Left join from demographics as anchor
    df = dfs["demo"]
    for name in ["ohxden", "ohq", "ohxref"]:
        df = df.merge(dfs[name], on="SEQN", how="left")

    print(f"  Merged: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLEAN
# ─────────────────────────────────────────────────────────────────────────────

ADMIN_DROP_COLS = [
    "SDDSRVYR",   # data release cycle (constant for this file)
    "RIDEXMON",   # six-month exam period
    "RIDAGEMN",   # age in months (redundant with RIDAGEYR for adults)
    "RIDEXAGM",   # age in months at exam (only for <19)
    "SIALANG", "SIAPROXY", "SIAINTRP",   # interview language/proxy flags
    "FIALANG", "FIAPROXY", "FIAINTRP",
    "MIALANG", "MIAPROXY", "MIAINTRP",
    "AIALANGA",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to examined adults 20+, drop admin cols, impute PIR."""

    # Keep only participants who completed interview AND MEC exam
    df = df[df["RIDSTATR"] == 2].copy()
    print(f"  After RIDSTATR==2 filter: {df.shape[0]:,} rows")

    # Drop administrative columns
    df = df.drop(columns=[c for c in ADMIN_DROP_COLS if c in df.columns])

    # Restrict to adults 20+
    df = df[df["RIDAGEYR"] >= 20].copy()
    print(f"  Adults 20+: {df.shape[0]:,} rows, {df.shape[1]} cols")

    # INDFMPIR: clip near-zero float artifacts to 0, median-impute missing
    df["INDFMPIR"] = df["INDFMPIR"].clip(lower=0)
    median_pir = df["INDFMPIR"].median()
    df["INDFMPIR"] = df["INDFMPIR"].fillna(median_pir)
    print(f"  INDFMPIR median imputation value: {median_pir:.3f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build tooth-count, caries, and derived features from raw exam data."""

    # ── Tooth count features (OHXnnTC — numeric codes) ──
    # 2 / 5 = permanent tooth present, 3 = implant, 4 = missing/extracted
    tc_cols = [c for c in df.columns
               if c.startswith("OHX") and c.endswith("TC") and "CTC" not in c]

    df["teeth_present"] = df[tc_cols].isin([2, 5]).sum(axis=1)
    df["teeth_missing"] = (df[tc_cols] == 4).sum(axis=1)
    df["teeth_implant"] = (df[tc_cols] == 3).sum(axis=1)

    # ── Coronal caries features (OHXnnCTC — letter codes) ──
    # P = decayed, J/Q = filled+decayed, F = filled sound,
    # E = missing (caries), S/Z = sound/sealant, R = root tip
    ctc_cols = [c for c in df.columns
                if c.startswith("OHX") and c.endswith("CTC")]

    df["teeth_decayed"]        = df[ctc_cols].isin(["P"]).sum(axis=1)
    df["teeth_filled_decayed"] = df[ctc_cols].isin(["J", "Q"]).sum(axis=1)
    df["teeth_filled_sound"]   = df[ctc_cols].isin(["F"]).sum(axis=1)
    df["teeth_missing_caries"] = df[ctc_cols].isin(["E"]).sum(axis=1)
    df["teeth_sound"]          = df[ctc_cols].isin(["S", "Z"]).sum(axis=1)
    df["teeth_root_tip"]       = df[ctc_cols].isin(["R"]).sum(axis=1)

    # ── Derived summary features ──
    # Classic DMFT: Decayed + Missing(caries) + Filled
    df["dmft_score"] = (
        df["teeth_decayed"]
        + df["teeth_filled_decayed"]
        + df["teeth_missing_caries"]
        + df["teeth_filled_sound"]
    )

    # Total active decay burden
    df["total_decay_burden"] = df["teeth_decayed"] + df["teeth_filled_decayed"]

    # Untreated decay ratio
    total_assessed = df[ctc_cols].notna().sum(axis=1)
    df["untreated_decay_ratio"] = (
        df["total_decay_burden"] / total_assessed.replace(0, np.nan)
    )

    # Treatment ratio (access-to-care proxy)
    ever_decayed = (
        df["teeth_decayed"]
        + df["teeth_filled_decayed"]
        + df["teeth_filled_sound"]
    )
    df["treatment_ratio"] = (
        df["teeth_filled_sound"] / ever_decayed.replace(0, np.nan)
    )

    print(f"  Engineered 13 dental features")
    print(f"  Participants with active decay: {(df['teeth_decayed'] > 0).sum():,}")
    print(f"  Mean DMFT score: {df['dmft_score'].mean():.2f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. TARGET VARIABLES
# ─────────────────────────────────────────────────────────────────────────────

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary and continuous target variables."""

    # Target 1 — Poor self-rated oral health (OHQ845 >= 4 → Fair/Poor)
    df["target_poor_selfrated"] = (df["OHQ845"] >= 4).astype(int)

    # Target 2 — Clinically needs dental care (OHAREC < 4 → any referral)
    # .where() preserves NaN for missing OHAREC instead of coding them as 0
    df["target_needs_care"] = (
        (df["OHAREC"] < 4)
        .where(df["OHAREC"].notna())
        .astype("Int64")
    )

    # Target 3 — DMFT score (already created in feature engineering)

    pos_self = df["target_poor_selfrated"].mean()
    pos_care = df["target_needs_care"].mean()
    care_missing = df["target_needs_care"].isna().sum()
    print(f"  target_poor_selfrated  positive rate: {pos_self:.3f}")
    print(f"  target_needs_care      positive rate: {pos_care:.3f}  (missing: {care_missing})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. RENAME COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

# Static mapping for non-tooth-level columns
_RENAME_MAP = {
    # ── Demographics ──
    "SEQN":      "participant_id",
    "RIDSTATR":  "interview_exam_status",
    "RIAGENDR":  "gender",
    "RIDAGEYR":  "age",
    "RIDRETH1":  "race_ethnicity",
    "RIDRETH3":  "race_ethnicity_detailed",
    "RIDEXPRG":  "pregnancy_status",
    "DMQMILIZ":  "served_active_military",
    "DMQADFC":   "served_in_foreign_country",
    "DMDBORN4":  "country_of_birth",
    "DMDCITZN":  "citizenship_status",
    "DMDYRSUS":  "years_in_us",
    "DMDEDUC3":  "education_youth",
    "DMDEDUC2":  "education_adult",
    "DMDMARTL":  "marital_status",
    "DMDHHSIZ":  "household_size",
    "DMDFMSIZ":  "family_size",
    "DMDHHSZA":  "num_children_under_5",
    "DMDHHSZB":  "num_children_6_to_17",
    "DMDHHSZE":  "num_adults_60_plus",
    "DMDHRGND":  "household_ref_gender",
    "DMDHRAGZ":  "household_ref_age",
    "DMDHREDZ":  "household_ref_education",
    "DMDHRMAZ":  "household_ref_marital_status",
    "DMDHSEDZ":  "household_ref_spouse_education",
    "WTINT2YR":  "interview_weight_2yr",
    "WTMEC2YR":  "exam_weight_2yr",
    "SDMVPSU":   "variance_psu",
    "SDMVSTRA":  "variance_stratum",
    "INDHHIN2":  "household_income",
    "INDFMIN2":  "family_income",
    "INDFMPIR":  "income_poverty_ratio",

    # ── Oral Health Dentition — summary variables ──
    "OHDDESTS":    "dentition_exam_status",
    "OHXIMP":      "has_dental_implant",
    "OHXRCAR":     "root_caries_present",
    "OHXRCARO":    "other_root_lesion",
    "OHXRRES":     "root_caries_restored",
    "OHXRRESO":    "other_root_lesion_restored",

    # ── Oral Health Questionnaire ──
    "OHQ030":  "last_dental_visit",
    "OHQ033":  "reason_last_dental_visit",
    "OHQ770":  "needed_dental_care_couldnt_get",
    "OHQ780A": "barrier_cost",
    "OHQ780B": "barrier_didnt_want_to_spend",
    "OHQ780C": "barrier_insurance_no_coverage",
    "OHQ780D": "barrier_office_too_far",
    "OHQ780E": "barrier_inconvenient_hours",
    "OHQ780F": "barrier_other_dentist_advised_against",
    "OHQ780G": "barrier_afraid_of_dentist",
    "OHQ780H": "barrier_cant_take_time_off_work",
    "OHQ780I": "barrier_too_busy",
    "OHQ780J": "barrier_expected_problem_to_resolve",
    "OHQ780K": "barrier_other",
    "OHQ555G": "age_started_brushing_flag",
    "OHQ555Q": "age_started_brushing_value",
    "OHQ555U": "age_started_brushing_unit",
    "OHQ560G": "age_started_toothpaste_flag",
    "OHQ560Q": "age_started_toothpaste_value",
    "OHQ560U": "age_started_toothpaste_unit",
    "OHQ566":  "received_rx_fluoride",
    "OHQ571Q": "age_started_rx_fluoride_value",
    "OHQ571U": "age_started_rx_fluoride_unit",
    "OHQ576G": "age_stopped_rx_fluoride_flag",
    "OHQ576Q": "age_stopped_rx_fluoride_value",
    "OHQ576U": "age_stopped_rx_fluoride_unit",
    "OHQ610":  "told_benefits_quit_cigarettes",
    "OHQ612":  "told_benefits_check_blood_sugar",
    "OHQ614":  "told_importance_cancer_check",
    "OHQ620":  "freq_mouth_aching_past_year",
    "OHQ640":  "freq_job_difficulty_from_mouth",
    "OHQ680":  "freq_embarrassed_from_mouth",
    "OHQ835":  "thinks_has_gum_disease",
    "OHQ845":  "self_rated_oral_health",
    "OHQ848G": "brushing_frequency_flag",
    "OHQ848Q": "brushing_times_per_day",
    "OHQ849":  "toothpaste_amount",
    "OHQ850":  "had_gum_disease_treatment",
    "OHQ860":  "told_bone_loss_around_teeth",
    "OHQ870":  "floss_days_past_week",
    "OHQ880":  "had_oral_cancer_exam",
    "OHQ895":  "when_last_oral_cancer_exam",
    "OHQ900":  "oral_cancer_exam_provider_type",

    # ── Oral Health Referral ──
    "OHDRCSTS":  "recommendation_exam_status",
    "OHAREC":    "overall_care_recommendation",
    "OHAROCDT":  "referral_decayed_teeth",
    "OHAROCGP":  "referral_gum_disease",
    "OHAROCOH":  "referral_oral_hygiene",
    "OHAROCCI":  "referral_soft_tissue",
    "OHAROCDE":  "referral_denture",
    "OHARNF":    "no_significant_findings",
    "OHAROTH":   "referral_other_finding",
    "OHAPOS":    "exam_position_recumbent",
}

# Regex patterns for tooth-level columns (handled programmatically)
_TOOTH_PATTERNS = {
    # pattern             →  prefix template (tooth number inserted)
    r"^OHX(\d{2})TC$":      "tooth_{}_status",
    r"^OHX(\d{2})CTC$":     "tooth_{}_caries",
    r"^OHX(\d{2})CSC$":     "tooth_{}_surface_condition",
    r"^OHX(\d{2})RTC$":     "tooth_{}_restoration_type",
    r"^OHX(\d{2})RSC$":     "tooth_{}_restoration_surface",
    r"^OHX(\d{2})SE$":      "tooth_{}_sealant",
}


def _build_tooth_rename(columns: list[str]) -> dict[str, str]:
    """Generate readable names for tooth-level OHX columns."""
    mapping = {}
    for col in columns:
        for pattern, template in _TOOTH_PATTERNS.items():
            m = re.match(pattern, col)
            if m:
                tooth_num = m.group(1)  # keep zero-padded "01"-"32"
                mapping[col] = template.format(tooth_num)
                break
    return mapping


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename all NHANES coded columns to readable English names."""

    # Handle OHDEXSTS which may appear with merge suffixes
    if "OHDEXSTS_x" in df.columns:
        _RENAME_MAP["OHDEXSTS_x"] = "oral_exam_status_dentition"
    if "OHDEXSTS_y" in df.columns:
        _RENAME_MAP["OHDEXSTS_y"] = "oral_exam_status_referral"
    if "OHDEXSTS" in df.columns:
        _RENAME_MAP["OHDEXSTS"] = "oral_exam_status"

    # Build tooth-level renames
    tooth_renames = _build_tooth_rename(df.columns.tolist())

    # Combine all renames
    full_map = {**_RENAME_MAP, **tooth_renames}

    # Drop routing/check-item columns that have no analytical value
    check_items = [c for c in df.columns if re.match(r"^OHQ(550|592|616|846)$", c)]
    df = df.drop(columns=check_items, errors="ignore")

    # Apply renames (only for columns that exist)
    applicable = {k: v for k, v in full_map.items() if k in df.columns}
    df = df.rename(columns=applicable)

    # Report any columns that didn't get renamed (shouldn't happen, but good to know)
    still_coded = [c for c in df.columns if re.match(r"^[A-Z]{2,}", c)]
    if still_coded:
        print(f"  Note: {len(still_coded)} columns kept original names: {still_coded[:5]}...")
    else:
        print(f"  All columns renamed to readable names")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)

    print("Step 1/5: Downloading NHANES data...")
    df = download_nhanes()

    print("\nStep 2/5: Cleaning...")
    df = clean(df)

    print("\nStep 3/5: Engineering features...")
    df = engineer_features(df)

    print("\nStep 4/5: Creating target variables...")
    df = create_targets(df)

    print("\nStep 5/5: Renaming columns...")
    df = rename_columns(df)

    # Save
    csv_path = os.path.join(output_dir, "nhanes_oral_health_adults.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(f"Final dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Quick validation
    expected_features = [
        "teeth_present", "teeth_missing", "teeth_implant", "teeth_sound",
        "teeth_decayed", "teeth_filled_decayed", "teeth_filled_sound",
        "teeth_missing_caries", "teeth_root_tip",
        "dmft_score", "total_decay_burden", "untreated_decay_ratio",
        "treatment_ratio", "target_poor_selfrated", "target_needs_care",
    ]
    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        print(f"WARNING — missing expected columns: {missing}")
    else:
        print(f"All {len(expected_features)} engineered features/targets present")


if __name__ == "__main__":
    main()
