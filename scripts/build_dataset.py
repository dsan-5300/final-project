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
    # a left join is used to keep all participants with demographics (the base file) and add oral health data where available
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
    "SDDSRVYR", # data release cycle (constant)
    "RIDEXMON", # six-month exam period
    "RIDAGEMN", # age in months (redundant)
    "RIDEXAGM", # age in months at exam (only for <19, and we are using 20+)
    "SIALANG", "SIAPROXY", "SIAINTRP",  # interview language/proxy flags
    "FIALANG", "FIAPROXY", "FIAINTRP",
    "MIALANG", "MIAPROXY", "MIAINTRP",
    "AIALANGA",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only participants who completed interview AND MEC exam
    df = df[df["RIDSTATR"] == 2].copy()
    print(f"  After RIDSTATR==2 filter: {df.shape[0]:,} rows")
    # Drop admin cols
    df = df.drop(columns=[c for c in ADMIN_DROP_COLS if c in df.columns])
    # adults 20+ only because they have stable dentition and relevant data
    df = df[df["RIDAGEYR"] >= 20].copy()
    print(f"  Adults 20+: {df.shape[0]:,} rows, {df.shape[1]} cols")

    # INDFMPIR has near-zero floats so we set to 0 and median-impute missing values
    df["INDFMPIR"] = df["INDFMPIR"].clip(lower=0)
    median_pir = df["INDFMPIR"].median()
    df["INDFMPIR"] = df["INDFMPIR"].fillna(median_pir)
    print(f"  INDFMPIR median imputation value: {median_pir:.3f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # tooth-count, caries, and derived features from raw exam data

    # ── Tooth count features (OHXnnTC — numeric codes) ──
    # 2 / 5 = permanent tooth present, 3 = implant, 4 = missing/extracted
    tc_cols = [c for c in df.columns
               if c.startswith("OHX") and c.endswith("TC") and "CTC" not in c]

    df["teeth_present"] = df[tc_cols].isin([2, 5]).sum(axis=1)
    df["teeth_missing"] = (df[tc_cols] == 4).sum(axis=1)
    df["teeth_implant"] = (df[tc_cols] == 3).sum(axis=1)

    # Coronal caries features
    # P = decayed, J/Q = filled+decayed, F = filled sound,
    # E = missing (caries), S/Z = sound/sealant, R = root tip
    ctc_cols = [c for c in df.columns # only the OHXnnCTC columns, not the OHXnnTC used for tooth counts
                if c.startswith("OHX") and c.endswith("CTC")] # caries status columns

    df["teeth_decayed"] = df[ctc_cols].isin(["P"]).sum(axis=1) # sum of all teeth with untreated decay
    df["teeth_filled_decayed"] = df[ctc_cols].isin(["J", "Q"]).sum(axis=1) # sum of all teeth with treated decay (filled) but still has active decay
    df["teeth_filled_sound"] = df[ctc_cols].isin(["F"]).sum(axis=1) # sum of all teeth with filled restorations that are currently sound (no active decay)
    df["teeth_missing_caries"] = df[ctc_cols].isin(["E"]).sum(axis=1) # sum of all teeth missing due to caries (extracted because of decay, not other reasons)
    df["teeth_sound"] = df[ctc_cols].isin(["S", "Z"]).sum(axis=1) # sum of all teeth that are sound (no decay) or have sealants (preventive, not decayed)
    df["teeth_root_tip"] = df[ctc_cols].isin(["R"]).sum(axis=1) # sum of all teeth with root tips (severe decay leading to root exposure, often a precursor to extraction)

    # summary features
    # Classic DMF Teeth (DMFT) score counts the total number of Decayed, Missing (due to caries), and Filled Teeth
    # Decayed + Missing(caries) + Filled
    df["dmft_score"] = (
        df["teeth_decayed"]
        + df["teeth_filled_decayed"]
        + df["teeth_missing_caries"]
        + df["teeth_filled_sound"]
    )

    # Total active decay (called burden)
    df["total_decay_burden"] = df["teeth_decayed"] + df["teeth_filled_decayed"]

    # Untreated decay ratio
    total_assessed = df[ctc_cols].notna().sum(axis=1)
    df["untreated_decay_ratio"] = (
        df["total_decay_burden"] / total_assessed.replace(0, np.nan)
    )

    # Treatment ratio 
    # access-to-care proxy because it reflects the proportion of teeth that have
    # received treatment out of those that have ever had decay (decayed + filled decayed + filled sound)
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
# TARGET VARS
# ─────────────────────────────────────────────────────────────────────────────

def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    # make binary targets for poor self-rated oral health
    # and clinically needing care, and keep DMFT score as a continuous target

    # Target 1
    # Poor self-rated oral health (OHQ845 >= 4 → Fair/Poor)
    df["target_poor_selfrated"] = (df["OHQ845"] >= 4).astype(int)

    # Target 2
    # Clinically needs dental care (OHAREC < 4 → any referral)
    # .where() preserves NaN for missing OHAREC instead of coding them as 0
    df["target_needs_care"] = (
        (df["OHAREC"] < 4)
        .where(df["OHAREC"].notna())
        .astype("Int64")
    )

    # Target 3 
    # DMFT score
    # already created in feature engineering as df["dmft_score"], so no changes needed here
    pos_self = df["target_poor_selfrated"].mean()
    pos_care = df["target_needs_care"].mean()
    care_missing = df["target_needs_care"].isna().sum()
    print(f"  target_poor_selfrated  positive rate: {pos_self:.3f}")
    print(f"  target_needs_care      positive rate: {pos_care:.3f}  (missing: {care_missing})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# RENAME COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
# AI CODE GENERATION NOTE: AI was used to help generate the static _RENAME_MAP and the regex patterns in _TOOTH_PATTERNS,
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
# END AI CODE

# Regex patterns for tooth level columns
_TOOTH_PATTERNS = {
    # pattern -> prefix template (tooth number)
    r"^OHX(\d{2})TC$": "tooth_{}_status",
    r"^OHX(\d{2})CTC$": "tooth_{}_caries",
    r"^OHX(\d{2})CSC$": "tooth_{}_surface_condition",
    r"^OHX(\d{2})RTC$": "tooth_{}_restoration_type",
    r"^OHX(\d{2})RSC$": "tooth_{}_restoration_surface",
    r"^OHX(\d{2})SE$": "tooth_{}_sealant",
}


def _build_tooth_rename(columns: list[str]) -> dict[str, str]:
    # map names like OHX01TC, OHX01CTC, etc. to tooth_01_status, tooth_01_caries, etc. for all 32 teeth and all relevant column types
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
    # rename the columns to human-readable names using the static map and the tooth-level regex patterns

    # handle some special cases that have multiple versions across files (OHDEXSTS vs OHDEXSTS_x vs OHDEXSTS_y)
    if "OHDEXSTS_x" in df.columns:
        _RENAME_MAP["OHDEXSTS_x"] = "oral_exam_status_dentition"
    if "OHDEXSTS_y" in df.columns:
        _RENAME_MAP["OHDEXSTS_y"] = "oral_exam_status_referral"
    if "OHDEXSTS" in df.columns:
        _RENAME_MAP["OHDEXSTS"] = "oral_exam_status"

    # tooth level names
    tooth_renames = _build_tooth_rename(df.columns.tolist())

    # combine all renames
    full_map = {**_RENAME_MAP, **tooth_renames}

    # drop the original coded columns that we have renamed, but only if they exist in the df
    check_items = [c for c in df.columns if re.match(r"^OHQ(550|592|616|846)$", c)]
    df = df.drop(columns=check_items, errors="ignore")

    # Apply
    applicable = {k: v for k, v in full_map.items() if k in df.columns}
    df = df.rename(columns=applicable)

    # report any columns that still look coded
    still_coded = [c for c in df.columns if re.match(r"^[A-Z]{2,}", c)]
    if still_coded:
        print(f"  Note: {len(still_coded)} columns kept original names: {still_coded[:5]}...")
    else:
        print(f"  All columns renamed to readable names")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
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

    # save csv
    csv_path = os.path.join(output_dir, "nhanes_oral_health_adults.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(f"Final dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # validation
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