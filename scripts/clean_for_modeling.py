
import pandas as pd
import numpy as np

INPUT  = "data/processed/nhanes_oral_health_adults.csv"
OUTPUT = "data/processed/nhanes_model_ready.csv"

print("Loading data...")
df = pd.read_csv(INPUT, low_memory=False) # low memory to avoid dtype warnings on wide datasets

# Drop structurally empty columns
# Sealant columns
# 100% NaN bc adults don't receive sealants in this dataset
sealant_cols = [c for c in df.columns if c.endswith("_sealant")]

# restoration type and surface columns are 99% NaN
restoration_cols = [c for c in df.columns if "_restoration_type" in c or "_restoration_surface" in c]

# all-NaN columns identified in EDA
other_empty = [
    "education_youth",
    "brushing_frequency_flag",
    "brushing_times_per_day",
    "toothpaste_amount",
    "age_started_brushing_flag",
    "age_started_brushing_value",
    "age_started_brushing_unit",
    "age_started_toothpaste_flag",
    "age_started_toothpaste_value",
    "age_started_toothpaste_unit",
    "age_stopped_rx_fluoride_flag",
    "age_stopped_rx_fluoride_value",
    "age_stopped_rx_fluoride_unit",
]
# surface condition columns are also mostly empty for most teeth
surface_cols = [c for c in df.columns if c.endswith("_surface_condition")]

drop_cols = sealant_cols + restoration_cols + other_empty + surface_cols
drop_cols = [c for c in drop_cols if c in df.columns]
df.drop(columns=drop_cols, inplace=True)

# Drop redundant engineered feature
redundant = ["total_decay_burden", "teeth_missing", "teeth_filled_decayed", "teeth_root_tip"]
redundant = [c for c in redundant if c in df.columns]
df.drop(columns=redundant, inplace=True)

# Re-code barrier columns as for binary flags
# 0 = either didn't have unmet need, or had unmet need but didn't cite this barrier
barrier_cols_raw = [c for c in df.columns if c.startswith("barrier_")]
for col in barrier_cols_raw:
    df[col] = df[col].notna().astype(int)

# Handle edentulous cases for treatment_ratio
# no active teeth means no treatment burden
df["is_edentulous"] = (df["teeth_present"] == 0).astype(int)
df["treatment_ratio"] = df["treatment_ratio"].fillna(0.0)

# transform income_poverty_ratio
# Near-zero floating points from original median imputation
# Hard ceiling at 5.0 is imposed by NHANES
# Log-transform compresses the range and handles the ceiling more gracefully in linear models
# We keep the original column AND add a log-transformed version
df["income_poverty_ratio"] = df["income_poverty_ratio"].clip(lower=0.05)
df["log_income_poverty_ratio"] = np.log(df["income_poverty_ratio"])

# handle target_needs_care missingness 
# 208 rows where the clinical examiner recommendation was not recorded (OHAREC missing).
# We preserve the full dataset and let downstream model scripts filter as needed.
# Add a flag so downstream code can easily split.
df["needs_care_observed"] = df["target_needs_care"].notna().astype(int)
n_missing_target = df["target_needs_care"].isna().sum()
print(f"  Added needs_care_observed flag ({n_missing_target} rows have NaN target_needs_care)")

# clean up bad codes in categorical columns 
# education_adult codes 7 (Refused) and 9 (Don't know) in 12 rows set to NaN
if "education_adult" in df.columns:
    df["education_adult"] = df["education_adult"].replace({7.0: np.nan, 9.0: np.nan})

# marital_status code 77 (Refused) in 5 rows set to NaN
if "marital_status" in df.columns:
    df["marital_status"] = df["marital_status"].replace({77.0: np.nan})

# self_rated_oral_health code 9 (Don't know) in 7 rows set to NaN
if "self_rated_oral_health" in df.columns:
    df["self_rated_oral_health"] = df["self_rated_oral_health"].replace({9.0: np.nan})

# last_dental_visit code 99 (Don't know) in 18 rows set to NaN
if "last_dental_visit" in df.columns:
    df["last_dental_visit"] = df["last_dental_visit"].replace({99.0: np.nan})

# Drop admin / survey infrastructure columns
# These are not predictive features
admin_cols = [
    "interview_exam_status",
    "oral_exam_status_dentition",
    "dentition_exam_status",
    "oral_exam_status_referral",
    "recommendation_exam_status",
    "exam_position_recumbent",
    "household_ref_gender",
    "household_ref_age",
    "household_ref_education",
    "household_ref_marital_status",
    "household_ref_spouse_education",
]
admin_cols = [c for c in admin_cols if c in df.columns]
df.drop(columns=admin_cols, inplace=True)

# summary of final shape and missingness
print(f"\nFinal shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# remaining missingness on key features
key_cols = [
    "age", "gender", "race_ethnicity", "education_adult", "income_poverty_ratio",
    "last_dental_visit", "dmft_score", "teeth_present", "teeth_decayed",
    "teeth_filled_sound", "untreated_decay_ratio", "treatment_ratio",
    "self_rated_oral_health", "needed_dental_care_couldnt_get",
    "target_poor_selfrated", "target_needs_care",
]
key_cols = [c for c in key_cols if c in df.columns]
missing_summary = df[key_cols].isnull().sum()
missing_summary = missing_summary[missing_summary > 0]
if len(missing_summary):
    print("\nRemaining missingness in key analytic columns:")
    for col, n in missing_summary.items():
        print(f"  {col}: {n} NaN ({n/len(df):.1%})")
else:
    print("\nNo missingness in key analytic columns.")

print(f"\nTarget rates:")
print(f"target_poor_selfrated: {df['target_poor_selfrated'].mean():.3f}")
print(f"target_needs_care: {df['target_needs_care'].mean():.3f} (excludes {n_missing_target} NaN rows)")

# save
df.to_csv(OUTPUT, index=False)
print(f"\nSaved to {OUTPUT}")

#BEGIN AI CODE
# column manifest
print("\n=== COLUMN MANIFEST ===")
groups = {
    "Survey Design (keep for weighted inference)": [
        "interview_weight_2yr", "exam_weight_2yr", "variance_psu", "variance_stratum"
    ],
    "Demographics": [
        "age", "gender", "race_ethnicity", "race_ethnicity_detailed",
        "education_adult", "marital_status", "country_of_birth",
        "citizenship_status", "years_in_us", "served_active_military",
        "served_in_foreign_country", "pregnancy_status",
    ],
    "Household / SES": [
        "household_size", "family_size", "num_children_under_5",
        "num_children_6_to_17", "num_adults_60_plus",
        "household_income", "family_income",
        "income_poverty_ratio", "log_income_poverty_ratio",
    ],
    "Tooth Status (32 raw columns)": [c for c in df.columns if c.endswith("_status") and "tooth" in c],
    "Caries by Tooth (28 raw columns)": [c for c in df.columns if c.endswith("_caries")],
    "Engineered Clinical Features": [
        "teeth_present", "teeth_implant", "teeth_decayed",
        "teeth_filled_sound", "teeth_missing_caries", "teeth_sound",
        "dmft_score", "untreated_decay_ratio", "treatment_ratio",
        "is_edentulous", "root_caries_present", "other_root_lesion",
        "root_caries_restored", "other_root_lesion_restored", "has_dental_implant",
    ],
    "Oral Health Questionnaire": [
        "last_dental_visit", "reason_last_dental_visit",
        "needed_dental_care_couldnt_get", "self_rated_oral_health",
        "thinks_has_gum_disease", "floss_days_past_week",
        "had_gum_disease_treatment", "told_bone_loss_around_teeth",
        "had_oral_cancer_exam", "when_last_oral_cancer_exam",
        "freq_mouth_aching_past_year", "freq_job_difficulty_from_mouth",
        "freq_embarrassed_from_mouth",
        "received_rx_fluoride", "told_benefits_quit_cigarettes",
        "told_benefits_check_blood_sugar", "told_importance_cancer_check",
    ],
    "Barriers (binary flags)": [c for c in df.columns if c.startswith("barrier_")],
    "Clinical Referrals": [
        "overall_care_recommendation", "referral_decayed_teeth",
        "referral_gum_disease", "referral_oral_hygiene", "referral_soft_tissue",
        "referral_denture", "no_significant_findings", "referral_other_finding",
    ],
    "Targets & Flags": [
        "target_poor_selfrated", "target_needs_care", "needs_care_observed",
    ],
}

total_accounted = 0
for group, cols in groups.items():
    present = [c for c in cols if c in df.columns]
    print(f"  {group}: {len(present)} cols")
    total_accounted += len(present)

unaccounted = [c for c in df.columns if c not in [col for cols in groups.values() for col in cols]]
if unaccounted:
    print(f"  Other / misc: {len(unaccounted)} cols")
    for c in unaccounted:
        print(f"    {c}")

# END AI CODE