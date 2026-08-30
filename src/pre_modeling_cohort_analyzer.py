"""
Deterministic Pre-Modeling Cohort Analysis & Heuristic Predictor.

Purpose:
    Demonstrates how historical domain knowledge (exact Ticket & Family cohorts)
    can be used to audit and predict test outcomes deterministically BEFORE
    running any complex Machine Learning models.

Key Principles:
    1. Group Survival Invariance: Families/travel units on the Titanic largely survived
       or perished as unified cohorts.
    2. Zero Leakage: Only ground-truth labels from the Train dataset are used to derive
       consensus rules; no Test labels are assumed.
    3. Hierarchical Fallback:
       - Tier 1: Exact Ticket match (shared booking / same lifeboat access)
       - Tier 2: Exact Surname + Pclass match (family lineage within same deck class)
       - Tier 3: Unambiguous demographic baselines (1st/2nd class women & children vs. 3rd class men)
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_raw_data(data_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load raw train and test CSV records."""
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    
    with open(train_path, mode="r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    with open(test_path, mode="r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
        
    return train_rows, test_rows


def extract_passenger_attributes(row: Dict[str, str]) -> Dict[str, Any]:
    """Extract standardized demographic and group keys from a raw record."""
    pid = int(row["PassengerId"])
    ticket = row["Ticket"].strip()
    pclass = int(row["Pclass"])
    sex = row["Sex"].lower().strip()
    fare = float(row["Fare"]) if row["Fare"] and row["Fare"].strip() else 0.0
    embarked = row["Embarked"].strip() if row["Embarked"] else "S"
    
    # Extract title from name: "Surname, Title. Firstname..."
    name = row["Name"]
    surname = name.split(",")[0].strip()
    title_part = name.split(",")[1].split(".")[0].strip() if "," in name and "." in name else "Mr"
    
    # Handle Age with safe fallback for demographic tagging
    age_str = row["Age"].strip() if row["Age"] else ""
    age = float(age_str) if age_str else 28.0
    
    # Woman / Child / Master indicator (High historical lifeboat priority)
    is_wc = (sex == "female") or (title_part == "Master") or (age <= 14.0)
    
    return {
        "pid": pid,
        "name": name,
        "surname": surname,
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "title": title_part,
        "ticket": ticket,
        "fare": fare,
        "embarked": embarked,
        "is_wc": is_wc,
        "family_class_key": f"{surname}_{pclass}",
    }


def build_training_cohort_knowledge_base(
    train_records: List[Dict[str, str]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Builds lookup registries of known survival outcomes for shared Tickets and Families.
    """
    ticket_registry = {}
    family_registry = {}
    
    for r in train_records:
        attr = extract_passenger_attributes(r)
        survived = int(r["Survived"])
        attr["survived"] = survived
        
        # 1. Register under Ticket
        t_key = attr["ticket"]
        ticket_registry.setdefault(t_key, []).append(attr)
        
        # 2. Register under Family (Surname + Class)
        f_key = attr["family_class_key"]
        family_registry.setdefault(f_key, []).append(attr)
        
    return ticket_registry, family_registry


def evaluate_cohort_consensus(members: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Determines if a training cohort had unanimous (100% or 0%) survival.
    """
    wc_members = [m for m in members if m["is_wc"]]
    all_surv = [m["survived"] for m in members]
    wc_surv = [m["survived"] for m in wc_members]
    
    return {
        "total_members": len(members),
        "wc_members_count": len(wc_members),
        "wc_unanimous_survived": len(wc_surv) > 0 and sum(wc_surv) == len(wc_surv),
        "wc_unanimous_perished": len(wc_surv) > 0 and sum(wc_surv) == 0,
        "all_unanimous_survived": sum(all_surv) == len(all_surv),
        "all_unanimous_perished": sum(all_surv) == 0,
        "wc_survival_rate": sum(wc_surv) / len(wc_surv) if wc_surv else None,
        "overall_survival_rate": sum(all_surv) / len(all_surv),
    }


def deterministic_pre_predict(
    train_records: List[Dict[str, str]],
    test_records: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """
    Applies hierarchical domain rules to make deterministic predictions for test records.
    """
    ticket_reg, family_reg = build_training_cohort_knowledge_base(train_records)
    predictions = []
    
    for r in test_records:
        attr = extract_passenger_attributes(r)
        pid = attr["pid"]
        t_key = attr["ticket"]
        f_key = attr["family_class_key"]
        is_wc = attr["is_wc"]
        
        pred = None
        confidence = None
        rule_applied = None
        
        # TIER 1: Exact Ticket Cohort Consensus (Highest Precision)
        if t_key in ticket_reg:
            consensus = evaluate_cohort_consensus(ticket_reg[t_key])
            
            if is_wc and consensus["wc_unanimous_survived"]:
                pred = 1
                confidence = "HIGH (100% Ticket WC Consensus)"
                rule_applied = f"Ticket {t_key}: All {consensus['wc_members_count']} train female/child peers survived."
                
            elif is_wc and consensus["wc_unanimous_perished"]:
                pred = 0
                confidence = "HIGH (100% Ticket WC Consensus)"
                rule_applied = f"Ticket {t_key}: All {consensus['wc_members_count']} train female/child peers perished."
                
            elif not is_wc and consensus["all_unanimous_survived"] and consensus["total_members"] >= 2:
                pred = 1
                confidence = "HIGH (100% Full Ticket Consensus)"
                rule_applied = f"Ticket {t_key}: 100% of entire traveling group ({consensus['total_members']} members) survived."

        # TIER 2: Family Surname + Pclass Consensus
        if pred is None and f_key in family_reg and is_wc:
            consensus = evaluate_cohort_consensus(family_reg[f_key])
            if consensus["wc_unanimous_survived"]:
                pred = 1
                confidence = "HIGH (100% Family WC Consensus)"
                rule_applied = f"Family {f_key}: All {consensus['wc_members_count']} train female/child peers survived."
            elif consensus["wc_unanimous_perished"]:
                pred = 0
                confidence = "HIGH (100% Family WC Consensus)"
                rule_applied = f"Family {f_key}: All {consensus['wc_members_count']} train female/child peers perished."

        # TIER 3: Unambiguous Demographic Baseline
        if pred is None:
            if attr["sex"] == "female" and attr["pclass"] in [1, 2]:
                pred = 1
                confidence = "MEDIUM (1st/2nd Class Female Baseline)"
                rule_applied = f"Class {attr['pclass']} females had >92% survival rate."
                
            elif not is_wc and attr["pclass"] in [2, 3]:
                pred = 0
                confidence = "MEDIUM (2nd/3rd Class Adult Male Baseline)"
                rule_applied = f"Class {attr['pclass']} adult males had <13% survival rate."
                
            else:
                pred = 1 if attr["pclass"] == 1 else 0
                confidence = "AMBIGUOUS (Requires ML Classifier)"
                rule_applied = f"Borderline demographic cohort (Pclass={attr['pclass']}, Sex={attr['sex']}, Age={attr['age']})."

        predictions.append({
            "PassengerId": pid,
            "Name": attr["name"],
            "Pclass": attr["pclass"],
            "Sex": attr["sex"],
            "Age": attr["age"],
            "Ticket": attr["ticket"],
            "Deterministic_Prediction": pred,
            "Confidence": confidence,
            "Explanation": rule_applied,
        })
        
    return predictions


def main():
    """Execute analysis and display pre-model insights."""
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "raw"
    
    train_rows, test_rows = load_raw_data(data_dir)
    results = deterministic_pre_predict(train_rows, test_rows)
    
    high_conf = [r for r in results if "HIGH" in r["Confidence"]]
    med_conf = [r for r in results if "MEDIUM" in r["Confidence"]]
    ambiguous = [r for r in results if "AMBIGUOUS" in r["Confidence"]]
    
    total_surv = sum(r["Deterministic_Prediction"] for r in results)
    
    print("=" * 80)
    print("      DETERMINISTIC PRE-MODELING COHORT ANALYSIS & AUDIT REPORT")
    print("=" * 80)
    print(f"Total Test Passengers Audited: {len(results)}")
    print(f"  • High-Confidence (Direct Family/Ticket Overlaps): {len(high_conf)} ({len(high_conf)/len(results)*100:.1f}%)")
    print(f"  • Medium-Confidence (Clear Demographic Priors)  : {len(med_conf)} ({len(med_conf)/len(results)*100:.1f}%)")
    print(f"  • Ambiguous (Demands Machine Learning Models)   : {len(ambiguous)} ({len(ambiguous)/len(results)*100:.1f}%)")
    print("-" * 80)
    print(f"Deterministic Projected Test Survival Rate: {total_surv}/{len(results)} ({total_surv/len(results)*100:.1f}%)")
    print("=" * 80)
    print("\nSample of High-Confidence Family/Ticket Determinations:")
    print("-" * 80)
    for r in high_conf[:8]:
        decision_str = "SURVIVED (1)" if r["Deterministic_Prediction"] == 1 else "PERISHED (0)"
        print(f"PID {r['PassengerId']} | {r['Name']:<35} | {decision_str}")
        print(f"  └── Reason: {r['Explanation']}\n")


if __name__ == "__main__":
    main()
