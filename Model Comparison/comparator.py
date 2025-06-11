import pandas as pd
from thefuzz import fuzz

from neo4j import GraphDatabase
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Your query_set list goes here
query_set = [
    {
        "name": "Query 1: Find All Conditions for a Specific Patient with ID: '0fe22cec-1a19-99da-b67f-7ce364c4cf3e'",
        "gt_query": """
            MATCH (p:Patient {Id: '0fe22cec-1a19-99da-b67f-7ce364c4cf3e'})-[:HAS_CONDITION]->(c:Condition)
RETURN c.CODE, c.DESCRIPTION, c.START, c.STOP
ORDER BY c.START
        """,
        "llm_query": """
        """
    },

        {
        "name": "Query 2: List All Medications with Reason for Use for a Patient with ID: 'b1f7b5a9-5cf5-6050-b23c-81027f53bdfd'",
        "gt_query": """
            MATCH (p:Patient {Id: 'b1f7b5a9-5cf5-6050-b23c-81027f53bdfd'})-[:HAS_MEDICATION]->(m:Medication)
RETURN m.CODE, m.DESCRIPTION, m.REASONDESCRIPTION, m.START, m.STOP
ORDER BY m.START
        """,
        "llm_query": """
        """
    },



    {
        "name": "Query 3: Find All Observations for a Patient with ID: 95559bb7-a9a3-104c-a710-e9a1e22b312e",
        "gt_query": """
            MATCH (p:Patient {Id: '95559bb7-a9a3-104c-a710-e9a1e22b312e'})-[:HAS_OBSERVATION]->(o:Observation)
RETURN o.CODE, o.DESCRIPTION, o.VALUE, o.UNITS, o.DATE
ORDER BY o.DATE
        """,
        "llm_query": """
        """
    },
    
    
    
    {
        "name": "Query 4: Get All Procedures Performed During an Encounter for ID: 5f4177ca-c592-bc22-116c-27de8b3b1cb5",
        "gt_query": """
            MATCH (pr:Procedure)-[:PART_OF_ENCOUNTER]->(e:Encounter {Id: '5f4177ca-c592-bc22-116c-27de8b3b1cb5'})
RETURN pr.CODE, pr.DESCRIPTION, pr.START, pr.STOP
        """,
        "llm_query": """
        """
    },
    
    
    {
        "name": "Query 5: Find Immunizations Given During a Specific Encounter whose ID is: 80dbe42a-c87e-ddd4-cf40-da2c6b764afd",  
        "gt_query": """
            MATCH (i:Immunization)-[:PART_OF_ENCOUNTER]->(e:Encounter {Id: '80dbe42a-c87e-ddd4-cf40-da2c6b764afd'})
RETURN i.CODE, i.DESCRIPTION, i.DATE
        """,
        "llm_query": """
        """
    },
    
      
    
    {
        "name": "Query 6: Number of Immunizations by Vaccine Type",
        "gt_query": """
            MATCH (i:Immunization)
RETURN i.DESCRIPTION, COUNT(*)
ORDER BY COUNT(*) DESC
        """,
        "llm_query": """
        """
    },
    
    
    
    {
        "name": "Query 7: Number of Patients by Race and Ethnicity",
        "gt_query": """
            MATCH (p:Patient)
RETURN p.RACE, p.ETHNICITY, COUNT(*)
ORDER BY COUNT(*) DESC
        """,
        "llm_query": """
        """
    },
    
    
    
    
    {
        "name": "Query 8: Average Duration of Medication Usage",
        "gt_query": """
            MATCH (m:Medication)
WHERE m.START IS NOT NULL AND m.STOP IS NOT NULL
WITH duration.between(datetime(m.START), datetime(m.STOP)) AS d
RETURN avg(d.days) AS averageMedicationDuration
        """,
        "llm_query": """
        """
    },
    
    
    
    
    {
        "name": "Query 9: Average Observation Value by Code",
        "gt_query": """
            MATCH (o:Observation)
WHERE o.VALUE IS NOT NULL
RETURN o.CODE, avg(toFloat(o.VALUE)), o.UNITS
ORDER BY avg(toFloat(o.VALUE)) DESC
        """,
        "llm_query": """
        """
    },
    
    
    {
        "name": "Query 10: Number of Procedures by Description",
        "gt_query": """
            MATCH (pr:Procedure)
RETURN pr.DESCRIPTION, COUNT(*)
ORDER BY COUNT(*) DESC
        """,
        "llm_query": """
        """
    },{
        # add more queries as needed
    }
 ] # 👈 paste your full query_set here

# Neo4j connection details
NEO4J_URI = ""
NEO4J_USER = ""
NEO4J_PASSWORD = ""

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def run_query(query):
    try:
        with driver.session() as session:
            result = session.run(query)
            records = result.data()
            return pd.DataFrame(records), None
    except Exception as e:
        return None, str(e)

def fuzzy_match_rows(gt_df, llm_df, threshold=85):
    if gt_df.empty or llm_df.empty:
        return 0, 0
    gt_rows = gt_df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()
    llm_rows = llm_df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()
    
    matched_gt = set()
    matched_llm = set()
    
    for i, gt_row in enumerate(gt_rows):
        for j, llm_row in enumerate(llm_rows):
            if j in matched_llm:
                continue
            score = fuzz.token_sort_ratio(gt_row, llm_row)
            if score >= threshold:
                matched_gt.add(i)
                matched_llm.add(j)
                break

    true_positive = len(matched_gt)
    false_positive = len(llm_rows) - true_positive
    false_negative = len(gt_rows) - true_positive

    return true_positive, false_positive, false_negative

# Metrics
all_tp, all_fp, all_fn = 0, 0, 0
y_true, y_pred = [], []

for q in query_set:
    print(f"\n🔎 {q['name']}")
    
    gt_df, gt_err = run_query(q['gt_query'])
    llm_df, llm_err = run_query(q['llm_query'])

    if gt_err:
        print(f"❌ GT Query Error: {gt_err}")
        continue

    if llm_err:
        print(f"❌ LLM Query Error: {llm_err}")
        # Penalize: assume 0 TP, all GT rows as FN, 0 FP
        tp, fp, fn = 0, 0, len(gt_df)
    else:
        common_cols = sorted(set(gt_df.columns) & set(llm_df.columns))

        if not common_cols:
            print("⚠️ No common columns. Penalizing the model.")
    # Apply a penalty: treat all GT rows as false negatives, and all LLM rows as false positives
            tp = 0
            fp = len(llm_df)
            fn = len(gt_df)

        else:
    # Proceed with normal fuzzy matching
            gt_common = gt_df[common_cols].drop_duplicates().reset_index(drop=True)
            llm_common = llm_df[common_cols].drop_duplicates().reset_index(drop=True)
            tp, fp, fn = fuzzy_match_rows(gt_common, llm_common)

    
    print(f"✅ TP: {tp}, ❌ FP: {fp}, ❗ FN: {fn}")
    all_tp += tp
    all_fp += fp
    all_fn += fn
    
    y_true += [1]*tp + [1]*fn
    y_pred += [1]*tp + [0]*fn
    y_true += [0]*fp
    y_pred += [1]*fp

# Final Metrics
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
accuracy = accuracy_score(y_true, y_pred)

print("\n📊 Final LLM Evaluation Metrics")
print(f"🔹 Precision: {precision:.2f}")
print(f"🔹 Recall:    {recall:.2f}")
print(f"🔹 F1 Score:  {f1:.2f}")
print(f"🔹 Accuracy:  {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["GT: No", "GT: Yes"],
            yticklabels=["Pred: No", "Pred: Yes"])

plt.title("LLM vs GT Confusion Matrix")
plt.xlabel("Ground Truth")
plt.ylabel("LLM Prediction")
plt.show()

# Combine confusion matrix and metrics in one DataFrame
cm_df = pd.DataFrame(cm, 
                     index=["Pred: No", "Pred: Yes"], 
                     columns=["GT: No", "GT: Yes"])
cm_df.index.name = "Confusion Matrix"

# Create a blank row and metrics DataFrame
blank_row = pd.DataFrame([["", ""]], columns=cm_df.columns[:2])  # empty row with matching columns
metrics_df = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1 Score", "Accuracy"],
    "Score": [precision, recall, f1, accuracy]
})

metrics_df.index = [""] * len(metrics_df)  # Remove index labels

# Concatenate all into one DataFrame
combined = pd.concat([
    cm_df,
    blank_row,
    metrics_df.rename(columns={"Metric": "Confusion Matrix", "Score": "GT: No"})
], ignore_index=False)


import os

# Prepare the metrics row
log_row = {
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "Accuracy": accuracy,
    "TP": all_tp,
    "FP": all_fp,
    "FN": all_fn
}

# Path to save
csv_path = ""

# Check if file exists
if os.path.exists(csv_path):
    existing_df = pd.read_csv(csv_path)
    new_df = pd.concat([existing_df, pd.DataFrame([log_row])], ignore_index=True)
else:
    new_df = pd.DataFrame([log_row])

# Save updated log
new_df.to_csv(csv_path, index=False)

print(f"\n✅ Appended evaluation results to {csv_path}")

