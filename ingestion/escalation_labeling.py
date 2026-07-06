import pandas as pd
import os

DATA_PATH = "./data/pool_b_eval.csv"

def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0)
    df["escalation_reason"] = df["escalation_reason"].astype("object")
    if "should_escalate" not in df.columns:
        df["should_escalate"] = pd.NA
    if "escalation_reason" not in df.columns:
        df["escalation_reason"] = pd.NA
    return df

def save_data(df):
    df.to_csv(DATA_PATH, index=True)

def label_tickets():
    df = load_data()

    unlabeled_mask = df["should_escalate"].isna()
    unlabeled_indices = df[unlabeled_mask].index.tolist()

    total = len(df)
    already_done = total - len(unlabeled_indices)

    print(f"\n{already_done}/{total} tickets already labeled. {len(unlabeled_indices)} remaining.\n")
    print("Rubric reminder:")
    print("1 = Security incident")
    print("2 = Financial dispute")
    print("3 = Active service outage (business-impacting)")
    print("4 = Legal/compliance/contractual")
    print("5 = Explicit anger/churn risk")
    print("6 = Ambiguous/multi-part (last resort)")
    print("-" * 60)

    for idx in unlabeled_indices:
        row = df.loc[idx]

        print(f"\n[Row index: {idx}]")
        print(f"Queue: {row['queue']} | Priority: {row['priority']}")
        print("-" * 60)
        print(row["body"])
        print("-" * 60)

        while True:
            decision = input("Escalate? (y/n/skip/quit): ").strip().lower()
            if decision in ("y", "n", "skip", "quit"):
                break
            print("Please type y, n, skip, or quit.")

        if decision == "quit":
            print("Exiting. Progress saved.")
            break

        if decision == "skip":
            continue

        if decision == "y":
            reason = input("Rule number(s) triggered (e.g. '2' or '2,5'): ").strip()
            df.at[idx, "should_escalate"] = True
            df.at[idx, "escalation_reason"] = reason
        else:
            df.at[idx, "should_escalate"] = False
            df.at[idx, "escalation_reason"] = ""

        save_data(df)  

    print("\nSession ended. All progress saved to CSV.")

if __name__ == "__main__":
    label_tickets()