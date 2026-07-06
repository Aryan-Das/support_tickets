from datasets import load_dataset
import pandas as pd

ds = load_dataset("Tobi-Bueck/customer-support-tickets")


# print(ds["train"].num_rows) 
# print(ds["train"].column_names) 

# print(ds["train"][:5])


en_dataset = ds.filter(lambda row: row["language"] == "en")
# de_dataset = ds.filter(lambda row: row["language"] == "de")

# print(en_dataset["train"].num_rows)
# print(de_dataset["train"].num_rows)

# en_body_null = en_dataset.filter(lambda row: (row["body"] == None or row["body"].strip() == ""))
# en_answer_null = en_dataset.filter(lambda row: (row["answer"] == None or row["answer"].strip() == ""))
# en_priority_null = en_dataset.filter(lambda row: (row["priority"] == None or row["priority"].strip() == ""))
# en_queue_null = en_dataset.filter(lambda row: (row["queue"] == None or row["queue"].strip() == ""))

# print(en_body_null["train"].num_rows)
# print(en_answer_null["train"].num_rows)
# print(en_priority_null["train"].num_rows)
# print(en_queue_null["train"].num_rows)




# print(pd.Series(en_dataset["train"]["queue"]).value_counts())
# print(pd.Series(en_dataset["train"]["priority"]).value_counts())

# lengths = [(len(row["body"]) if row["body"] is not None else 0) for row in en_dataset["train"]]
# print(pd.Series(lengths).describe())

# lengths = [(len(row["answer"]) if row["answer"] is not None else 0) for row in en_dataset["train"]]
# print(pd.Series(lengths).describe())



# filter nulls

en_dataset = en_dataset.filter(lambda row: (row["body"] != None and row["body"].strip() != ""))
en_dataset = en_dataset.filter(lambda row: (row["answer"] != None and row["answer"].strip() != ""))

# stratified sample 

df = pd.DataFrame(en_dataset["train"])
print(len(df))
MIN_PER_CATEGORY = 10
TARGET_TOTAL = 180

minimum_samples = df.groupby("queue").sample(n=MIN_PER_CATEGORY, random_state=42)
df_remaining = df.drop(index=minimum_samples.index)
remaining_needed = TARGET_TOTAL - (MIN_PER_CATEGORY * df["queue"].nunique())
remaining_sample = df_remaining.sample(n=remaining_needed, random_state=42)

pool_b = pd.concat([remaining_sample, minimum_samples], axis=0)
print(pool_b["queue"].value_counts())

pool_a = df.drop(index=pool_b.index)

overlap = set(pool_a.index) & set(pool_b.index)
assert len(overlap) == 0, f"Leakage detected: {len(overlap)} overlapping rows"
pool_a.to_csv("pool_a_kb.csv", index=True)
pool_b.to_csv("pool_b_eval.csv", index=True)
print(len(pool_a))
print(len(pool_b))
print(pool_a["queue"].value_counts())