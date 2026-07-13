from graph import build_graph
import time
import pandas as pd
import sklearn

def print_loading_bar(iteration, total, prefix='', suffix='', length=40, fill='█'):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
  
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    
   
    if iteration == total: 
        print()

def main():
    df = pd.read_csv('./data/pool_b_eval.csv')
   
    app = build_graph()
    sample = df.sample(n=80, random_state=42)
    try:
        output_df = pd.read_csv('./data/graph_eval.csv', index_col=0)
    except Exception:
        output_df = pd.DataFrame(columns=['body', 'actual_should_escalate', 'actual_escalation_reason', 'predicted_final_status', 'predicted_final_reason', 'top_distance', 'predicted_should_escalate', 'predicted_queue', 'predicted_priority'])
    i = len(output_df)
    for row in sample.itertuples():
        if row.body in output_df["body"].values:
            continue
        result_state = app.invoke({
            "ticket_body": row.body,
            "override_result": None,
            "classifier_result": None,
            "draft_result": None,
            "final_status": None,
            "final_reason": None,
            "final_output": None,
        })

        

        new_row = pd.DataFrame([{
            'body': row.body,
            'actual_should_escalate': row.should_escalate,
            'actual_escalation_reason': row.escalation_reason,
            'predicted_queue': result_state["classifier_result"].queue if result_state["classifier_result"] else None,
            'predicted_priority': result_state["classifier_result"].priority if result_state["classifier_result"] else None,
            'predicted_final_reason':result_state["final_reason"],
            'predicted_final_status': result_state["final_status"],
            'predicted_should_escalate': result_state["final_status"] == "escalate",
            'top_distance': result_state["draft_result"].get("top_distance") if result_state["draft_result"] else None,
            'final_output': result_state["final_output"] 
        }])
        output_df = pd.concat([output_df, new_row], ignore_index=True)
        output_df.to_csv('./data/graph_eval.csv', index=True)

        time.sleep(6.0) 
        print_loading_bar(i, 80, prefix='Progress:', suffix='Complete', length=40)
        i += 1
    # output metrics
    print(sklearn.metrics.classification_report(output_df["actual_should_escalate"], output_df["predicted_should_escalate"]))
    print(output_df["predicted_final_reason"].value_counts())
    outage_escalations = output_df[output_df["predicted_final_reason"] == "high_priority_outage"]

    print(f"Total tickets escalated via high_priority_outage rule: {len(outage_escalations)}")
    print(outage_escalations["actual_should_escalate"].value_counts())
    print()
    print(f"Precision of this specific rule: {(outage_escalations['actual_should_escalate'] == True).mean():.2%}")
    false_negatives = output_df[
        (output_df["actual_should_escalate"] == True) & 
        (output_df["predicted_should_escalate"] == False)
    ]

    print(f"False negative count: {len(false_negatives)}")
    for idx, row in false_negatives.iterrows():
        print("---")
        print("Ticket:", row["body"])
        print("Predicted queue/priority:", row["predicted_queue"], "/", row["predicted_priority"])
        print("Top distance:", row["top_distance"])
        print("Final output sent:", row["final_output"])

if __name__ == "__main__":
    main()

    