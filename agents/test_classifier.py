from classifier import classifier_agent, parse_classifier_output
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
    sample = df.sample(n=80, random_state=42)
    try:
        output_df = pd.read_csv('./data/classifier_eval_data.csv', index_col=0)
    except Exception:
        output_df = pd.DataFrame(columns=['body', 'actual_queue', 'actual_priority', 'predicted_queue', 'predicted_priority', 'predicted_confidence', 'parse_successful'])
    i = 0
    for row in sample.itertuples():
        if row.body in output_df["body"].values:
            continue
        raw_response = classifier_agent(row.body)
        parsed = parse_classifier_output(raw_response)

        if parsed is not None:
            parse_succesful = True
            predicted_queue = parsed.queue
            predicted_priority = parsed.priority
            predicted_confidence = parsed.confidence
        else:
            parse_succesful = False
            predicted_queue = ""
            predicted_priority = ""
            predicted_confidence = 0


        new_row = pd.DataFrame([{
            'body': row.body,
            'actual_queue': row.queue,
            'actual_priority': row.priority,
            'predicted_queue': predicted_queue,
            'predicted_priority': predicted_priority,
            'predicted_confidence':predicted_confidence,
            'parse_successful': parse_succesful
        }])
        output_df = pd.concat([output_df, new_row], ignore_index=True)
        output_df.to_csv('./data/classifier_eval_data.csv', index=True)

        # rate limit = 30 requests per minute = one request every 2 seconds
        print_loading_bar(i, 80, prefix='Progress:', suffix='Complete', length=40)
        i += 1
        time.sleep(2.0) 
    print(f"Succesful Parse Rate: {(output_df['parse_successful'] == True).mean() * 100}")  
    print(f"Queue Accuracy: {(output_df['actual_queue'] == output_df['predicted_queue']).mean() * 100}")  
    print(f"Priority Accuracy: {(output_df['actual_priority'] == output_df['predicted_priority']).mean() * 100}")  
    limit_categories = output_df[
        (~output_df["actual_queue"].isin(["Human Resources", "Returns and Exchanges"]))
    ]
    print(f"Queue Accuracy Without Wrong Categories: {(limit_categories['actual_queue'] == limit_categories['predicted_queue']).mean() * 100}")  
    mismatches = output_df[output_df["actual_queue"] != output_df["predicted_queue"]]
    print(mismatches[["actual_queue", "predicted_queue"]].value_counts())
    mismatches = output_df[
        (output_df["actual_queue"] != output_df["predicted_queue"]) &
        (~output_df["actual_queue"].isin(["Human Resources", "Returns and Exchanges"]))
    ]
    
    print(mismatches[["actual_queue", "predicted_queue"]].value_counts())
    mismatches = output_df[output_df["actual_priority"] != output_df["predicted_priority"]]
    print(mismatches[["actual_priority", "predicted_priority"]].value_counts())
    print(output_df[output_df["actual_queue"] == "Returns and Exchanges"]["body"].to_list())
    print("----")
    print(output_df[output_df["actual_queue"] == "Human Resources"]["body"].to_list())
    print(sklearn.metrics.classification_report(output_df["actual_queue"], output_df["predicted_queue"]))
    print(sklearn.metrics.classification_report(output_df["actual_priority"], output_df["predicted_priority"]))

if __name__ == "__main__":
    main()
    