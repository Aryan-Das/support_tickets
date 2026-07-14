# Autonomous Support Ticket Router
RAG-based support ticket triage pipeline 

## Features
- LangGraph multi-agent orchestration
- ChromaDB retrieval
- Rule-based and retrieval-confidence escalation gating
- Data loading, labeling, and embeddings scripts to form a knowledge base for RAG
- Pipeline evaluation harness
- 88% recall and 68% precision on human-escalation decisions against hand-labeled ground truth 


## File Structure
 
```
support_tickets/
├── agents/
    ├── classifier.py                # classifier agent
    ├── draft_response.py            # response drafting agent
    ├── graph.py                     # LangGraph nodes and pipeline
    ├── override_detector.py         # override/escalation detection agent
    ├── test_classifier.py           # runs a batch test on the classifier agent and prints statistics
    ├── test_draft_response.py       # some sample response drafts
    ├── test_graph.py                # some sample graph invokations
    ├── test_override_detector.py    # some sample override detection outputs
    └── test_pipeline.py             # eval harness for full pipeline         
├── chroma_db/                       # embedded knowledge base
├── data/
    ├── classifier_eval_data.csv     # saved test_graph results 
    ├── graph_eval.csv               # saved eval harness results
    ├── pool_a_kb.csv                # knowledge base 
    └── pool_b_eval.csv              # evaluation dataset
├── ingestion/
    ├── embeddings.py                # embed knowledge base in ChromaDB
    ├── escalation_labeling.py       # helper script for manually labeling should_escalate field
    ├── load_dataset.py              # load HuggingFace dataset and split into eval and kb
    └── test_query.py                # sample ChromaDB query
└── requirements.txt          
```

## Known Limitations
- Mislabeled data in the original dataset effects knowledge base and eval results
- Low classifier accuracy in predicting the queue category of a ticket (as many are ambiguous)
- Root cause analysis of both false positives (in the high_priority_outage override) and false negatives (missed escalations) traces back to this same issue

