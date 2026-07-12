from draft_response import draft_response_agent, parse_draft_response
from override_detector import override_detector_agent, parse_override_response
from classifier import classifier_agent, parse_classifier_output

def confidence_gate(ticket_body: str):
    override_check = parse_override_response(override_detector_agent(ticket_body))
    classifier_data = parse_classifier_output(classifier_agent(ticket_body))
    if override_check.triggered:
        return {"status": "escalate", "reason": "rule_override", "matched_rules": override_check.matched_rules,  "classifier_data": classifier_data}
    
    draft_response = draft_response_agent(ticket_body)
    if draft_response["status"] == "escalate":
        return {"status": "escalate", "reason": "no_context",  "classifier_data": classifier_data}
    elif draft_response["status"] == "success":
        if draft_response["top_distance"] >= 0.3:
            return {"status": "escalate", "reason": "weak_retrieval_confidence", "classifier_data": classifier_data}
        
    return {"status": "auto_send", "data": draft_response["data"], "classifier_data": classifier_data}