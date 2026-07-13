from typing import TypedDict, Annotated, List, Dict, Any, Optional
from override_detector import OverrideCheck, override_detector_agent, parse_override_response
from classifier import ClassifierResponse, classifier_agent, parse_classifier_output
from draft_response import draft_response_agent, parse_draft_response

from langgraph.graph import StateGraph, START, END

class TicketState(TypedDict):
    ticket_body: str
    override_result: Optional[OverrideCheck]
    classifier_result: Optional[ClassifierResponse]
    draft_result: Optional[Dict]
    final_status: Optional[str]
    final_reason: Optional[str]
    final_output: Optional[str]

def override_check_node(state: TicketState):
    ticket_body = state["ticket_body"]
    parsed_result = parse_override_response(override_detector_agent(ticket_body)) # the parse function already does the ["data"] step internally.
    if parsed_result is None:
        print("WARNING: override check failed to parse, defaulting to fail-safe escalation")

    return {"override_result": parsed_result}

def classifier_node(state: TicketState):
    ticket_body = state["ticket_body"]
    parsed_result = parse_classifier_output(classifier_agent(ticket_body)) 
    if parsed_result is None:
        print("WARNING: classifier failed to parse, defaulting to fail-safe escalation")
    return {"classifier_result": parsed_result}

def draft_node(state: TicketState) -> dict:
    raw_result = draft_response_agent(state["ticket_body"])
    
    if raw_result["status"] == "escalate":
        return {"draft_result": raw_result}
    
    if raw_result["status"] == "success":
        parsed = parse_draft_response(raw_result)
        if parsed is None:
            print("WARNING: draft check failed to parse, defaulting to fail-safe escalation")
        return {
            "draft_result": {
                "status": "success",
                "data": parsed,  
                "top_distance": raw_result["top_distance"],
            }
        }

    return {"draft_result": raw_result}

def gate_decision_node(state: TicketState):
    distance_threshold = 0.3
    OUTAGE_ADJACENT_QUEUES = {
        "Service Outages and Maintenance",
        "Technical Support",
        "IT Support",
        "Product Support",
    }

    if state["override_result"] is None:
        return {"final_status":"escalate", "final_reason":"override_parse_failed"}
    elif state["override_result"].triggered:
        return {"final_status":"escalate", "final_reason":"rule_override"}
    elif (state["classifier_result"] 
          and state["classifier_result"].priority == "high" 
          and state["classifier_result"].queue in OUTAGE_ADJACENT_QUEUES):
        return {"final_status": "escalate", "final_reason": "high_priority_outage"}
    elif state["draft_result"]["data"] is None:
        return {"final_status":"escalate", "final_reason":"draft_parse_failed"}
    elif state["draft_result"]["status"] == "escalate":
        return {"final_status":"escalate", "final_reason":"no_context"}
    elif state["draft_result"]["top_distance"] >= distance_threshold:
        return {"final_status":"escalate", "final_reason":"weak_retrieval_confidence"}
    else:
        return {"final_status":"auto_send", "final_output":state["draft_result"]["data"].draft_response}


def route_after_override(state: TicketState) -> str:
    if state["override_result"] is None or state["override_result"].triggered:
        return "gate_decision"
    return "classify"
    
def build_graph():
    graph = StateGraph(TicketState)
    graph.add_node("override_check", override_check_node)
    graph.add_node("classify", classifier_node)
    graph.add_node("draft", draft_node)
    graph.add_node("gate_decision", gate_decision_node)
    graph.set_entry_point("override_check")
    graph.add_conditional_edges("override_check", route_after_override)
    graph.add_edge("classify", "draft")
    graph.add_edge("draft", "gate_decision")
    graph.add_edge("gate_decision", END)

    return graph.compile()
