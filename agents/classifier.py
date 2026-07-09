from pydantic import BaseModel, Field, ValidationError
from typing import Literal
import re
import os
import json
from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIError

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

import re

def clean_response(response: str):
    cleaned = response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()

class ClassifierResponse(BaseModel):
    reasoning: str
    queue: Literal["Technical Support", "Product Support", "IT Support", "Customer Service", "Service Outages and Maintenance", "Billing and Payments", "Returns and Exchanges", "General Inquiry", "Human Resources", "Sales and Pre-Sales"]
    priority: Literal["high", "medium", "low"]
    confidence: float = Field(..., ge=0, le=1)

def classifier_agent(ticket_body : str):
    system_prompt = '''
    You are a support ticket classifier whose job is to take a support ticket, give your reasoning about it, and classify its queue and priority level, as well as your confidence in that classification.
    Your reasoning should be a plain string.
    For queue, select from the following options: "Technical Support", "Product Support", "IT Support", "Customer Service", "Service Outages and Maintenance", "Billing and Payments", "Returns and Exchanges", "General Inquiry", "Human Resources", "Sales and Pre-Sales"
    For priority, either select "high", "medium", or "low"
    For confidence, provide a decimal between 0 and 1
    Fair enough — this is prompt content rather than pipeline logic, so here it is. I based the distinguishing rules on typical support-ticket triage conventions, but you should sanity-check this against 3-5 real Pool A examples per category before trusting it fully — if your dataset's actual labeling convention differs from what I'm assuming, this won't fix the confusion.
CATEGORY DISAMBIGUATION RULES:
Several categories can seem similar — use these specific distinctions:

- "Technical Support": issues with the COMPANY'S OWN systems, services, or infrastructure that the customer is trying to use — account access problems, service outages, login failures, system errors on the company's platform, integration/sync failures with the company's product.

- "Product Support": questions about HOW A SPECIFIC PRODUCT FEATURE WORKS, product compatibility with other software/tools, requests for product documentation or guidance, or how to use a product capability. Focus: "how does this feature work" or "does this work with X," not "something is broken."

- "IT Support": issues rooted in the CUSTOMER'S OWN internal IT environment — their hardware, their internal network/device configuration, or internal software conflicts unrelated to a specific product feature. Focus: the customer's own infrastructure, not the company's.

- "Customer Service": general account, order, or service questions that are NOT technical in nature — policy questions, general assistance requests, non-technical complaints, requests unrelated to a specific technical problem or product feature.

- "Billing and Payments": ONLY questions or issues specifically about charges, invoices, payment methods, billing cycles, or refunds/disputes related to money.

- "Returns and Exchanges": ONLY questions specifically about returning or exchanging a physical or purchased item — do not use this category for unrelated business, marketing, or technical requests even if the ticket mentions dissatisfaction.

- "Sales and Pre-Sales": questions from prospective or existing customers about purchasing, upgrading, pricing, or evaluating whether to buy — NOT existing customers troubleshooting something they already have.

- "Service Outages and Maintenance": reports of a SPECIFIC, CURRENT outage or service disruption affecting multiple users or systems — not a single user's isolated technical issue.

- "Human Resources": ONLY tickets that are genuinely about internal company HR matters (employment, benefits, internal policy) — this is rare in a customer support context; do not use this category for technical or business tickets just because they seem serious or came from a business context.

- "General Inquiry": use ONLY when a ticket does not clearly fit any other category above — this should be a last resort, not a default when uncertain. If a ticket could plausibly fit a more specific category, prefer that category over General Inquiry.

When a ticket could plausibly fit multiple categories, choose the MOST SPECIFIC one that matches the customer's actual request, not the broadest one.
    Answer in the JSON structure shown in the following examples, with no markdown formatting or preamble text. respond ONLY in valid JSON.
    
    Input:
    Dear Customer Support Team,\\n\\nI am writing to report a significant problem with the centralized account management portal, which currently appears to be offline. This outage is blocking access to account settings, leading to substantial inconvenience. I have attempted to log in multiple times using different browsers and devices, but the issue persists.\\n\\nCould you please provide an update on the outage status and an estimated time for resolution? Also, are there any alternative ways to access and manage my account during this downtime? 
    
    Expected Output:
    {
        "reasoning": "This ticket deals with an offline account management portal, which is a Technical Support Concern, and is causing substantial inconvenience meaning it is high priority",
        "queue": "Technical Support",
        "priority": "high",
        "confidence": 0.85
    }

    Input: 
    Dear Customer Support Team,\\n\\nI hope this message finds you well. I am reaching out to request clarification about the billing and payment procedures linked to my account. Recently, I observed some inconsistencies in the charges applied and would like to ensure I fully understand the billing cycle, accepted payment options, and any potential extra charges.\\n\\nFirstly, I would be grateful if you could provide a detailed explanation of how the billing cycle functions. Specifically, I am interested in knowing the start and end dates.\\n\\nThank you for your assistance regarding these billing inquiries.
    Expected Output:
    {
        "reasoning": "This ticket is asking about the billing process and payment options, but is merely inquiring about inconsistencies and not an urgent financial issue, meaning it is low priority",
        "queue": "Billing and Payments",
        "priority": "low",
        "confidence": 0.43
    }

    Input:
    Dear Customer Support,\n\nWe are experiencing extensive connectivity problems impacting numerous devices throughout the office. The issues have been observed with headsets, printers, and workstations all at once, significantly disrupting daily activities. Our initial investigation indicates that the cause may be a network outage or a misconfiguration within the system infrastructure.\n\nOur team has already tried several troubleshooting methods, including rebooting affected devices and swapping hardware components, but unfortunately, these efforts did not resolve the disruptions.
    Expected Output:
    {
        "reasoning": "This ticket addresses connectivity problems with devices and system infrastructure, which is an IT concern, but it is not a major security issue or critical outage so it is medium priority",
        "queue": "IT Support",
        "priority": "medium",
        "confidence": 0.88
    }

    Input:
    Dear Customer Support Team,\n\nI am reaching out to request an update on the structural details of our organization. The current records reflect information about the marketing agency, which has recently experienced several significant changes. To ensure our records accurately depict the current operations, I would appreciate your assistance in updating the information.\n\nFirst, I would like to clarify the refined roles within the various departments of the agency. Each department now has clearly defined responsibilities aimed at enhancing efficiency and accountability. For instance, the Creative Department is now dedicated solely to content creation.
    Expected Output:
    {
        "reasoning": "This ticket asks a general details question, and mentions effects to the marketing agencies meaning it is medium priority.",
        "queue": "General Inquiry",
        "priority": "medium",
        "confidence": 0.67
    }

    Input:
    Dear Customer Support Team,\n\nI am reaching out to request assistance regarding ongoing issues with the RAID controller device connection. Over the past few weeks, I have observed sporadic device detection failures and frequent disconnections, which significantly impact system performance and data integrity.\n\nThe RAID controller appears to intermittently lose communication with connected drives, leading to degraded RAID arrays and occasional system crashes. Basic troubleshooting steps such as checking cable connections and updating firmware, as well as verifying power supply stability, have been attempted, but the problem persists.\n\nCould you please perform advanced diagnostic procedures?
    Expected Output:
    {
        "reasoning": "This ticket requests support about a particular product, the RAID controller device, causing system crashes, but these crashes are only occasional and not mentioned as a major risk, so it is medium priority.",
        "queue": "Product Support",
        "priority": "medium",
        "confidence": 0.99
    }
    '''

    user_message = f'--- TICKET ---\n{ticket_body}\n--- END TICKET ---'
    max_retries = 5
    for i in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                    
                
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1
            )

            return response.choices[0].message.content
        except RateLimitError as e:
            time.sleep(2.0)
            
        except APIError as e:
            time.sleep(2.0)
    return None


def parse_classifier_output(response: str):
    try:
        response_json = json.loads(clean_response(response))
        response_object = ClassifierResponse(**response_json)
        return response_object
    except json.JSONDecodeError:
        print("ERROR: Invalid JSON: \n" + response)
        return None
    except ValidationError:
        print("ERROR: Schema Validation Error: \n" + response)
        return None
