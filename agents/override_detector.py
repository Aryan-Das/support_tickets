from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIError
import os 
import json
import re


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)


def clean_response(response: str):
    cleaned = response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()

class OverrideCheck(BaseModel):
    reasoning: str
    triggered: bool
    matched_rules: list[int] 

def override_detector_agent(ticket_body : str):
    system_prompt = '''
    You are an override detector agent whose job is to take a customer support ticket and flag whether it applies to certain rules and conditions:

    1. Security incident — actual or suspected breach, unauthorized access, data leak, cyberattack (not general "how do I secure my account" questions — those are informational).
    2. Financial dispute — unauthorized/incorrect charges, refund requests, payment disputes. (Explicitly not: general questions about how billing cycles or payment methods work — those stay bot-handled.)
    4. Legal, compliance, or contractual language — mentions of contracts, SLAs, compliance obligations, legal action, threats of lawsuit.
    5. Explicit frustration/churn risk — customer expresses anger, threatens to cancel, or uses escalatory language ("this is unacceptable," "I want a refund immediately or I'm leaving").
    
    (#3 is INTENTIONALLY OMITTED.)

    Your reasoning should be a plain string.
    For triggered, return true if any rule was matched or false if no rules were matched (i.e. if an override is triggered or not)
    For matched_rules, provide a list of integers referring to which rules the ticket matched to (an empty list if triggered=False).
    
    Answer in the JSON structure shown in the following examples, with no markdown formatting or preamble text. respond ONLY in valid JSON.
    
 
    {
        "reasoning": "your reasoning",
        "triggered": true (OR false),
        "matched_rules": [1, 4],
    }

    '''

    user_message = f'--- TICKET ---\n{ticket_body}\n--- END TICKET ---\n'
   
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

            return {"status": "success", "data": response.choices[0].message.content}
        except RateLimitError as e:
            time.sleep(2.0)
            
        except APIError as e:
            time.sleep(2.0)
    return {"status": "failed", "data": None}


def parse_override_response(response: str):
    try:
        response_json = json.loads(clean_response(response["data"]))
        response_object = OverrideCheck(**response_json)
        return response_object
    except json.JSONDecodeError:
        print("ERROR: Invalid JSON: \n" + response)
        return None
    except ValidationError:
        print("ERROR: Schema Validation Error: \n" + response)
        return None