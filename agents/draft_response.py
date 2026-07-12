import chromadb
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


chroma_client = chromadb.PersistentClient(path='./chroma_db')
collection = chroma_client.get_or_create_collection('kb_tickets')

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def clean_response(response: str):
    cleaned = response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


class DraftResponse(BaseModel):
    reasoning: str
    draft_response: str
    sources_used: list[int]
    


def retriever(body: str):
    
    embeddings = model.encode(body)
    query = collection.query(query_embeddings=[embeddings], n_results=3)
    out = []
   
    for i in range(len(query['documents'][0])):
        out.append({
            "answer": query['documents'][0][i],
            "queue": query['metadatas'][0][i]['queue'],
            "priority": query['metadatas'][0][i]['priority'],
            "distance": query['distances'][0][i]
        })
    return [i for i in out if i["distance"] < 0.9]


def draft_response_agent(ticket_body : str):
    sources = retriever(ticket_body)
    if len(sources) <= 0:
        return {"status": "escalate", "data": "no_relevant_context_found"}
    system_prompt = '''
    You are a customer service agent whose job is to take a support ticket, give your reasoning about it, and draft a response based on provided example context answers given to you related to similar issues.
    Ground ALL ANSWERS SOLELY IN THE CONTEXT GIVEN.  if the context doesn't contain enough information to answer confidently, say so rather than inventing details.
    
    Your reasoning should be a plain string.
    For draft_response, provide a plain string containing only the customer facing reponse
    For sources used, provide a list of integers referring to which sources you used.
    
    Answer in the JSON structure shown in the following examples, with no markdown formatting or preamble text. respond ONLY in valid JSON.
    
 
    {
        "reasoning": "your reasoning",
        "draft_response": "your draft response",
        "sources_used": [0, 1, ... (any integer numbers of sources you used)],
    }

    '''

    user_message = f'--- TICKET ---\n{ticket_body}\n--- END TICKET ---\n'
    sources_string = ""
    for i in range(len(sources)):
        sources_string += f"Source {i}: {sources[i]['answer']}"
    sources_message = f'--- CONTEXT SOURCES ---\n{sources_string}\n--- END CONTEXT SOURCES ---\n'
    max_retries = 5
    for i in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message + "\n" + sources_message},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1
            )

            return {"status": "success", "data": response.choices[0].message.content, "top_distance": sources[0]["distance"]}
        except RateLimitError as e:
            time.sleep(2.0)
            
        except APIError as e:
            time.sleep(2.0)
    return {"status": "failed", "data": None}


def parse_draft_response(response):
    try:
        if response["data"] == "no_relevant_context_found":
            return None
        response_json = json.loads(clean_response(response["data"]))
        response_object = DraftResponse(**response_json)
        return response_object
    except json.JSONDecodeError:
        print("ERROR: Invalid JSON: \n" + response)
        return None
    except ValidationError:
        print("ERROR: Schema Validation Error: \n" + response)
        return None



