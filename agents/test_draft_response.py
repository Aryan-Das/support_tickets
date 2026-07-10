from draft_response import draft_response_agent, parse_draft_response

# good query
response = draft_response_agent('Is there any information on the compatibility between Microsoft Teams and Kaspersky Internet Security 2021? I am currently using both and want to ensure they work seamlessly together. Could you please inform me about any known issues or conflicts between these two applications? I would appreciate any guidance or recommendations to ensure compatibility.')
print(response)
print(parse_draft_response(response))

# bad query

response = draft_response_agent('What is the weather like on mars?.')
print(response)
print(parse_draft_response(response))
