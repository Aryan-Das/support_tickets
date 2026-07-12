from confidence_gate import confidence_gate

# Auto send
print(confidence_gate("Is there any information on the compatibility between Microsoft Teams and Kaspersky Internet Security 2021? I am currently using both and want to ensure they work seamlessly together. Could you please inform me about any known issues or conflicts between these two applications? I would appreciate any guidance or recommendations to ensure compatibility."))

# Rule override
print(confidence_gate("Dear Customer Support Team,\n\nI am reaching out to urgently highlight a severe security breach impacting our telehealth systems. Recently, we detected abnormal activities that could indicate a security intrusion, potentially jeopardizing confidential patient information and hindering service operations. Due to the vital importance of telehealth services, it is crucial that this issue is addressed without delay.\n\nWe kindly request a comprehensive investigation to determine the root cause and identify any vulnerabilities that may have been exploited."))

# No retrieval
print(confidence_gate("What's the weather like on mars?"))


