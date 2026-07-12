from override_detector import override_detector_agent, parse_override_response

# regular question
response = override_detector_agent("Dear Customer Support Team,\n\nI hope this message finds you well. I am reaching out to request detailed documentation related to the CI/CD pipeline employed in the current project. Comprehensive information on setup procedures, configurations, and best practices would be immensely helpful for our development team to optimize workflows and ensure seamless deployment processes.\n\nIn particular, I am interested in understanding the steps involved in the pipeline, the tools and technologies integrated, as well as recommended guidelines for maintenance and troubleshooting. Additionally, if there are sample configuration templates available, please share those as well.")
print(response)
print(parse_override_response(response))

# security breach
response = override_detector_agent("Dear Customer Support Team,\n\nI am reaching out to urgently highlight a severe security breach impacting our telehealth systems. Recently, we detected abnormal activities that could indicate a security intrusion, potentially jeopardizing confidential patient information and hindering service operations. Due to the vital importance of telehealth services, it is crucial that this issue is addressed without delay.\n\nWe kindly request a comprehensive investigation to determine the root cause and identify any vulnerabilities that may have been exploited. ")
print(response)
print(parse_override_response(response))