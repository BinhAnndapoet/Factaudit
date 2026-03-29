# Quality Inspector Agent

judge_new_case_prompt = """Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, the LLM must be tested to accurately assess the factuality of the information presented within the source claim according to the claim itself or the auxiliary information.

Please judge whether the new test cases "{new_point}"  are suitable as diverse and comprehensive exam questions on the sub task "{task_name}". The judgment criteria are as follows:
1. Each claim of the new test cases should be important and meaningful to the main task, avoiding unnecessary ambiguity in the key point.
2. If "auxiliary_info" is not empty in each of the new test cases, it can be noisy but must be helpful to the fact verification process; If "auxiliary_info" is empty, just keep it empty.
3. If "test_mode" is [claim], "auxiliary_info" must be empty.
4. If "test_mode" is [wisdom of crowds], please check "auxiliary_info" that: a) the user comments in "auxiliary_info" should be valuable enough as the wisdom of crowds for fact verification and b) the depth of the propagation conversation tree composed of the user response in "auxiliary_info" must be a random integer more than two.
5. If "test_mode" is [evidence], please check "auxiliary_info" that: a) four or more random pieces of evidence are in "auxiliary_info", and b) the provided pieces of detailed evidence in "auxiliary_info" must be ONLY ground truth based on Wikipedia or other authority, where all supported, refuted, and neutral evidence to the source claim should be included.
6. The fact-checking topic in each test case should be diverse enough and sufficiently different from each other.

If the new test cases are judged suitable as the exam questions on the sub task "{task_name}" by checking the judgment criteria, please ONLY keep the original content "{new_point}" as output in a JSON format: [json]; If there is one test case not conforming to the judgment criteria, you have to revise and improve the original content "{new_point}" to conform to the aforementioned judgment criteria, and ONLY output the improved test cases in a JSON format: [json]."""


verification_prompt = """You are a strict, objective fact-checker.
We have an AI-generated test case containing a claim and some evidence. We need to verify if the generated evidence is based on reality or if it is hallucinated.

[Claim to check]: 
{claim}

[AI-Generated Evidence]: 
{evidence}

[Real-time Web Search Results]:
{web_results}

Task:
Compare the [AI-Generated Evidence] against the [Real-time Web Search Results].
1. If the generated evidence is supported by the web results, set is_factual to True.
2. If the generated evidence is hallucinated, fabricated, or contradicts the real facts in the web results, set is_factual to False.
3. If False, write the correct facts found in the search results into the 'correction' field so the evidence can be rewritten later.
"""