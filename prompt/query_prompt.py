QUERY_PROMPT = """Please follow the instructions and answer the question based on the given context:

Question Time: {question_time}
Question: {question}

Context:
- Triples: {triples}
- Text Chunks: {chunks}

-Instructions-
1. Keep your answer brief without any additional explanations
2. Only use the information from the context to answer the question
3. If information collides, prioritize the information with time priority (CLOSER to question time)
4. If the context doesn't contain sufficient information, say so

################
Output:"""

SUMMARY_QUERY_PROMPT = """Please follow the instructions and answer the question based on the given context:

Question Time: {question_time}
Question: {question}

Context:
- Session Summaries: {summaries}
- Triples: {triples}
- Text Chunks: {chunks}

-Instructions-
1. Keep your answer brief without any additional explanations
2. Only use the information from the context to answer the question
3. If information collides, prioritize the information with time priority (CLOSER to question time)
4. If the context doesn't contain sufficient information, say so

################
Output:"""



