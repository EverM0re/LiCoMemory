SUMMARY_PROMPT = """You are a helpful assistant trying to summarize a given multi-turn chat transcript between a User and an Assistant.

-Goal-
Produce a compact multi-theme summary that groups the conversation by distinct topics. Aggregate mentions of the same theme even if they appear across different turns.

-Output format-
1. session_id: <session id> provided in the input text.

2. session_time: <session time> provided in the input text.

3. keys: strictly up to 5 <key information> strings extracted from the session that may include personal information, specific date and location, or a common topic. Reduced form of session_time MUST be included.

4. Themes: For each distinct theme, output a block with:
  - theme_x: <short title> followed by a number.
  - summary_x: key points from the user (preferences, questions, constraints) followed by key recommendations/answers (only from the transcript) from the assistant. x corresponds to the number of theme_x.

-Style & Rules-
1. IMPORTANT: Focus more on the user's preferences and personalized information.

2. Write concisely (~150-220 words total). Do not quote long text; keep any quotes within 10 words.

3. Use neutral tone; third-person (“The user…”, “The assistant…”).

4. Be strictly factual; no hallucinations or outside knowledge.

5. Output in the json format STRICTLY from the example template.

######################
-Examples-
A resulting summary should follow the following format and may look like this:
######################

{
  "session_id": "sharegpt_yywfIrx_0",
  "session_time": "2025/05/04",
  "keys": "May 4th, Podcast, January 25th, Software Engineer",
  "context": {
    "theme_1": "Entrepreneurship podcasts",
    "summary_1": "The user expressed interest in listening to podcasts during his job as a software engineer. The assistant recommended over a dozen podcasts, such as Tim Ferriss Show, Entrepreneur on Fire, etc., and provided detailed explanations of their unique features.",
    "theme_2": "News podcast",
    "summary_2": "The user shared that they listen to The Daily every day and mentioned an episode on February 10th about the COVID-19 vaccine rollout. The assistant acknowledged the depth of reporting and storytelling on The Daily."
    "theme_3": "Comedy podcasts",
    "summary_3": "The user recalled laughing out loud on January 25th while riding the bus while listening to My Brother, My Brother and Me due to the absurd advice from the McElroy brothers. The assistant responded by acknowledging the humor and charm of the show.",
  }
}

######################
-Real Input Text-
######################

Input Text:
{text}

################
Output:"""

ADDITION_PROMPT = """You are a helpful assistant trying to update a summary of a given multi-turn chat transcript between a User and an Assistant.

-Goal-
Given a existing summary and a new dialogue chunk, make an addition to the summary according to the new dialogue. Genrate new keys and themes if necessary or aggregate existing ones.

-Output format-
1. session_id: copy from the existing summary.

2. session_time: copy from the existing summary.

3. keys: if the dialogue contains new key information which may include personal information, specific date and location, or a common topic that is not included in the existing keys, extract it and add to the keys.

4. Themes: If new themes are introduced in the dialogue, extract them and add to the themes in the following format. Otherwise, integrate the new dialogue into the existing themes.
  - theme_x: <short title> followed by a number.
  - summary_x: key points from the user (preferences, questions, constraints) followed by key recommendations/answers (only from the transcript) from the assistant. x corresponds to the number of theme_x.

-Style & Rules-
1. IMPORTANT: Focus more on the user's preferences and personalized information.

2. Use neutral tone; third-person ("The user…", "The assistant…").

3. Be strictly factual; no hallucinations or outside knowledge.

4. Try your best to integrate the new dialogue into the existing themes.

5. You are allowed to make no changes if all information in the new dialogue are already included in the existing summary.

######################
-Examples-
A resulting summary should follow the following format and may look like this:
######################

Existing Summary:
{
  "session_id": "sharegpt_yywfIrx_0",
  "session_time": "2025/05/04",
  "keys": "May 4th, Podcast, January 25th, Software Engineer",
  "context": {
    "theme_1": "Entrepreneurship podcasts",
    "summary_1": "The user expressed interest in listening to podcasts related to entrepreneurship (besides How I Built This). The assistant recommended over a dozen podcasts, such as Tim Ferriss Show, Entrepreneur on Fire, GaryVee Audio Experience, etc.",
    "theme_2": "Tim Ferriss and Naval Ravikant",
    "summary_2": "The user specifically mentioned listening to the episode of Tim Ferriss Show featuring Naval Ravikant. The assistant summarized Naval's views on self-awareness, meditation, entrepreneurial mindset, and wealth creation, citing several of his memorable quotes.",
  }
}


Dialogue Chunk:
User: I have also been fascinated about Steve Jobs since I finished a podcast of his biography. Can you tell me more about him?
Assistant: Sure thing! Steve Jobs was the co-founder, chairman, and CEO of Apple Inc. He was also the co-founder and CEO of Pixar Animation Studios when he was 30 years old. He was known for his innovative designs and his vision for the future.


Updated Summary:
{
  "session_id": "sharegpt_yywfIrx_0",
  "session_time": "2025/05/04",
  "keys": "May 4th, Podcast, January 25th, Software Engineer, Steve Jobs",
  "context": {
    "theme_1": "Entrepreneurship podcasts",
    "summary_1": "The user expressed interest in listening to podcasts during his job as a software engineer. The assistant recommended over a dozen podcasts, such as Tim Ferriss Show, Entrepreneur on Fire, etc.",
    "theme_2": "News podcast",
    "summary_2": "The user shared that they listen to The Daily every day and mentioned an episode on February 10th about the COVID-19 vaccine rollout. The assistant acknowledged the depth of reporting and storytelling on The Daily."
    "theme_3": "Steve Jobs",
    "summary_3": "The user expressed interest in Steve Jobs after finishing a podcast of his biography. The assistant provided additionalinformation about his background.",
  }
}

######################
-Real Input Text-
######################
Existing Summary:
{summary}

Dialogue Chunk:
{text}

################
Output:"""

