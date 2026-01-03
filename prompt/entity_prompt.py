DIALOGUE_EXTRACTION_PROMPT = """You are a helpful assistant trying to extract entities and relations from a given multi-turn chat transcript between a User and an Assistant.

-Goal-
Given a multi-turn chat transcript between a User and an Assistant in json format, identify all entities of pre-defined entity types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity in original format, capitalized. User and Assistant MUST be among the entities.
- entity_type: ONLY from one of the following types: [person, time, organization, location, event, concept, object]. Note that temporal information should be processed and extracted as a time entity.
Format each entity as ("entity"|<entity_name>|<entity_type>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other. FOCUS MORE on relationships clarifying user's personal information and preferences.
For each pair of related entities, extract the following information:
- create_time: copy from the <session_time> in the provided json corpus
- session_id: copy from the <session_id> in the provided json corpus
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_name: how the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity from 1 and 10. User related relationships should have a higher strength.
Format each relationship as ("relationship"|<create_time>|<session_id>|<source_entity>|<target_entity>|<relationship_name>|<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.

4. When finished, output ##END##

######################
-Example-
######################
Text:
'text': 'User: I've been thinking about starting my own business as I graudated from business administration and I was wondering if you could recommend some inspiring podcasts about entrepreneurship, aside from \"How I Built This\" which I finished this Monday on my sister's recommendation.\nAssistant: Congrats on starting your business! An inspiring podcast similar to "How I Built This" is **Side Hustle School** - Short tips to grow side gigs into full-time.\nA single spark can launch bold goals. Good luck!',
'session_id': 'yywfIrx_0',
'session_time': '2023/05/20 (Sat) 02:21',

################
("entity"|User|person)##
("entity"|Assistant|person)##
("entity"|How I Built This|concept)##
("entity"|Side Hustle School|concept)##
("entity"|Business Administration|concept)##
("entity"|Entrepreneurship|concept)##
("entity"|Sister|person)##
("entity"|January 15th|time)##
("entity"|Business|concept)##
("entity"|Podcast|object)##
("relationship"|2023/05/20|yywfIrx_0|User|Business Administration|has educational background|10)##
("relationship"|2023/05/20|yywfIrx_0|User|Entrepreneurship|interested in|9)##
("relationship"|2023/05/20|yywfIrx_0|User|How I Built This|finished listening to|8)##
("relationship"|2023/05/20|yywfIrx_0|User|Sister|received recommendation from|9)##
("relationship"|2023/05/20|yywfIrx_0|How I Built This|January 15th|finished on|7)##
("relationship"|2023/05/20|yywfIrx_0|User|Business|planning to start|8)##
("relationship"|2023/05/20|yywfIrx_0|User|Podcast|interested in|7)##
("relationship"|2023/05/20|yywfIrx_0|Assistant|Side Hustle School|recommended|5)##
("relationship"|2023/05/20|yywfIrx_0|How I Built This|Side Hustle School|Similar to|7)##

##END##

######################
-Real Text-
######################
Text:
{text}

################
Output:"""

QUERY_ENTITY_EXTRACTION_PROMPT = """You are a helpful assistant trying to extract entities from a given user query.

-Goal-
Given a user query and the query date, process identified temporal information and extract all entities of pre-defined entity types from the text.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized. This is a user query to the assistant so words like "I", "me", "my" refer to the user and "you", "your" refer to the assistant.
- entity_type: ONLY from one of the following types: [person, time, organization, location, event, concept, object]
Format each entity as ("entity"|<entity_name>|<entity_type>)
Note: For entities that are related to the query time, process the temporal information and extract the result as a time entity.

2. Return output in English as a single list of all the entities identified in steps 1. Use **##** as the list delimiter.

3. When finished, output ##END##

######################
-Examples-
######################
Example 1:

Query:
What coffee did you recommend me on January 15th?
Query time:
2023/02/01 (Wed) 10:20
################
Output:
("entity"|User|person)##
("entity"|Assistant|person)##
("entity"|Coffee|object)##
("entity"|January 15th|time)##
##END##

######################
Example 2:

Query:
What podcast did my sister recommend last Saturday?
Query time:
2024/07/12 (Mon) 8:20
################
Output:
("entity"|User|person)##
("entity"|Podcast|object)##
("entity"|Sister|person)##
("entity"|July 10th|time)##
##END##

######################
Example 3:

Query:
Who gave me a new stand mixer as a birthday gift last year?
Query time:
2022/05/12 (Fri) 12:20
################
Output:
("entity"|User|person)##
("entity"|Stand Mixer|object)##
("entity"|Birthday Gift|object)##
("entity"|2021|time)##
##END##

######################
-Real Text-
######################
Query:
{text}
Query time:
{session_time}

################
Output:"""

LOCOMO_EXTRACTION_PROMPT = """You are a helpful assistant trying to extract entities and relations from a given multi-turn chat transcript.

-Goal-
Given a multi-turn chat transcript in json format, identify all entities of pre-defined entity types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity in original format, capitalized. Name of the speaker MUST be among the entities.
- entity_type: ONLY from one of the following types: [person, time, organization, location, event, concept, object]. Note that temporal information should be processed and extracted as a time entity.
Format each entity as ("entity"|<entity_name>|<entity_type>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other. FOCUS MORE on relationships clarifying speakers' personal information and preferences.
For each pair of related entities, extract the following information:
- create_time: copy from the <session_time> in the provided json corpus
- session_id: copy from the <session_id> in the provided json corpus
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_name: how the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity from 1 and 10. User related relationships should have a higher strength.
Format each relationship as ("relationship"|<create_time>|<session_id>|<source_entity>|<target_entity>|<relationship_name>|<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.

4. When finished, output ##END##

######################
-Example-
######################
Text:
'text': 'Caroline: Hey Melanie! Take a look at this. (attached is a photo of a woman holding a necklace with a cross and a heart)
Melanie: Hey, Caroline! Nice to hear from you! Love the necklace!
Caroline: Thanks! This necklace is super special to me - a gift from my grandma in my home country, Sweden.
Melanie: It's so pretty! This reminds me, I will swing by and return you the earing I borrowed from you yesterday.'
'session_id': 'D1',
'session_time': '2023/05/20',

################
("entity"|Caroline|person)##
("entity"|Melanie|person)##
("entity"|Grandma|person)##
("entity"|Earing|object)##
("entity"|Necklace|object)##
("entity"|Sweden|location)##
("entity"|May 19th|time)##
("relationship"|2023/05/20|D1|Caroline|Necklace|owns|10)##
("relationship"|2023/05/20|D1|Caroline|Sweden|is from|9)##
("relationship"|2023/05/20|D1|Grandma|Caroline|gave the necklace to|9)##
("relationship"|2023/05/20|D1|Caroline|Melanie|sent a photo of a necklace to|9)##
("relationship"|2023/05/20|D1|Melanie|Caroline|borrowed an earing from|9)##
("relationship"|2023/05/20|D1|Melanie|May 19th|borrowed an earing on|9)##
##END##

######################
-Real Text-
######################
Text:
{text}

################
Output:"""
