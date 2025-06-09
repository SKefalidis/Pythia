PROMPT_GEOSPATIAL_RELATIONS = """    
In GIS a geometry is a representation of a spatial object, such as a point, line, or polygon, that can be used to represent geospatial features like cities, roads, and lakes.
Depending on the context, the same object can be represented by a more simple geometry, such as a point, or by a more complex geometry, such as a polygon.

A geospatial relation is a relationship between two or more entities or classes that that involves their locations, distances, or spatial arrangements.
It describes how entities are related to each other in terms of their geographical positions, boundaries, distances etc.
For a geospatial relationto exist, the entities or classes must have a spatial component, i.e., a geometry.

You are given a question and a set of classes and nodes (entities) from a Knowledge Graph.

Your task is twofold:
- First, you must understand the question and identify any geospatial relations that are referenced or implied between the nodes/classes. 
You will express these relations using a specific syntax which will be explained later.
There might be more nodes or classes than are required, some might not be relevant, and there might be no geospatial relations.
If there is a geospatial relation between unspecified classes or entities, instead of a URI you can use an ALL_CAPS descriptor, such as "CITY", "COUNTRY", "RIVER", "SHOP" etc.
- Second, you must separate the geospatial relations from the original question, and rewrite the question without the geospatial relations, but keeping the rest of the original content intact.
    
The geospatial relations that we care about are:

- "distance": This relation is used to express the distance between two entities or classes.
  syntax: `distance(A, B)` means the distance between A and B.
  example: `distance(<http://knowledge.com/resource/Paris>, <http://knowledge.com/resource/London>)` means the distance between Paris and London.

- "contains": This relation is used to express that one entity or class contains another.
  syntax: `contains(A, B)` means that A contains B.
  example: `contains(<http://knowledge.com/resource/France>, <http://knowledge.com/resource/Paris>)` means that France contains Paris. Paris is entirely within France.
  
- "touch": This relation is used to express that two entities or classes touch each other, meaning that they share a common boundary, but they do not overlap. Only their boundaries touch, but they do not share a common area.
  syntax: `touch(A, B)` means that A touches B.
  example: `touch(<http://knowledge.com/resource/France>, <http://knowledge.com/resource/Spain>)` means that France and Spain touch, they share a common border.
  
- "overlaps": This relation is used to express that two entities or classes overlap, meaning that they share a common area, however small that might be.
This is different from "touch", because "overlaps" means that the two entities or classes share a common area, while "touch" means that they share a common boundary.
It is also more general than "contains", because "contains" means that one entity or class is entirely within another, while "overlaps" means that they share a common area, but one might not be entirely within the other.
It is though possible that one entity or class is entirely within another, and they still overlap.
  syntax: `overlaps(A, B)` means that A overlaps with B.
  example: `overlaps(<htpp://knowledge.com/resource/Turkey>, <http://knowledge.com/resource/Europe>)` means that Turkey overlaps with Europe, they share a common area. This holds
because a part of Turkey is in Europe.

- "crosses": This relation is used to express that one entity or class crosses another, meaning that they intersect in such a way that they share a common area, but they are not entirely within each other.
The fact that they are not entirely within each other is what distinguishes "crosses" from "overlaps".
  syntax: `crosses(A, B)` means that A crosses B.
  example: `crosses(<http://knowledge.com/resource/Nile>, <http://knowledge.com/resource/Egypt>)` means that the Nile crosses Egypt, because it flows through Egypt, but it is not entirely within Egypt.
  
- "north_of": This relation is used to express that one entity or class is located north of another.
  syntax: `north_of(A, B)` means that A is located north of B.
  example: `north_of(<http://knowledge.com/resource/Canada>, <http://knowledge.com/resource/USA>)` means that Canada is located north of the USA.
- "south_of": This relation is used to express that one entity or class is located south of another.
  syntax: `south_of(A, B)` means that A is located south of B.
  example: `south_of(<http://knowledge.com/resource/Argentina>, <http://knowledge.com/resource/USA>)` means that Argentina is located south of the USA.
- "east_of": This relation is used to express that one entity or class is located east of another.
  syntax: `east_of(A, B)` means that A is located east of B.
  example: `east_of(<http://knowledge.com/resource/Australia>, <http://knowledge.com/resource/India>)` means that Australia is located east of India.
- "west_of": This relation is used to express that one entity or class is located west of another.
  syntax: `west_of(A, B)` means that A is located west of B.
  example: `west_of(<http://knowledge.com/resource/Germany>, <http://knowledge.com/resource/Greece>)` means that Germany is located west of Greece.

For the first part of your response you must follow the previously defined syntax.
Each geospatial relation must be expressed as a single line without additional text, in the form of `relation(A, B)`, surrounded by curly braces {{ }}.
You must start the first line with the string # GEOSPATIAL RELATIONS.

For the second part of your response, you must rewrite the question without the context of the identified geospatial relations, but keeping the rest of the original content intact.
You can split the question into multiple sentences, in cases where the question does not make sense without the geospatial relations.
If the question does not contain any geospatial relations, just copy it as is.
If the question is entirely dependent on the geospatial relations, you must write an empty line after the # REWRITTEN QUESTION line.
You must start the second part with the string # REWRITTEN QUESTION.

Here are some examples:
Example A:
    Question: Which objects of class A are located north of B and less than 100 km away from C?
    Classes:
    <http://knowledge.com/ontology/A>
    <http://knowledge.com/ontology/AD>
    Entities:
    <http://knowledge.com/resource/B>
    <http://knowledge.com/resource/C>
    
    Expected output:
    # GEOSPATIAL RELATIONS
    {{north_of(<http://knowledge.com/ontology/A>, <http://knowledge.com/resource/B>)}} 
    {{distance(<http://knowledge.com/ontology/A>, <http://knowledge.com/resource/C>)}}
    # REWRITTEN QUESTION
    
    Explanation:
    The first line expresses that we are looking for objects of class A that are located north of B.
    The second line expresses that we are looking for objects of class A that are less than 100 km away from C.
    Since the question is entirely about geospatial relations, we do not need to rewrite the question, so we can leave it empty.

Example B:
    Question: Which are the two largest Omegas that share a border, and how much P does the largest one have?
    classes:
    <http://knowledge.com/ontology/Omega>
    Entities:
    
    Expected output:
    # GEOSPATIAL RELATIONS
    {{touch(<http://knowledge.com/ontology/Omega>, <http://knowledge.com/ontology/Omega>)}}
    # REWRITTEN QUESTION
    What is the size of the largest Omega?
    
    Explanation:
    We have only one geospatial relation, that we are looking for two Omegas that share a border. 
    Then we need to rewrite the question to keep the original content intact, but without the geospatial relation. 
    The original question is asking for the two largest Omegas that share a border, but since we want to remove the geospatial relation, we rewrite the question to ask for the size of the largest Omega.
    
Example C:
    Question: Which alpha has the most gammas in Omega?
    Classes:
    <http://knowledge.com/ontology/Gamma>
    Entities:
    <http://knowledge.com/resource/Omega>
    
    Expected output:
    # GEOSPATIAL RELATIONS
    {{contains(<http://knowledge.com/ontology/Omega>, ALPHA)}}
    {{contains(ALPHA, <http://knowledge.com/ontology/Gamma>)}}
    
    # REWRITTEN QUESTION
    
    Explanation:
    Since we don't have an entity or class that refers to alpha, we use the ALL_CAPS descriptor ALPHA to represent it in the geospatial relations. We care about alphas that are contained in Omega.
    We also want to count how many gammas are contained in each alpha, so we express that as well. Again we use the ALL_CAPS descriptor ALPHA to represent it in the geospatial relations.
    The original question is entirely dependent on the geospatial relations, so we leave the rewritten question empty.
    
Remember, your task is to identify geospatial relations between the classes and entities, and rewrite the question without the geospatial relations.
        
Question: {sentence}
Classes: 
{classes}
Entities: 
{entities}

Very very important:
You must split your response into two parts, one starting with # GEOSPATIAL RELATIONS and the other starting with # REWRITTEN QUESTION.
We usually rewrite the question when there is some property or relation that is not geospatial. If the question is entirely dependent on the geospatial relations, you must write an empty line after the # REWRITTEN QUESTION line.

Answer:
"""


PROMPT_GEOSPATIAL_RELATIONS_DISTANCE = """    
In GIS a geometry is a representation of a spatial object, such as a point, line, or polygon, that can be used to represent geospatial features like cities, roads, and lakes.
Depending on the context, the same object can be represented by a more simple geometry, such as a point, or by a more complex geometry, such as a polygon.

A geospatial relation is a relationship between two or more entities or classes that that involves their locations, distances, or spatial arrangements.
It describes how entities are related to each other in terms of their geographical positions, boundaries, distances etc.
For a geospatial relationto exist, the entities or classes must have a spatial component, i.e., a geometry.

You are given a question and a set of classes and nodes (entities) from a Knowledge Graph.

Your task is twofold:
- First, you must understand the question and identify any geospatial relations that are referenced or implied between the nodes/classes. 
You will express these relations using a specific syntax which will be explained later.
There might be more nodes or classes than are required, some might not be relevant, and there might be no geospatial relations.
If there is a geospatial relation between unspecified classes or entities, instead of a URI you can use an ALL_CAPS descriptor, such as "CITY", "COUNTRY", "RIVER", "SHOP" etc.
- Second, you must separate the geospatial relations from the original question, and rewrite the question without the geospatial relations, but keeping the rest of the original content intact.
    
The geospatial relations that we care about are:

- "distance": This relation is used to express the distance between two entities or classes.
  syntax: `distance(A, B)` means the distance between A and B.
  example: `distance(<http://knowledge.com/resource/Paris>, <http://knowledge.com/resource/London>)` means the distance between Paris and London.
  
  Distance can also be used to express a specific condition, such as "less than 100 km away from C".
  syntax: `distance(A, C) < NUMBER UNIT_OF_MEASUREMENT` means that the distance between A and C is less than 100 meters.
  example: `distance(<http://knowledge.com/ontology/City>, <http://knowledge.com/resource/London>) < 100 km` means that the City must be less than 100 km away from London.

- "contains": This relation is used to express that one entity or class contains another.
  syntax: `contains(A, B)` means that A contains B.
  example: `contains(<http://knowledge.com/resource/France>, <http://knowledge.com/resource/Paris>)` means that France contains Paris. Paris is entirely within France.
  
- "touch": This relation is used to express that two entities or classes touch each other, meaning that they share a common boundary, but they do not overlap. Only their boundaries touch, but they do not share a common area.
  syntax: `touch(A, B)` means that A touches B.
  example: `touch(<http://knowledge.com/resource/France>, <http://knowledge.com/resource/Spain>)` means that France and Spain touch, they share a common border.
  
- "overlaps": This relation is used to express that two entities or classes overlap, meaning that they share a common area, however small that might be.
This is different from "touch", because "overlaps" means that the two entities or classes share a common area, while "touch" means that they share a common boundary.
It is also more general than "contains", because "contains" means that one entity or class is entirely within another, while "overlaps" means that they share a common area, but one might not be entirely within the other.
It is though possible that one entity or class is entirely within another, and they still overlap.
  syntax: `overlaps(A, B)` means that A overlaps with B.
  example: `overlaps(<htpp://knowledge.com/resource/Turkey>, <http://knowledge.com/resource/Europe>)` means that Turkey overlaps with Europe, they share a common area. This holds
because a part of Turkey is in Europe.

- "crosses": This relation is used to express that one entity or class crosses another, meaning that they intersect in such a way that they share a common area, but they are not entirely within each other.
The fact that they are not entirely within each other is what distinguishes "crosses" from "overlaps".
  syntax: `crosses(A, B)` means that A crosses B.
  example: `crosses(<http://knowledge.com/resource/Nile>, <http://knowledge.com/resource/Egypt>)` means that the Nile crosses Egypt, because it flows through Egypt, but it is not entirely within Egypt.
  
- "north_of": This relation is used to express that one entity or class is located north of another.
  syntax: `north_of(A, B)` means that A is located north of B.
  example: `north_of(<http://knowledge.com/resource/Canada>, <http://knowledge.com/resource/USA>)` means that Canada is located north of the USA.
- "south_of": This relation is used to express that one entity or class is located south of another.
  syntax: `south_of(A, B)` means that A is located south of B.
  example: `south_of(<http://knowledge.com/resource/Argentina>, <http://knowledge.com/resource/USA>)` means that Argentina is located south of the USA.
- "east_of": This relation is used to express that one entity or class is located east of another.
  syntax: `east_of(A, B)` means that A is located east of B.
  example: `east_of(<http://knowledge.com/resource/Australia>, <http://knowledge.com/resource/India>)` means that Australia is located east of India.
- "west_of": This relation is used to express that one entity or class is located west of another.
  syntax: `west_of(A, B)` means that A is located west of B.
  example: `west_of(<http://knowledge.com/resource/Germany>, <http://knowledge.com/resource/Greece>)` means that Germany is located west of Greece.

For the first part of your response you must follow the previously defined syntax.
Each geospatial relation must be expressed as a single line without additional text, in the form of `relation(A, B)`, surrounded by curly braces {{ }}.
You must start the first line with the string # GEOSPATIAL RELATIONS.

For the second part of your response, you must rewrite the question without the context of the identified geospatial relations, but keeping the rest of the original content intact.
You can split the question into multiple sentences, in cases where the question does not make sense without the geospatial relations.
If the question does not contain any geospatial relations, just copy it as is.
If the question is entirely dependent on the geospatial relations, you must write an empty line after the # REWRITTEN QUESTION line.
You must start the second part with the string # REWRITTEN QUESTION.

Here are some examples:
Example A:
    Question: Which objects of class A are located north of B and less than 100 km away from C?
    Classes:
    <http://knowledge.com/ontology/A>
    <http://knowledge.com/ontology/AD>
    Entities:
    <http://knowledge.com/resource/B>
    <http://knowledge.com/resource/C>
    
    Expected output:
    # GEOSPATIAL RELATIONS
    {{north_of(<http://knowledge.com/ontology/A>, <http://knowledge.com/resource/B>)}} 
    {{distance(<http://knowledge.com/ontology/A>, <http://knowledge.com/resource/C>)}}
    # REWRITTEN QUESTION
    
    Explanation:
    The first line expresses that we are looking for objects of class A that are located north of B.
    The second line expresses that we are looking for objects of class A that are less than 100 km away from C.
    Since the question is entirely about geospatial relations, we do not need to rewrite the question, so we can leave it empty.

Example B:
    Question: Which are the two largest Omegas that share a border, and how much P does the largest one have?
    classes:
    <http://knowledge.com/ontology/Omega>
    Entities:
    
    Expected output:
    # GEOSPATIAL RELATIONS
    {{touch(<http://knowledge.com/ontology/Omega>, <http://knowledge.com/ontology/Omega>)}}
    # REWRITTEN QUESTION
    What is the size of the largest Omega?
    
    Explanation:
    We have only one geospatial relation, that we are looking for two Omegas that share a border. 
    Then we need to rewrite the question to keep the original content intact, but without the geospatial relation. 
    The original question is asking for the two largest Omegas that share a border, but since we want to remove the geospatial relation, we rewrite the question to ask for the size of the largest Omega.
    
Example C:
    Question: Which alpha has the most gammas in Omega?
    Classes:
    <http://knowledge.com/ontology/Gamma>
    Entities:
    <http://knowledge.com/resource/Omega>
    
    Expected output:
    # GEOSPATIAL RELATIONS
    {{contains(<http://knowledge.com/ontology/Omega>, ALPHA)}}
    {{contains(ALPHA, <http://knowledge.com/ontology/Gamma>)}}
    
    # REWRITTEN QUESTION
    
    Explanation:
    Since we don't have an entity or class that refers to alpha, we use the ALL_CAPS descriptor ALPHA to represent it in the geospatial relations. We care about alphas that are contained in Omega.
    We also want to count how many gammas are contained in each alpha, so we express that as well. Again we use the ALL_CAPS descriptor ALPHA to represent it in the geospatial relations.
    The original question is entirely dependent on the geospatial relations, so we leave the rewritten question empty.
    
Remember, your task is to identify geospatial relations between the classes and entities, and rewrite the question without the geospatial relations.
        
Question: {sentence}
Classes: 
{classes}
Entities: 
{entities}

Very very important:
You must split your response into two parts, one starting with # GEOSPATIAL RELATIONS and the other starting with # REWRITTEN QUESTION.
We usually rewrite the question when there is some property or relation that is not geospatial. If the question is entirely dependent on the geospatial relations, you must write an empty line after the # REWRITTEN QUESTION line.

Answer:
"""