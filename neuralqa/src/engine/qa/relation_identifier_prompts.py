_PROMPT_TASK_EXPLANATION = """
A graph reasoning path is a sequence of connections between entities and classes in a Knowledge Graph that can be used to retrieve a subgraph of the Knowledge Graph relevant to a specific question.
The graph reasoning path essentially describes how to navigate through the graph to collect all the information that is required to answer the question.
A graph reasoning path always begins with a class or an entity and ends with a class, an entity, or a value.

You are given a question and a set of classes and nodes (entities) from a Knowledge Graph.
Your task is to understand the question and identify a valid graph reasoning path to collect the information to be used to answer the question. 
There might be more nodes or classes than are required, so you must try to find only the connections that are required to answer the request.
Because for graph reasoning paths that end with values are not given in the input, you should describe the value."""

_PROMPT_GRAMMAR_RULES ="""
Specifically, you must use the following grammar to express the graph reasoning path:

<statement> ::= <initial_class_or_entity> "->" "member"
              | <initial_class_or_entity> "->" <rhs>

<rhs> ::= <relation_statement>
       | <property_description>
       | <class_or_entity>

<relation_statement> ::= <class_or_entity> "->" <rhs>

<initial_class_or_entity> ::= "<" URI_STRING ">"

<class_or_entity> ::= "<" URI_STRING ">" 
                   | CAPS_IDENTIFIER

<property_description> ::= <string_literal>"""

_PROMPT_GRAMMAR_EXAMPLES = """
## Some examples to clarify the grammar:
    Question: Sergios is the brother of Christina.
    Context given: http://knowledge.com/resource/Sergios, http://knowledge.com/resource/Christina
    Reasoning path: <http://knowledge.com/resource/Sergios> -> <http://knowledge.com/resource/Christina>. This path means get the paths that connect Sergios to Christina.
    In this case the reasoning path consists only of entities. The graph path that best connects these entities will be discovered. 
    Notice how I don't use the "brother" keyword. We only include entities and classes in the graph reasoning path. 
    The "brother" keyword is a property that connects the two entities, so it is not correct to insert it.
    
    Question: Who is the tallest person in the world?
    Context given: http://knowledge.com/ontology/Person
    Reasoning path: <http://knowledge.com/ontology/Person> -> "height". This path means get the property "height" for all the members of the class http://knowledge.com/ontology/Person.
    The reasoning path starts with a class. Classes do not directly have a height. 
    Implicitly, all members of the class will be collected, and then the the predicate that matches "height" the best will be used to rank them.
    This reasoning path ends with a value, we describe the value, in this case the height of the person.
    
    Question: How old is the mother of Sergios' sister?
    Context given: http://knowledge.com/resource/Sergios
    Reasoning path: <http://knowledge.com/resource/Sergios> -> "SISTER" -> "MOTHER" -> "age". This path means get the age of the mother of the sister of Sergios.
    This question is a bit more complex. It starts with an entity that we know, but there is a long reaspning chain to get the answer.
    This chain includes entities that we are not given in the input. 
    Since we do not have URIs for these entities, we write their descriptions in ALL CAPS, separating them from properties/predicates which are written in lowercase. Try to use the ALL CAPS identifiers only when necessary.
    The reasoning path ends with a value, we describe the value, in this case the age of the MOTHER. Tis is a series of <relation_statement> that ends with a <property_statement>.
    Notice how we do not use the "mother" and "sister" properties in the graph reasoning path. We show that Sergios is connected to an entity named "SISTER" and that this entity is connected to another entity named "MOTHER".
    The "mother" and "sister" properties are not used in the reasoning path. 
    We just use the ALL CAPS identifiers to describe the entities that are not given in the input. Again, try to use ALL CAPS identifiers sparingly, only when necessary to convey the meaning of the reasoning path.
    
    Question: Give me 5 books.
    Context given: http://knowledge.com/ontology/Book
    Reasoning path: <http://knowledge.com/ontology/Book> -> "member". This path means get the members of the class http://knowledge.com/ontology/Book.
    Here we do not have a exactly a reasoning path. We just want to get the members of the class. In this case, we just use the "member" keyword to indicate that. We use "member" because this is the sole output of the reasoning path.
    And otherwise we could not make a reasoning path. The use of "member" is strictly necessary in this case and also only allowed in cases like this."""

_PROMPT_TASK_EXAMPLES = """
## Here are some more complete examples. The examples include classes and entities that are not used in the reasoning path, as well as the expected input and output:
Example A:
    Question: Which books were written by J.R.R. Tolkien?
    Classes:
    http://knowledge.com/ontology/Book
    http://knowledge.com/ontology/Author
    http://knowledge.com/ontology/Person
    Entities:
    http://knowledge.com/resource/J.R.R._Tolkien
    http://knowledge.com/resource/The_Lord_of_the_Rings
    http://knowledge.com/resource/The_Hobbit
    
    Expected output:
    {http://knowledge.com/Book -> http://knowledge.com/resource/J.R.R._Tolkien} because we are searching for books written by J.R.R. Tolkien.
    This is a complete graph reasoning path to answer the given question. Notice how we do not use the "written by" property in the reasoning path, it is implicit in the connection between the book and the author.
    We also do not use the "member" keyword, it is implied that we will retrieve all books, and then find the ones that are written by J.R.R. Tolkien. It would be wrong to use the "member" keyword here.

Example B:
    Some questions might contain more connections. 
    They could also need multiple reasoning paths to be answered.
    There could also be alternative graph reasoning paths, but you output a set of paths that are consistent, and part of a single line of thinking.

    Question: Which is the largest capital city in Europe?
    Classes:
    http://knowledge.com/ontology/City
    http://knowledge.com/ontology/Country
    http://knowledge.com/ontology/EuropeanRegion
    Entities:
    http://knowledge.com/resource/Europe
    
    Expected output:
    {http://knowledge.com/ontology/City -> http://knowledge.com/ontology/Country} to find the capital city of every country.
    {http://knowledge.com/ontology/Country -> http://knowledge.com/resource/Europe} to find the countries in Europe.
    {http://knowledge.com/ontology/City -> surface area} to find the largest city.
    The output here is a set of 3 reasoning paths.
    
    Alternatively, you could also have:
    {http://knowledge.com/ontology/City -> http://knowledge.com/ontology/Country -> http://knowledge.com/resource/Europe} to find the capital city of every country in Europe.
    {http://knowledge.com/ontology/City -> surface area} to find the largest city.
    This output is a set of 2 reasoning paths.
    
    Or even:
    {http://knowledge.com/resource/Europe -> http://knowledge.com/ontology/Country -> http://knowledge.com/ontology/City -> surface area} to find the largest city in Europe.
    This output is a single reasoning path.
    
    All of these outputs are valid. You can choose the one that you think is the best. If you are asked to provide an alternative to a given reasoning path, you can provide one of these alternatives."""
    
_PROMPT_PREVIOUS_GENERATIONS = """You have previously generated the following reasoning paths, which are not valid per the grammar.
You must not use these reasoning paths in your final answer. You must try to generate a new reasoning path that is valid per the grammar.
The paths:
{previous_paths}

The grammar is as follows:
{grammar}
"""
    
_PROMPT_REMINDERS = """
Remember, your task is to understand the question and identify a valid graph reasoning path to collect the information to be used to answer the question. 
Remember, you must use only the entities and classes that are given in the input. Do not use any URIs outside the given ones. 
Remember, there might be more nodes or classes than are required, so you must try to find only the connections that are required to answer the request.
Remember, because the values are not given in the input, you should just describe the connecting value.
Remember, a connection between a class and an entity or a class and a property is enough to describe the connection. "member" is applied, and it is not correct to use it in such cases.
Only use "member" when you want to get the members of a class as the sole output of the reasoning path.

Very important:
A connection between an entity and a class is enough to describe the connection. You must not use an ALL CAPS identifiers or "member" keyword to connect them.
You can use ALL CAPS identifiers to describe entities that are not given in the input. Only entities. You can not use ALL CAPS identifiers for anything other than entities. 
You can't start a reasoning path with an ALL CAPS identifier as per the grammar.
You must use ALL CAPS identifiers only when absolutely neccessary and only as a last resort. You must not use ALL CAPS identifiers to describe classes or properties.
"""

_PROMPT_CREATE_GRAPH_REASONING_PATH = """    
{task_explanation}
{grammar_rules}
{grammar_examples}
{task_examples}
    
Your final answer must be a list of reasoning paths, each surrounded by curly braces {{ }}.

{reminders}

{previous_generations}
        
Question: {sentence}
Classes: 
{classes}
Entities: 
{entities}

Very very important:
You must begin the final answer with the string # FINAL ANSWER.
You can't use URIs that are not given in the input.
You must capture all the information that is required to answer the question. The reasoning path is very important to retrieve the information from  the Knowledge Graph.
Your final answer must be just reasoning paths. You can not use any other text. You should representing everything with the reasoning paths.

Answer:
"""