from yachalk import chalk
import sys
import os
sys.path.append(os.path.abspath("."))
import dotenv
import json
#import ollama.client as client 
from openai import AzureOpenAI

dotenv.load_dotenv()

import helpers.clean_data as clean_data 




client=AzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        azure_endpoint=os.getenv('AZURE_OpenAI_ENDPOINT'),
        api_version=os.getenv('AZURE_OpenAI_API_VERSION')
    )


def extractConcepts(prompt:str, metadata={},model='gpt-4'):
    SYS_PROMPT=(
         "Your task is extract the key concepts (and non personal entities) mentioned in the given context. "
        "Extract only the most important and atomistic concepts, if  needed break the concepts down to the simpler concepts."
        "Categorize the concepts in one of the following categories: "
        "[event, concept, place, object, document, organisation, condition, misc]\n"
        "Format your output as a list of json with the following format:\n"
        "[\n"
        "   {\n"
        '       "entity": The Concept,\n'
        '       "importance": The concontextual importance of the concept on a scale of 1 to 5 (5 being the highest),\n'
        '       "category": The Type of Concept,\n'
        "   }, \n"
        "{ }, \n"
        "]\n"
       
    )
    client=AzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        azure_endpoint=os.getenv('AZURE_OpenAI_ENDPOINT'),
        api_version=os.getenv('AZURE_OpenAI_API_VERSION')
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role':"system","content":SYS_PROMPT},
            {'role':'user',"content":prompt}
        ],
        temperature=0.5,
        max_tokens=250
    )

    try:
        result=json.loads(response)
        result=[dict(item,**metadata) for item in result]
        print(result)
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result=None
    return result





# for testing extractConcepts function

# data=""" 
# Elon Reeve Musk FRS (/ˈiːlɒn/; born June 28, 1971) is a businessman and investor known for his key roles in the space company SpaceX and the automotive company Tesla, Inc. Other involvements include ownership of X Corp., the company that operates the social media platform X (formerly known as Twitter), and his role in the founding of the Boring Company, xAI, Neuralink, and OpenAI. He is one of the wealthiest individuals in the world; as of August 2024 Forbes estimates his net worth to be US$247 billion.[3]

# Musk was born in Pretoria, South Africa, to Maye (née Haldeman), a model, and Errol Musk, a businessman and engineer. Musk briefly attended the University of Pretoria before immigrating to Canada at the age of 18, acquiring citizenship through his Canadian-born mother. Two years later he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University, but dropped out after two days and, with his brother Kimbal, co-founded the online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999. That same year Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal. In 2002 Musk acquired US citizenship. That October eBay acquired PayPal for $1.5 billion. Using $100 million of the money he made from the sale of PayPal, Musk founded SpaceX, a spaceflight services company, in 2002.

# In 2004, Musk was an early investor who provided most of the initial financing in the electric-vehicle manufacturer Tesla Motors, Inc. (later Tesla, Inc.), assuming the position of the company's chairman. He later became the product architect and, in 2008, the CEO. In 2006, Musk helped create SolarCity, a solar energy company that was acquired by Tesla in 2016 and became Tesla Energy. In 2013, he proposed a hyperloop high-speed vactrain transportation system. In 2015, he co-founded OpenAI, a nonprofit artificial intelligence research company. The following year Musk co-founded Neuralink, a neurotechnology company developing brain–computer interfaces, and The Boring Company, a tunnel construction company. In 2018 the U.S. Securities and Exchange Commission (SEC) sued Musk, alleging that he had falsely announced that he had secured funding for a private takeover of Tesla. To settle the case Musk stepped down as the chairman of Tesla and paid a $20 million fine. In 2022, he acquired Twitter for $44 billion, merged the company into the newly-created X Corp. and rebranded the service as X the following year. In March 2023, Musk founded xAI, an artificial-intelligence company.

# Musk's actions and expressed views have made him a polarizing figure.[4] He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation, promoting right-wing conspiracy theories, and "endorsing an antisemitic theory"; he later apologized for the last of these.[5][4][6] His ownership of Twitter has been controversial because of the layoffs of large numbers of employees, an increase in hate speech, misinformation and disinformation posts on the website, and changes to website features, including verification. Musk has been active in American politics as a vocal and financial supporter of Donald Trump, becoming Trump's second-largest individual donor in October 2024.

# """
#data=clean_data.clean_text(data)
#extractConcepts(data)
def graphPrompt(input: str, metadata={}, model="gpt-4"):
    if model == None:
        model = "mistral-openorca:latest"

    # model_info = client.show(model_name=model)
    # print( chalk.blue(model_info))

    SYS_PROMPT = (
    "You are a network graph maker who extracts terms and their relations from a given context. "
    "You are provided with a context chunk (delimited by ```), and your task is to extract the ontology "
    "of terms mentioned in the given context. These terms should represent the key concepts as per the context.\n"
    
    "Thought 1: While traversing through each sentence, think about the key terms mentioned in it.\n"
    "\tTerms may include object, entity, location, organization, person, \n"
    "\tcondition, acronym, documents, service, concept, etc.\n"
    "\tTerms should be as atomistic as possible.\n\n"
    
    "Thought 2: Consider how these terms can have one-on-one relations with other terms.\n"
    "\tTerms mentioned in the same sentence or paragraph are typically related to each other.\n"
    "\tTerms can be related to many other terms.\n\n"
    
    "Thought 3: Identify the relation between each related pair of terms. \n\n"
    
    "Format your output as a list of JSON objects, with a maximum of 5 term pairs. Each object should contain "
    "a pair of terms and their relationship, like the following:\n"
    "[\n"
    "   {\n"
    '       "node_1": "A concept from extracted ontology",\n'
    '       "node_2": "A related concept from extracted ontology",\n'
    '       "edge": "Relationship between node_1 and node_2 in one concise sentence"\n'
    "   }\n"
    "]"
)



    # USER_PROMPT = f"context: ```{input}``` \n\n output: "
    response=client.chat.completions.create(
        model=model,
        messages=[
            {'role':"system","content":SYS_PROMPT},
            {'role':'user',"content":input}
        ],
        temperature=0.5,
        max_tokens=250,
    )
    try:
        # print(response.choices[0].message.content)
        result = json.loads(response.choices[0].message.content)
        result = [dict(item, **metadata) for item in result]
        
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result

# def graphPrompt(input:str,metadata={},model='gpt-4'):
#     SYS_PROMPT=(
#       "You are a network graph maker who extracts terms and their relations from a given context. "
#         "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
#         "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
#         "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
#             "\tTerms may include object, entity, location, organization, person, \n"
#             "\tcondition, acronym, documents, service, concept, etc.\n"
#             "\tTerms should be as atomistic as possible\n\n"
#         "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
#             "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
#             "\tTerms can be related to many other terms\n\n"
#         "Thought 3: Find out the relation between each such related pair of terms. \n\n"
#         "Format your output as a list of json. Each element of the list contains a pair of terms"
#         "and the relation between them, like the follwing: \n"
#         "[\n"
#         "   {\n"
#         '       "node_1": "A concept from extracted ontology",\n'
#         '       "node_2": "A related concept from extracted ontology",\n'
#         '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
#         "   }, {...}\n"
#         "]"  
#     )

#     #USER_PROMPT=f'context:```{input}``` \n\n output: '
#     response=client.chat.completions.create(
#         model=model,
#         messages=[
#             {'role':"system","content":SYS_PROMPT},
#             {'role':'user',"content":input}
#         ],
#         temperature=0.5,
#         max_tokens=100,
#     )

#     # print(response)
#     try:
#         result=json.loads(response)
#         result=[dict(item,**metadata) for item in result]
        
#     except:
#         print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
#         result=None
#     return result

#for testing graphPrompt function

data=""" 
Elon Reeve Musk FRS (/ˈiːlɒn/; born June 28, 1971) is a businessman and investor known for his key roles in the space company SpaceX and the automotive company Tesla, Inc. Other involvements include ownership of X Corp., the company that operates the social media platform X (formerly known as Twitter), and his role in the founding of the Boring Company, xAI, Neuralink, and OpenAI. He is one of the wealthiest individuals in the world; as of August 2024 Forbes estimates his net worth to be US$247 billion.[3]

Musk was born in Pretoria, South Africa, to Maye (née Haldeman), a model, and Errol Musk, a businessman and engineer. Musk briefly attended the University of Pretoria before immigrating to Canada at the age of 18, acquiring citizenship through his Canadian-born mother. Two years later he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University, but dropped out after two days and, with his brother Kimbal, co-founded the online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999. That same year Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal. In 2002 Musk acquired US citizenship. That October eBay acquired PayPal for $1.5 billion. Using $100 million of the money he made from the sale of PayPal, Musk founded SpaceX, a spaceflight services company, in 2002.

In 2004, Musk was an early investor who provided most of the initial financing in the electric-vehicle manufacturer Tesla Motors, Inc. (later Tesla, Inc.), assuming the position of the company's chairman. He later became the product architect and, in 2008, the CEO. In 2006, Musk helped create SolarCity, a solar energy company that was acquired by Tesla in 2016 and became Tesla Energy. In 2013, he proposed a hyperloop high-speed vactrain transportation system. In 2015, he co-founded OpenAI, a nonprofit artificial intelligence research company. The following year Musk co-founded Neuralink, a neurotechnology company developing brain–computer interfaces, and The Boring Company, a tunnel construction company. In 2018 the U.S. Securities and Exchange Commission (SEC) sued Musk, alleging that he had falsely announced that he had secured funding for a private takeover of Tesla. To settle the case Musk stepped down as the chairman of Tesla and paid a $20 million fine. In 2022, he acquired Twitter for $44 billion, merged the company into the newly-created X Corp. and rebranded the service as X the following year. In March 2023, Musk founded xAI, an artificial-intelligence company.

Musk's actions and expressed views have made him a polarizing figure.[4] He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation, promoting right-wing conspiracy theories, and "endorsing an antisemitic theory"; he later apologized for the last of these.[5][4][6] His ownership of Twitter has been controversial because of the layoffs of large numbers of employees, an increase in hate speech, misinformation and disinformation posts on the website, and changes to website features, including verification. Musk has been active in American politics as a vocal and financial supporter of Donald Trump, becoming Trump's second-largest individual donor in October 2024.

"""
data=clean_data.clean_text(data)
graphPrompt(data)