# from yachalk import chalk
import sys
import os
sys.path.append(os.path.abspath("."))
import dotenv
import json
#import ollama.client as client 
from openai import AzureOpenAI
from langchain_groq import ChatGroq
dotenv.load_dotenv()

import helpers.clean_data as clean_data 




client=ChatGroq(
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model_name=os.getenv('GROQ_MODEL')
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
    client=ChatGroq(
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model_name=os.getenv('GROQ_MODEL')
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role':"system","content":SYS_PROMPT},
            {'role':'user',"content":prompt}
        ],
        temperature=0.5,
        
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
   
        """From the biography text of a famous person or celebrity below, extract Entities and Relationships strictly as instructed:
    1. Identify all significant entities and relationships from the text, categorizing each as described below. The `id` property of each entity must be unique and alphanumeric, to ensure distinct nodes in Neo4j. You will be using this `id` property to define relationships between entities. Only use the entity types below; do NOT create new types.

    **Entity Types:**
    - **Person**: label:'Person', id:string, role:string, description:string // Summary of the individual’s primary role or identity.
    - **Achievement**: label:'Achievement', id:string, description:string // Major accomplishments, awards, notable works, or contributions.
    - **Organization**: label:'Organization', id:string, role:string, description:string // Companies, institutions, or groups the person is associated with.
    - **Event**: label:'Event', id:string, description:string // Important events related to the individual (e.g., awards, major milestones).
    - **Relationship**: label:'Relationship', id:string, type:string, source:string, target:string // Defines the connection type (e.g., "Collaborated with," "Founded," "Won") and references `source` and `target` entities by `id`.

    2. **Description Property Requirements**:
    - Each description should be a concise summary, no more than 100 characters, focused on the essential detail.
    - **Person**: Summarize their primary identity (e.g., "Renowned American actor").
    - **Achievement**: Specify the nature of the accomplishment (e.g., "Oscar-winning performance in 'Good Will Hunting'").
    - **Organization**: Indicate the person’s role or connection (e.g., "Co-founder of Tesla, Inc.").
    - **Event**: Highlight the key details of the event (e.g., "Nobel Prize in Physics, 1921").
    - **Relationship**: Clearly describe the connection and indicate its direction, ensuring each relationship is accurately categorized. 

    3. **Rules for Extraction**:
    - Avoid fictional or inferred data; extract only verifiable entities and relationships.
    - Do not create duplicate entities with the same label.
    - Only extract entities directly relevant to the person’s role, achievements, or associations.
    - Avoid personal anecdotes or unrelated information.
    - Avoid incomplete results in JSON format
    - choose only important relationships dont create relationships for simple things like for 'noted', 'met_with'

    4. **Example Output JSON**:
    {
    "entities": [
        {
            "label": "Elon Musk",
            "id": "Elon Musk",
            "role": "Entrepreneur",
            "description": "Visionary entrepreneur and business magnate"
        },
        {
            "label": "NASA",
            "id": "NASA",
            "role": "Partner",
            "description": "National Aeronautics and Space Administration"
        },
        {
            "label": "SpaceX",
            "id": "SpaceX",
            "role": "Founder",
            "description": "Founder and CEO of SpaceX"
        },
        {
            "label": "OrbitMission",
            "id": "OrbitMission",
            "role": "OrbitMission",
            "description": "First private company to send humans to orbit, 2020"
        }
        
    ],
    "Relationships":[
                {
                    "label": "Founded",
                    "id": "relationship1",
                    "type": "Founded",
                    "source": "person1",
                    "target": "organization1",
                    "description": "Elon Musk founded Spacex a spaceflight services company, in 2002"  
                },
                {
                    "label": "Collaborated with",
                    "id": "relationship6",
                    "type": "Collaborated with",
                    "source": "person1",
                    "target": "organization3",
                    "description": "Elon Musk collaborated with NASA for the Crew Dragon mission in 2020"
                }
    
    ]
    }"""

)



    # USER_PROMPT = f"context: ```{input}``` \n\n output: "
    response=client.chat.completions.create(
        model=model,
        messages=[
            {'role':"system","content":SYS_PROMPT},
            {'role':'user',"content":input}
        ],
        temperature=0.1,
        # max_tokens=500,
    )
    # return response.choices[0].message.content
    try:
        print(response.choices[0].message.content)
        result = json.loads(response.choices[0].message.content)
        # result = [dict(item, **metadata) for item in result]
        # print(result)
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result



# data=""" 
# Elon Reeve Musk FRS (/ˈiːlɒn/; born June 28, 1971) is a businessman and investor known for his key roles in the space company SpaceX and the automotive company Tesla, Inc. Other involvements include ownership of X Corp., the company that operates the social media platform X (formerly known as Twitter), and his role in the founding of the Boring Company, xAI, Neuralink, and OpenAI. He is one of the wealthiest individuals in the world; as of August 2024 Forbes estimates his net worth to be US$247 billion.[3]

# Musk was born in Pretoria, South Africa, to Maye (née Haldeman), a model, and Errol Musk, a businessman and engineer. Musk briefly attended the University of Pretoria before immigrating to Canada at the age of 18, acquiring citizenship through his Canadian-born mother. Two years later he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University, but dropped out after two days and, with his brother Kimbal, co-founded the online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999. That same year Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal. In 2002 Musk acquired US citizenship. That October eBay acquired PayPal for $1.5 billion. Using $100 million of the money he made from the sale of PayPal, Musk founded SpaceX, a spaceflight services company, in 2002.

# In 2004, Musk was an early investor who provided most of the initial financing in the electric-vehicle manufacturer Tesla Motors, Inc. (later Tesla, Inc.), assuming the position of the company's chairman. He later became the product architect and, in 2008, the CEO. In 2006, Musk helped create SolarCity, a solar energy company that was acquired by Tesla in 2016 and became Tesla Energy. In 2013, he proposed a hyperloop high-speed vactrain transportation system. In 2015, he co-founded OpenAI, a nonprofit artificial intelligence research company. The following year Musk co-founded Neuralink, a neurotechnology company developing brain–computer interfaces, and The Boring Company, a tunnel construction company. In 2018 the U.S. Securities and Exchange Commission (SEC) sued Musk, alleging that he had falsely announced that he had secured funding for a private takeover of Tesla. To settle the case Musk stepped down as the chairman of Tesla and paid a $20 million fine. In 2022, he acquired Twitter for $44 billion, merged the company into the newly-created X Corp. and rebranded the service as X the following year. In March 2023, Musk founded xAI, an artificial-intelligence company.

# Musk's actions and expressed views have made him a polarizing figure.[4] He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation, promoting right-wing conspiracy theories, and "endorsing an antisemitic theory"; he later apologized for the last of these.[5][4][6] His ownership of Twitter has been controversial because of the layoffs of large numbers of employees, an increase in hate speech, misinformation and disinformation posts on the website, and changes to website features, including verification. Musk has been active in American politics as a vocal and financial supporter of Donald Trump, becoming Trump's second-largest individual donor in October 2024.

# """
# data=clean_data.clean_text(data)
# graphPrompt(data)

system=""" You are a network graph maker tasked with extracting terms and their relations from a given context. Your job is to analyze the provided context chunk (delimited by ```) and extract the ontology of terms mentioned in the context. These terms should represent the key concepts based on the context.

**Guidelines for Extraction:**

1. **Identify Key Terms**:
   - While analyzing the context, identify key terms from each sentence or paragraph.
   - Key terms may include:
     - Objects
     - Entities
     - Locations
     - Organizations
     - Persons
     - Conditions
     - Acronyms
     - Documents
     - Services
     - Concepts, etc.
   - Ensure the terms are as **atomistic** and specific as possible.
   - Ensure to give response only in JSON format and no other texts or anything to indicate the format

2. **Identify Relationships**:
   - Terms mentioned in the same sentence or paragraph are typically related.
   - Each term can have one-on-one relationships with other terms.
   - Determine how terms are related to one another.

3. **Define Relationships**:
   - Clearly identify the nature of the relationship between related terms.
   - Use concise and descriptive language to explain the connection.

**Output Format:**
- Provide your output as a list of JSON objects, with a maximum of 7 term pairs.
- Each object should include:

  - `"node_1"`: A concept extracted from the context.
  - `"node_2"`: A related concept extracted from the context.
  - `"edge"`: A concise sentence describing the relationship between `node_1` and `node_2`.

**Example Output:**
[
   {   
       "node_1": "Climate change",
       "node_2": "Greenhouse gases",
       "edge": "Greenhouse gases contribute significantly to climate change."
   },
   {
       
       "node_1": "Tesla, Inc.",
       "node_2": "Elon Musk",
       "edge": "Elon Musk is the founder and CEO of Tesla, Inc."
   }
]"""  