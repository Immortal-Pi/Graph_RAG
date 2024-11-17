
import os
import sys
sys.path.append(os.path.abspath("."))
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import AzureChatOpenAI
import re 
from langchain_community.graphs import Neo4jGraph
import dotenv
from helpers.df_helpers import documents2Dataframe
from langchain_community.graphs import Neo4jGraph
from helpers.clean_data import clean_text
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import numpy as np 
from helpers.df_helpers import documents2Dataframe, df2Graph,df2ConceptsList,concepts2Df,graph2Df
import pandas as pd






# dotenv.load_dotenv()
# graph=Neo4jGraph(url=os.getenv('NEO4J_URI_ONLINE'),
#                  username=os.getenv('NEO4J_USERNAME_ONLINE'),
#                  password=os.getenv('NEO4J_PASSWORD_ONLINE')
#                 #  database=os.getenv('NEO4J_DATABASE')
# )

llm=AzureChatOpenAI(
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_MODEL'),
    api_version=os.getenv('AZURE_OpenAI_API_VERSION'),
    temperature=0,
    
)

# data=""" 
# A train (from Old French trahiner, from Latin trahere, "to pull, to draw"[1]) is a series of connected vehicles that run along a railway track and transport people or freight. Trains are typically pulled or pushed by locomotives (often known simply as "engines"), though some are self-propelled, such as multiple units or railcars. Passengers and cargo are carried in railroad cars, also known as wagons or carriages. Trains are designed to a certain gauge, or distance between rails. Most trains operate on steel tracks with steel wheels, the low friction of which makes them more efficient than other forms of transport.

# Trains have their roots in wagonways, which used railway tracks and were powered by horses or pulled by cables. Following the invention of the steam locomotive in the United Kingdom in 1802, trains rapidly spread around the world, allowing freight and passengers to move over land faster and cheaper than ever possible before. Rapid transit and trams were first built in the late 1800s to transport large numbers of people in and around cities. Beginning in the 1920s, and accelerating following World War II, diesel and electric locomotives replaced steam as the means of motive power. Following the development of cars, trucks, and extensive networks of highways which offered greater mobility, as well as faster airplanes, trains declined in importance and market share, and many rail lines were abandoned. The spread of buses led to the closure of many rapid transit and tram systems during this time as well.

# Since the 1970s, governments, environmentalists, and train advocates have promoted increased use of trains due to their greater fuel efficiency and lower greenhouse gas emissions compared to other modes of land transport. High-speed rail, first built in the 1960s, has proven competitive with cars and planes over short to medium distances. Commuter rail has grown in importance since the 1970s as an alternative to congested highways and a means to promote development, as has light rail in the 21st century. Freight trains remain important for the transport of bulk commodities such as coal and grain, as well as being a means of reducing road traffic congestion by freight trucks.

# While conventional trains operate on relatively flat tracks with two rails, a number of specialized trains exist which are significantly different in their mode of operation. Monorails operate on a single rail, while funiculars and rack railways are uniquely designed to traverse steep slopes. Experimental trains such as high speed maglevs, which use magnetic levitation to float above a guideway, are under development in the 2020s and offer higher speeds than even the fastest conventional trains. Trains which use alternative fuels such as natural gas and hydrogen are another 21st-century development.

# """

url_input=("https://en.wikipedia.org/wiki/Mark_Zuckerberg")
loader=WebBaseLoader([url_input])
data=loader.load().pop().page_content
# clean the data


data=clean_text(data)
documents=[Document(page_content=data)]

# split the data into chunks 
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
    )
pages=splitter.split_documents(documents)



## creating dataframes of chunks 
df=documents2Dataframe(pages)
#print(df)


regenerate=True
data_dir='cureus'
outputdirectory=Path(f'./data/{data_dir}')
inputdirectory=Path(f'./data/{data_dir}')
if regenerate:
    
    concepts_list=df2Graph(df,model='gpt-35-turbo')
    df1=graph2Df(concepts_list)
    
    if not os.path.exists(outputdirectory):
        os.makedirs(outputdirectory)

    df1.to_csv(outputdirectory/'graph.csv',sep='|',index=False)
    df.to_csv(outputdirectory/'chunks.csv',sep='|',index=False)
else:
    df1=pd.read_csv(outputdirectory/'graph.csv',sep='|')
df1.replace('',np.nan,inplace=True)
df1.dropna(subset=['node_1','node_2','edge'],inplace=True)
df1['count']=1
print(df1)

##try to insert this into the graph db