import uuid 
import pandas as pd
import numpy as np
from .prompts import extractConcepts
from .prompts import graphPrompt
import time
from openai.error import RateLimitError

def documents2Dataframe(documents)->pd.DataFrame:
    rows=[]
    
    for chunk in documents:
        
        row={
            'text':chunk.page_content,
            # 'source':chunk.metadata,
            **chunk.metadata,
            'chunk_id':uuid.uuid4().hex
        }
        rows=rows+[row]
    df=pd.DataFrame(rows)
    return df


def df2ConceptsList(dataframe:pd.DataFrame)->list:
    results=dataframe.apply(
        lambda row:extractConcepts(
            row.text, {'chunk_id':row.chunk_id,'type':'concept'}
        ),
        axis=1,
    )

    # invalidate json resluts in NaN
    results=results.dropna()
    results=results.reset_index(drop=True)

    ## Flatten the lists to one single list of entities 
    concepts_list=np.concatenate(results).ravel().tolist()
    return concepts_list


def concepts2Df(concepts_List)->pd.DataFrame:
    concepts_dataframe=pd.DataFrame(concepts_List).replace(' ',np.nan)
    concepts_dataframe=concepts_dataframe.dropna(subset=['entity'])
    concepts_dataframe['entity']=concepts_dataframe['entity'].apply(
        lambda x:x.lower()
    )
    return concepts_dataframe


def df2Graph(dataframe:pd.DataFrame,model=None)->list:
    try:
        results=dataframe.apply(
        lambda row: graphPrompt(row.text,{'chunk_id':row.chunk_id},model),
        axis=1
        )
    # invalid json results in Nan
    except RateLimitError as e:
        print(f"Rate limit exceeded. Retrying in {9} seconds...")
        time.sleep(9)
    results=results.dropna()
    results=results.reset_index(drop=True)

    ## flatten the list of list to one single list of entities 
    print(results)
    concept_list=np.concatenate(results).ravel().tolist()
    return concept_list


def graph2Df(node_list)->pd.DataFrame:
    graph_dataframe=pd.DataFrame(node_list).replace(' ',np.nan)
    graph_dataframe=graph_dataframe.dropna(subset=['node_1','node_2'])
    graph_dataframe['node_1']=graph_dataframe['node_1'].apply(lambda x: x.lower())
    graph_dataframe['node_2']=graph_dataframe['node_2'].apply(lambda x: x.lower())
    return graph_dataframe