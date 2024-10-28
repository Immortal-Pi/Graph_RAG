import uuid 
import pandas as pd
import numpy as np
from .prompts import extractConcepts
from .prompts import graphPrompt


def documents2Dataframe(documents)->pd.DataFrame:
    rows=[]
    for chunk in documents:
        row={
            'text':chunk.page_content,
            **chunk.metadata,
            'chunk_id':uuid.uuid4().hex
        }
    df=pd.DataFrame(rows)
    return df