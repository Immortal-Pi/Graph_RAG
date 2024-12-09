import asyncio
import crawl4ai
from crawl4ai import AsyncWebCrawler
from crawl4ai import WebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field
import os
import dotenv
import json
import sys

dotenv.load_dotenv()


async def main(url):
    async with AsyncWebCrawler() as crawler:
        result=await crawler.arun(url=url)
        
        return result


if __name__=='__main__':
    if len(sys.argv)>1:
        url=sys.argv[1]
    asyncio.run(main(url))
    
    
    
