from trafilatura import fetch_url, extract
from langchain.schema import Document

class WebTextExtractor:
    def __init__(self, url):
        if isinstance(url, list):  # Ensure it's a string
            url = url[0]
            self.url = url.strip('"')  
        

    def extract_all(self):
        documents = []
        downloaded = fetch_url(self.url)
        if not downloaded:
            print(f"Failed to fetch content from: {self.url}")
            return documents 
        if downloaded:
                content = extract(downloaded, include_comments=False, include_tables=True)
                if content:
                    documents.append(Document(page_content=content, metadata={"source": self.url}))

        return documents