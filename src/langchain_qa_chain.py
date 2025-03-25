from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

class LangchainQAChain:
    def __init__(self, model_path, retriever):
        # Load HuggingFace model and tokenizer
        model_name = "google/flan-t5-small"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        # Custom Prompt Template
        prompt_template = """
        You are a helpful AI assistant. Use the following context to answer the question in short.
        If the answer is not found in the context, say "I don't know." 
        Note: Do not use your internal knowledge to answer.

        Question: {question}

        Context:
        {context}

        Answer:"""

        prompt = PromptTemplate(
            input_variables=["question", "context"],
            template=prompt_template
        )

        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"}  # Explicit context key
        )

    def run(self, question):
        # Manually check retrieved context for debugging
        retrieved_docs = self.chain.retriever.get_relevant_documents(question)
        context_text = "\n".join([doc.page_content for doc in retrieved_docs])  # Convert to plain text

        print(" Final Context Sent to Model:\n", context_text)  # Debugging
        
        result = self.chain.invoke({"query": question})  # Only send query, context is auto-handled
        return result["result"]
