from langchain.memory import ConversationBufferMemory

class MemoryManager:
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def get_memory(self):
        return self.memory