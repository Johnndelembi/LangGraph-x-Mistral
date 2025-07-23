from typing import TypedDict, List, Annotated, Union
import operator
from langgraph.graph import StateGraph, END
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig, AutoConfig
import torch
from dataclasses import dataclass

# ===== 1. Define Models =====
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
MODEL_NAME = "microsoft/DialoGPT-medium"  # Much better for conversation
# MODEL_NAME = "google/flan-t5-base"
# MODEL_NAME = "google/flan-t5-small"

@dataclass
class HFAgent:
    model: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM]  # Fixed type annotation
    tokenizer: AutoTokenizer
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        ) if self.device == "cuda" else None
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        config = AutoConfig.from_pretrained(MODEL_NAME)
        if config.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME,
                device_map="auto" if self.device == "cuda" else "cpu",
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto" if self.device == "cuda" else "cpu",
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
    
    def generate(self, prompt: str, max_tokens=256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Use appropriate generation parameters for DialoGPT
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        outputs = self.model.generate(**inputs, **generation_kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Initialize agents
main_agent = HFAgent()
fact_checker = HFAgent()
moderator = HFAgent()

# ===== 2. Define State =====
class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]  # Format: [{"role": "user|ai", "content": str}]
    needs_correction: bool

# ===== 3. Define Nodes =====
def chatbot_node(state: AgentState):
    """Main response generator"""
    # Get the most recent user message
    user_message = None
    for message in reversed(state["messages"]):
        if message["role"] == "user":
            user_message = message["content"]
            break
    
    if not user_message:
        user_message = "Hello"
    
    # Create a conversation-friendly prompt for DialoGPT
    prompt = f"Human: {user_message}\nAssistant:"
    
    response = main_agent.generate(prompt, max_tokens=150)
    
    # Clean up the response
    response = response.strip()
    if response.startswith("Assistant:"):
        response = response[10:].strip()
    
    return {
        "messages": [{"role": "ai", "content": response}],
        "needs_correction": False
    }

def fact_check_node(state: AgentState):
    """Fact-checking agent"""
    last_msg = state["messages"][-1]["content"]
    
    # Skip fact-checking for simple greetings or questions
    simple_phrases = ["hello", "hi", "hey", "how are you", "what's your name", "who are you"]
    if any(phrase in last_msg.lower() for phrase in simple_phrases):
        print(f"Fact-Check: Skipping simple phrase")
        return {"needs_correction": False}
    
    prompt = f"""Question: Is the following statement a factual claim that can be verified as true or false? Answer with only TRUE or FALSE.

Statement: {last_msg}

Answer:"""
    
    verdict = fact_checker.generate(prompt, max_tokens=5).strip().upper()
    print(f"Fact-Check Verdict: {verdict}")
    
    # Only flag for correction if it's clearly FALSE
    return {"needs_correction": verdict == "FALSE"}

def moderator_node(state: AgentState):
    """Safety check"""
    last_msg = state["messages"][-1]["content"].lower()
    unsafe_phrases = ["harm", "hate", "illegal", "dangerous"]
    
    if any(phrase in last_msg for phrase in unsafe_phrases):
        print("Moderator: Blocked unsafe content!")
        return {
            "messages": [{"role": "ai", "content": "I can't comply with that request."}],
            "needs_correction": False
        }
    return state

# ===== 4. Build Workflow =====
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("fact_checker", fact_check_node)
workflow.add_node("moderator", moderator_node)

# Define edges
workflow.add_edge("chatbot", "fact_checker")
workflow.add_edge("fact_checker", "moderator")

# Conditional self-correction
workflow.add_conditional_edges(
    "moderator",
    lambda state: str(state["needs_correction"]),
    {"True": "chatbot", "False": END}
)

workflow.set_entry_point("chatbot")
agent = workflow.compile()

# ===== 5. Chat Interface =====
def chat():
    print("Multi-Agent Chatbot (type 'quit' to exit)\n")
    state = {"messages": [], "needs_correction": False}
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        
        # Add user message to state
        state["messages"].append({"role": "user", "content": user_input})
        
        # Run the workflow
        state = agent.invoke(state)
        
        # Print last AI message
        print(f"\nAssistant: {state['messages'][-1]['content']}\n")

if __name__ == "__main__":
    chat()