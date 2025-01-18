from phi.model.groq import Groq

api_key = "your_groq_api_key"
groq_client = Groq(api_key=api_key)

model = Groq(id="llama3-70b-8192")
print(f"Groq model initialized: {model}")
