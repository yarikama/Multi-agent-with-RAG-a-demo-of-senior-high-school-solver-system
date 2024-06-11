# Varibles
model_name="llama3:70b"
custom_model_name="crewai-llama3:70b"

# ollama pull $model_name

ollama create $custom_model_name -f ./Modelfile