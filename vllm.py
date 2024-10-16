from flask import Flask, request, jsonify
from vllm import LLM, SamplingParams

# Initialize the Flask app
app = Flask(__name__)

# Load the vLLM model (you can specify any pre-trained model available in vLLM)
llm = LLM(model="gpt2")  # You can change "gpt2" to a larger model if desired

@app.route("/generate", methods=["POST"])
def generate():
    # Extract the input text from the request
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    # Define sampling parameters
    sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
    
    # Generate the response from the model
    output = llm.generate(prompt, sampling_params)
    
    # Return the output in a JSON response
    return jsonify({
        "input": prompt,
        "output": output[0].text.strip()  # Get the generated text from the first result
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
