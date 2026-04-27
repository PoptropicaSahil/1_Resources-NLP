# What Makes Modern Day LLMs Agentic

It's Still Just Next Token Prediction !!

> Reference: [https://amaarora.github.io/posts/2025-09-14-llms-agentic.html](https://amaarora.github.io/posts/2025-09-14-llms-agentic.html)

## Core Insight

**Tool calling is still just next token prediction with specialized tokens** - there's no magic, just clever token patterns learned during training.

## Evolution: 2022 vs 2025

**2022 ChatGPT:**

- Simple prompt completion based on training data
- Could answer questions but couldn't ACT
- No web search, no image generation, no RAG

**2025 ChatGPT:**

- Can search internet, create images, take actions
- "Agentic" through tool calling capabilities
- Same generate loop underneath, different token formatting

## The Generate Loop (Unchanged Since GPT-2)

```py
def generate(context, ntok=20):
    for _ in range(ntok):
        out = model(context)
        logits = out[:, -1, :]
        next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        context = torch.cat([context, next_tok.unsqueeze(-1)], dim=-1)
    return context
```

Same loop today - only difference is how input tokens are formatted.

## Example: Prompt Completion WITHOUT Tools

### Setup

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries."},
  {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
]

# tokenize=False to see formatted prompt as string
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(inputs)
```

### Formatted Input

```
<|im_start|>system
You are a bot that responds to weather queries.<|im_end|>
<|im_start|>user
Hey, what's the temperature in Paris right now?<|im_end|>
<|im_start|>assistant
```

**Key observation:** Special tokens `<|im_start|>` and `<|im_end|>` mark message boundaries.

### Generation

```py
# tokenize=True to get tensors for generation
inputs = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True, 
    tokenize=True,  # Now True to get tensors
    return_tensors="pt", 
    return_dict=True
)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=1024)
print(tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):]))
```

### Output

```
<think>
I need to check weather data for Paris. Since I don't have real-time data, 
I should mention that I can't provide the exact temperature.
</think>

I don't have access to real-time weather data. Could you please ask a different 
question? For the current temperature in Paris, you can check a weather service.
```

**Result:** Model can only respond from training data - not agentic.

## Example: Prompt Completion WITH Tools

### Setup with Tools

```py
def get_current_temperature(location: str, unit: str):
    """
    Get the current temperature at a location.
    
    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    """
    return 22.  # Mock function

def get_current_wind_speed(location: str):
    """
    Get the current wind speed in km/h at a given location.
    
    Args:
        location: The location to get the wind speed for, in the format "City, Country"
    """
    return 6.  # Mock function

tools = [get_current_temperature, get_current_wind_speed]

# Pass tools to template - still tokenize=False to inspect
inputs = tokenizer.apply_chat_template(
    messages, 
    tools=tools, 
    add_generation_prompt=True, 
    tokenize=False
)
print(inputs)
```

### Formatted Input (WITH Tools)

```
<|im_start|>system
You are a bot that responds to weather queries.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_current_temperature", "description": "Get the current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \"City, Country\""}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in."}}, "required": ["location", "unit"]}}}
{"type": "function", "function": {"name": "get_current_wind_speed", ...}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Hey, what's the temperature in Paris right now?<|im_end|>
<|im_start|>assistant
```

**Key insight:** Tools are injected as TEXT in the prompt - just tokens describing available functions.

### Generation

```py
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
inputs = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"  # tokenize=True implied
)
outputs = model.generate(**inputs.to(model.device), max_new_tokens=1024)
print(tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):]))
```

### Output (Tool Call Prediction)

```
<think>
User is asking for temperature in Paris. I need to use get_current_temperature 
function. Parameters required are location and unit. Paris is in France which 
uses Celsius, so I'll set unit to celsius.
</think>

<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "Paris", "unit": "celsius"}}
</tool_call><|im_end|>
```

**Critical insight:** Model isn't "calling" a function - it's **predicting tokens** that happen to be `<tool_call>` followed by JSON. This is a learned pattern from training, just like predicting "Paris" after "The capital of France is".

## Full Workflow: Complete Tool Calling Cycle

### Step 1: Update messages with tool call and response

```py
tool_call = {
    "name": "get_current_temperature", 
    "arguments": {"location": "Paris, France", "unit": "celsius"}
}
messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]})
messages.append({"role": "tool", "content": "22"})  # Tool execution result

# Inspect formatted conversation
inputs = tokenizer.apply_chat_template(
    messages, 
    tools=tools, 
    add_generation_prompt=True, 
    tokenize=False
)
print(inputs)
```

### Formatted Conversation State

```
<|im_start|>system
[...tools definitions...]<|im_end|>
<|im_start|>user
Hey, what's the temperature in Paris right now?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
22
</tool_response><|im_end|>
<|im_start|>assistant
```

**Note:** Tool response is formatted as a user message with `<tool_response>` tags.

### Step 2: Generate final response with tool result

```py
inputs = tokenizer.apply_chat_template(
    messages, 
    tools=tools, 
    add_generation_prompt=True, 
    return_dict=True, 
    return_tensors="pt"
)
out = model.generate(**inputs.to(model.device), max_new_tokens=1024)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))
```

### Final Output

```
<think>
User asked for temperature in Paris. I used get_current_temperature with Paris, 
France, and celsius. The response came back as 22. Now I need to present this 
answer clearly. Should mention city, unit, and temperature.
</think>

The current temperature in Paris is **22°C**. Let me know if you need further details! 🌤️<|im_end|>
```

## Complete Conversation Flow

```
User: Hey, what's the temperature in Paris right now?

Assistant (1st generation):
<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
</tool_call>

[Developer executes tool → returns "22"]

Assistant (2nd generation, with tool result):
The current temperature in Paris is **22°C**. Let me know if you need further details! 🌤️
```

## Key Takeaways

1. **No magic:** Tool calling = next token prediction with specialized tokens like `<tool_call>`, `<tool_response>`
2. **LLMs don't execute tools:** They only predict which tool to call and arguments; developer executes
3. **Training data difference:** Modern LLMs learned to parse structured outputs and choose appropriate tools
4. **Same generate loop:** From GPT-2 to GPT-5, same underlying mechanism
5. **API abstraction:** Behind APIs (like `response.output[0].function_call`), it's all token prediction
6. **tokenize parameter:**
   - `tokenize=False` → inspect formatted string prompt
   - `tokenize=True` (or implied with `return_tensors`) → get tensors for generation
7. **Conversation is stateful:** Tool calls and responses are appended to messages list for next generation

## The Illusion of Agency

What transforms an LLM into an "Agent" is the ability to:

- Recognize when a tool should be called (predict `<tool_call>` tokens)
- Parse required arguments from user query (structured output prediction)
- Incorporate tool results into final response (continue generation after `<tool_response>`)

**Behind the scenes:** It's all learned token patterns, not genuine agency or function execution.
