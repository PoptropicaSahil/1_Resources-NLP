from pydantic import BaseModel
# from .. import run_llm

class Agent(BaseModel):
    name: str = "Agent"
    llm: str = "gpt-4o-mini"
    system_message: str = "You are a helpful Agent"
    tools: list = []



def add_numbers(a: float, b: float) -> float:
    """
    Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return a + b


def subtract_numbers(a: float, b: float) -> float:
    """
    Subtract b from a.

    Args:
        a: First number
        b: Second number

    Returns:
        The result of a - b
    """
    return a - b


def multiply_numbers(a: float, b: float) -> float:
    """
    Multiply two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The product of a and b
    """
    return a * b


def divide_numbers(a: float, b: float) -> float:
    """
    Divide a by b.

    Args:
        a: First number (dividend)
        b: Second number (divisor)

    Returns:
        The result of a / b

    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


react_system_message = """You are a mathematical reasoning agent that follows the ReAct pattern: Thought, Action, Observation.

For each step of your reasoning:
1. Thought: First explain your thinking
2. Action: Then choose and use a tool
3. Observation: Observe the result

Format your responses as:
Thought: <your reasoning>
Action: <tool_name>(<parameters>)
Observation: <result>
Thought: <your next step>
...

Available tools:
- add_numbers(a, b): Add two numbers
- subtract_numbers(a, b): Subtract b from a
- multiply_numbers(a, b): Multiply two numbers
- divide_numbers(a, b): Divide a by b

Always break down complex calculations into steps using this format."""

react_agent = Agent(
    name="ReActMath",
    llm="gpt-4o-mini",
    system_message=react_system_message,
    tools=[add_numbers, subtract_numbers, multiply_numbers, divide_numbers],
)


# Example usage
messages = [{"role": "user", "content": "Calculate (23 + 7) * 3 - 15"}]

response = run_agent(react_agent, messages)

# ReActMath: Thought: First, I need to calculate the sum of 23 and 7. Then I will multiply the result by 3, and finally, I will subtract 15 from that product. I'll break this down into steps for clarity.

# Action: I will first add 23 and 7.
# functions.add_numbers({ a: 23, b: 7 })

# Observation: Let's perform the addition.
# ReActMath: add_numbers({'a': 23, 'b': 7})
# ReActMath: Thought: The sum of 23 and 7 is 30. Now, I will multiply this result by 3.

# Action: I will multiply 30 by 3.
# functions.multiply_numbers({ a: 30, b: 3 })

# Observation: Let's perform the multiplication.
# ReActMath: multiply_numbers({'a': 30, 'b': 3})
# ReActMath: Thought: The product of 30 and 3 is 90. Now, I need to subtract 15 from this result.

# Action: I will subtract 15 from 90.
# functions.subtract_numbers({ a: 90, b: 15 })

# Observation: Let's perform the subtraction.
# ReActMath: subtract_numbers({'a': 90, 'b': 15})
# ReActMath: Thought: The result of subtracting 15 from 90 is 75. Therefore, the final result of the calculation (23 + 7) * 3 - 15 is 75.

# Final Result: 75
