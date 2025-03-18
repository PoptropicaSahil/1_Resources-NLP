# Math Genius Tools
def solve_equation(equation: str) -> str:
    """
    Solves mathematical equations.
    Args:
        equation: str - A mathematical equation to solve
            e.g. "2x + 5 = 13" or "integrate(x^2)"
    Returns:
        str: The solution to the equation
    """
    # In a real implementation, this would use a math solving library
    return f"Solution for equation: {equation}"


def perform_statistics(data: str) -> str:
    """
    Performs statistical analysis on given data.
    Args:
        data: str - Data in format "[1,2,3,4]" or "1,2,3,4"
    Returns:
        str: Statistical analysis including mean, median, mode, std dev
    """
    # In a real implementation, this would use numpy/pandas
    return f"Statistical analysis of: {data}"


# Code Writer Tools
def generate_code(specification: str, language: str) -> str:
    """
    Generates code based on specifications.
    Args:
        specification: str - Description of what the code should do
        language: str - Programming language to use
    Returns:
        str: Generated code
    """
    return f"Generated {language} code for: {specification}"


def review_code(code: str) -> str:
    """
    Reviews code for best practices and potential issues.
    Args:
        code: str - Code to review
    Returns:
        str: Code review comments
    """
    return f"Code review for: {code}"


def transfer_to_math_genius():
    """Transfer to math genius agent for complex calculations."""
    return math_genius_agent


def transfer_to_code_writer():
    """Transfer to code writer agent for programming tasks."""
    return code_writer_agent


def transfer_to_supervisor():
    """Transfer back to supervisor agent."""
    return supervisor_agent


# Agent Definitions
math_genius_agent = Agent(
    name="Math Genius",
    llm="gpt-4o",
    system_message=(
        "You are a mathematical genius capable of solving complex equations "
        "and performing advanced statistical analysis. Always show your work "
        "step by step. If a task is outside your mathematical expertise, "
        "transfer back to the supervisor."
    ),
    tools=[solve_equation, perform_statistics, transfer_to_supervisor],
)

code_writer_agent = Agent(
    name="Code Writer",
    llm="gpt-4o",
    system_message=(
        "You are an expert programmer capable of writing efficient, clean code "
        "and providing detailed code reviews. Always explain your code and "
        "include comments. If a task is outside your programming expertise, "
        "transfer back to the supervisor."
    ),
    tools=[generate_code, review_code, transfer_to_supervisor],
)

supervisor_agent = Agent(
    name="Supervisor",
    llm="gpt-4o",
    system_message=(
        "You are a supervisor agent responsible for delegating tasks to specialized agents. "
        "For mathematical problems, delegate to the Math Genius. "
        "For programming tasks, delegate to the Code Writer. "
        "Ensure all responses are complete and coordinate between agents when needed."
    ),
    tools=[transfer_to_math_genius, transfer_to_code_writer],
)


# Example usage
def process_request(user_input: str):
    """
    Process user request through the multi-agent system.
    Args:
        user_input: str - User's question or request
    Returns:
        List of messages containing the conversation
    """
    messages = [{"role": "user", "content": user_input}]
    response = run_full_turn(supervisor_agent, messages)
    return response.messages


# Interactive testing loop
if __name__ == "__main__":
    print("Welcome to the Multi-Agent System!")
    print("You can interact with a supervisor who will delegate to either:")
    print("1. Math Genius - for mathematical problems")
    print("2. Code Writer - for programming tasks")
    print("Type 'exit', 'quit', or 'bye' to end the conversation\n")

    agent = supervisor_agent
    messages = []

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break

            messages.append({"role": "user", "content": user_input})
            response = run_full_turn(agent, messages)
            agent = response.agent
            messages.extend(response.messages)

        except Exception as e:
            print(f"Error: {str(e)}")
            print("Resetting conversation...")
            messages = []
            agent = supervisor_agent
