import openai
import tiktoken

OPENAI_API_KEY="your-openai-api-key" # Replace with your OpenAI API key
STARTING_PROMPT = """Write an interactive story based around Varian Wrynn's past 
                     when his soul was split into Varian and Lo'Gosh, The Ghost Wolf.
                     And tell it from the perspective of either half of his soul, 
                     depending on decisions made early in the story. Reference the
                     actual character, Varian Wrynn, king of Stormwind, from WoW.""" # Write what type of story you want
STORY_DEPTH = 8 # How many levels deep the story tree should go (you will make this many decisions - 1 before the story ends)
NUM_DECISIONS = 3 # How many decisions each node should have
MAX_TOKEN_LIMIT = 500 # limit of tokens per models response (1 token ~ 0.75 words, 100 tokens ~ 75 words)

# Global variables, do not change
TEMPERATURE = 0.7
client = openai.OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-3.5-turbo-0125"
MODEL_INPUT_COST = 0.0005 / 1000
MODEL_OUTPUT_COST = 0.0015 / 1000
GPT4_SYSTEM_INSTRUCTIONS = """
Mission:
You are tasked with generating segments of an interactive story based on the provided context.
Your output must adhere to specific formatting rules to ensure clarity and coherence in the narrative structure.

1. For the root node (the beginning of the story), format your response as an introduction:
   Intro: [Your story introduction here.]

2. For all subsequent nodes, you will generate the specified number of action-outcome pairs based on the current story context.
   Format your responses with actions taken by the player and the outcomes of those actions, as follows:
   Action 1: [The first player's action.]
   Outcome 1 [The outcome of the first action.]
   Action 2: [The second player's action.]
   Outcome 2: [The outcome of the second action.]
   ...
   Ensure you generate a coherent and engaging set of actions and outcomes that branch out from the current story node.

Proper formatting with 'Intro:', 'Action [number]:', and 'Outcome [number]:' is crucial for this mission. 
It ensures that each part of the story can be easily identified and processed for creating an engaging and coherent narrative.

Sample Intro Story Node:
Intro: Once upon a time, in a land far, far away, a young adventurer sets out to discover hidden treasures.

Sample Standard Action-Outcome Story Nodes:
Action: You decide to explore the mysterious cave. Outcome: Inside the cave, they find a treasure chest filled with gold and jewels, but also awaken a sleeping dragon.
Action: You choose to consult the ancient map they found earlier. Outcome: The map reveals a hidden path leading to an undiscovered part of the forest, promising new adventures.

The examples above were merely to demonstrate the formatting. You are encouraged to create much more complex and engaging stories. 
Please consider how many decisions away the end of the story is, so that you can begin to wrap up the story in a satisfying way, 
ensuring a cohesive narrative arc.

Remember, each story segment should offer decision points that lead to meaningful outcomes, 
enriching the interactive experience of the story. Your stories should be engaging, immersive, 
and allow for multiple branching paths and outcomes, giving players a sense of agency and impact on the story's direction.
You should refer to the player as 'you' in your responses to maintain a second-person perspective.
If at any point in time the storyline will use a quote, ensure that it's using '' rather than "".
Do not, for any reason, put newline characters in the middle of an intro, action, or outcome.
The intro should be one continuous line, with proper puctuation, of course.
Likewise, the action and outcome should be together on one continuous line.
Any deviation from these instructions will result in a failed mission.
"""

class Node:
    def __init__(self, content):
        self.content = content
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

class StoryNode:
    def __init__(self, intro=None, action=None, outcome=None):
        self.intro = intro
        self.action = action
        self.outcome = outcome

def create_root_node(intro_text):
    intro_content = intro_text.replace("Intro: ", "").strip()
    story_node = StoryNode(intro=intro_content)
    return Node(story_node)

def create_story_node(action_text, outcome_text):
    action_content = action_text.replace("Action: ", "").strip()
    outcome_content = outcome_text.replace("Outcome: ", "").strip()
    story_node = StoryNode(action=action_content, outcome=outcome_content)
    return Node(story_node)

def count_tokens(text):
    encoder = tiktoken.encoding_for_model(MODEL)
    tokens = encoder.encode(text)
    return len(tokens)
    
def estimate_story_cost():
    input_tokens = 0
    output_tokens = 0

    # System message tokens are only added once per prompt, not per token
    system_message_tokens = count_tokens(GPT4_SYSTEM_INSTRUCTIONS)

    # Calculate tokens for the introductory prompt
    intro_prompt_tokens = count_tokens(generate_story_prompt(depth=1, max_depth=STORY_DEPTH, children_nodes=0, story_context=[], starting_prompt=STARTING_PROMPT))
    input_tokens += intro_prompt_tokens  # Add once for the intro prompt
    
    # Initialize a dummy root node for context simulation
    root_node = create_root_node("Intro: Placeholder intro for token counting.")
    context = [root_node]  # Start with root node in context

    # Generate tokens for subsequent prompts, simulating the cumulative story context
    for depth in range(2, STORY_DEPTH):
        # Simulate story context up to the current depth
        simulated_context = generate_story_context(context)
        cumulative_prompt_tokens = count_tokens(generate_story_prompt(depth=depth, max_depth=STORY_DEPTH, children_nodes=NUM_DECISIONS, story_context=context))
        num_nodes_at_depth = NUM_DECISIONS ** (depth - 1)  # Nodes that will generate prompts at this depth
        
        # Add input tokens for each node at the current depth, include system message tokens once per node
        input_tokens += (cumulative_prompt_tokens + system_message_tokens) * num_nodes_at_depth
        
        # Update context with a dummy node to simulate depth increase for token counting
        if depth < STORY_DEPTH - 1:  # Avoid adding beyond the last depth since no prompts are generated for the last layer
            dummy_node = create_story_node("Action: Placeholder action for token counting.", "Outcome: Placeholder outcome for token counting.")
            context.append(dummy_node)  # Simulate the addition of nodes for depth

    # Calculate output tokens for all nodes except the last layer (which doesn't generate new prompts)
    # Each node produces output, shared equally among NUM_DECISIONS children
    total_nodes = sum(NUM_DECISIONS ** i for i in range(STORY_DEPTH))  # Total nodes including the last layer
    output_tokens = total_nodes * (MAX_TOKEN_LIMIT // NUM_DECISIONS)

    # Calculate the total cost
    cost = (input_tokens * MODEL_INPUT_COST) + (output_tokens * MODEL_OUTPUT_COST)
    return cost

def generate_story_context(story_till_now):
    context = ""
    if len(story_till_now) == 0:
        return context
    for node in story_till_now:
        if node.content.intro:
            context += "Intro: " + node.content.intro + "\n"
        if node.content.action:
            context += "Action: " + node.content.action + "\n"
        if node.content.outcome:
            context += "Outcome: " + node.content.outcome + "\n"
    
    return context.strip()

def generate_story_prompt(depth, max_depth, children_nodes, story_context, starting_prompt=None):
    distance_from_end = max_depth - depth
    story_context = generate_story_context(story_context)
    
    if starting_prompt:
        return (f"You are to write a story based on the following prompt from the user: " +
                f"'{starting_prompt}'\n\n" +
                "Please write an introductory segment for the story. " +
                "Your introduction should set the scene for a rich and engaging narrative. " +
                "Remember to format it as specified: 'Intro: [Your story introduction here.]'")
    
    return (f"Given the current story context:\n{story_context}\n" +
              f"With {distance_from_end} decisions remaining until the story ends, " +
              "generate the next segment of the story. " +
              "Remember to format your response with an action and its outcome as specified: " +
              "'Action: [The player's action.]' 'Outcome: [The outcome of that action.]'" +
              f"There should be exactly {children_nodes} action-outcome pairs.")

def generate_story_text(prompt):
    messages = [
        {"role": "system", "content": GPT4_SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=MAX_TOKEN_LIMIT,
        temperature=TEMPERATURE
    )
    
    story_text = response.choices[0].message.content.strip()
    return story_text

def parse_action_outcome_pairs(action_outcome_text):
    pairs = []
    action_outcome_pairs = action_outcome_text.split("Action: ")[1:]
    
    for pair in action_outcome_pairs:
        try:
            parts = pair.split("Outcome: ")
            if len(parts) < 2:
                raise ValueError("Incorrectly formatted action-outcome pair.")
            
            action = parts[0].strip()
            outcome = "Outcome: ".join(parts[1:]).strip()
            child_node = create_story_node("Action: " + action, "Outcome: " + outcome)
            pairs.append(child_node)
        except ValueError as e:
            print(f"Error processing pair: '{pair}'. Error: {e}")
            # Return a sentinel value indicating failure
            return None
    
    return pairs

def generate_storyline(estimated_story_cost):
    stack = []
    context = []
    depth = 0

    current_story_cost = 0 # used for tracking the cost of generating the story
    total_nodes = sum(NUM_DECISIONS ** i for i in range(STORY_DEPTH)) # used for estimating the cost of generating the story
    nodes_complete = 0 # used for tracking the number of nodes generated
    notified_of_cost_exceedance = False

    print(f"Total nodes: {total_nodes}")
    
    prompt = generate_story_prompt(depth=1, max_depth=STORY_DEPTH, children_nodes=0, story_context=[], starting_prompt=STARTING_PROMPT)
    intro_text = generate_story_text(prompt)
    root = create_root_node(intro_text)
    stack.append((root, 1))

    # calculate the cost of generating the intro
    nodes_complete += 1
    current_story_cost += count_tokens(prompt) * MODEL_INPUT_COST
    current_story_cost += count_tokens(intro_text) * MODEL_OUTPUT_COST
    print(f"Current story cost: ${current_story_cost}")

    print(f"Nodes complete: {nodes_complete}, Nodes left: {total_nodes - nodes_complete}")

    while stack:
        node, node_depth = stack.pop()
        depth = node_depth

        context = context[:depth-1]
        context.append(node)

        retries = 5
        while retries > 0:
            prompt = generate_story_prompt(depth=depth, max_depth=STORY_DEPTH, children_nodes=NUM_DECISIONS, story_context=context)
            action_outcome_text = generate_story_text(prompt)
            node.children = parse_action_outcome_pairs(action_outcome_text)
            
            # calculate the cost of generating the action-outcome pairs
            current_story_cost += count_tokens(prompt) * MODEL_INPUT_COST
            current_story_cost += count_tokens(action_outcome_text) * MODEL_OUTPUT_COST

            if not node.children:  # Parsing failure
                print(f"Erroneous response, trying again: {nodes_complete}, Nodes left: {total_nodes - nodes_complete}")
                print(f"Current story cost: ${current_story_cost}")
                retries -= 1  # Decrement retries after a failed attempt
            else:  # Successful parsing, exit the retry loop
                break

            if retries == 0:  # Retry limit reached, exit with a failure message
                print("Erroneous response returned 5x in a row. Terminating story generation.")
                return None

        nodes_complete += NUM_DECISIONS

        print(f"Nodes complete: {nodes_complete}, Nodes left: {total_nodes - nodes_complete}")

        if current_story_cost > estimated_story_cost and not notified_of_cost_exceedance:
            updated_cost_estimate = (current_story_cost / nodes_complete) * (total_nodes - nodes_complete)
            print(f"The cost of generating the story has now exceeded the estimated cost. Current cost: ${current_story_cost}. "
                f"There are still {total_nodes - nodes_complete} story nodes remaining to generate. "
                f"The estimated cost to complete the story is: ${updated_cost_estimate}")
            user_input = input("Would you like to continue? If you stop now, the story up until this point will still be saved, but it will be incomplete (Yes/No): ")
            notified_of_cost_exceedance = True
            if user_input.lower() != 'yes':
                print("Story generation terminated.")
                break
        else:
            print(f"Current story cost: ${current_story_cost}")

        if depth + 1 < STORY_DEPTH:
            for child in reversed(node.children):
                stack.append((child, depth + 1))

    return root

def linearize_tree(node):
    if node is None:
        return [None]
    linearized = [(f'intro: "{node.content.intro}"' if node.content.intro else '') +
                  (f'action: "{node.content.action}"' if node.content.action else '') +
                  (f' outcome: "{node.content.outcome}"' if node.content.outcome else '')]
    for child in node.children:
        linearized.extend(linearize_tree(child))
    linearized.append(None)
    return linearized

def serialize_tree(node):
    linearized_tree = linearize_tree(node)
    serialized = ""
    node_count = 0 
    for value in linearized_tree:
        if value is not None:
            serialized += f"[{node_count}]: {value}\n"
            node_count += 1
        else:
            serialized += "[X]\n"
    return serialized

if __name__ == "__main__":
    openai.api_key = OPENAI_API_KEY

    estimated_story_cost = estimate_story_cost()

    print(f"Estimated cost for generating the story: ${estimated_story_cost}")
    while True:
        confirmation = input("Would you like to proceed with generating the story? (Enter 'Confirm' to begin, or 'X' to cancel...): ")
        if confirmation == "Confirm":
            break
        elif confirmation == "X":
            print("Story generation cancelled.")
            break
        else:
            print("Invalid input. Please enter 'Confirm' to begin or 'X' to cancel.")

    root = generate_storyline(estimated_story_cost)
    serialized_story = serialize_tree(root)
    with open("story.txt", "w") as file:
        file.write(serialized_story)
    
    print("Storyline generation complete.")