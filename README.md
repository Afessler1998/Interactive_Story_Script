# Interactive Story Script

This is a python script that leverages GPT 3.5 to write interactive storylines with different endings depending on a player's decisions.
At the top of the script there is a set of parameters you can customize to get varying dimensions for your story. The script will estimate the cost of generating the story based off of the assumption that every response from GPT 3.5 will contain the max token limit, in practice this doesn't actually happen, but it's useful for determining the upper bound of cost.

The storyline generation function fills out the tree depth-first by prompting GPT 3.5 with the direct, linear path through the tree up to the current node, and notifies it how far it is from the end of the story, so it can tie it up with a satisfying conclusion. Upon completion, the tree is serialized into a custom file format I came up with that can be loaded into a text based adventure game you can find on my GitHub.