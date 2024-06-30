import tiktoken


def count_tokens(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# ------ TESTS ------
messages_history = [
    {"role": "user", "content": "Hi there! I'm looking for a new laptop."},
    {"role": "assistant", "content": "Sure! What are your requirements?"},
    {"role": "user", "content": "I need something powerful for gaming and also lightweight for travel."},
    {"role": "assistant", "content": "I recommend considering models like the Asus ROG Zephyrus or the Razer Blade."},
    {"role": "user", "content": "Great! Do they come with a good warranty?"},
    {"role": "assistant", "content": "Yes, both options usually offer solid warranty coverage. Do you have any other preferences?"},
    {"role": "user", "content": "I prefer a sleek design and long battery life."},
    {"role": "assistant", "content": "Both laptops have stylish designs and decent battery performance. Would you like more details on their specs?"},
    {"role": "user", "content": "Yes, please provide me with the specifications."},
    {"role": "assistant", "content": "Sure, let me fetch that information for you."},
    {"role": "assistant", "content": "Here are the detailed specs for both laptops..."},
    {"role": "assistant", "content": "Based on your needs, I would recommend the Razer Blade for its balance between performance and portability."},
    {"role": "user", "content": "Thank you! I'll look into it."},
    {"role": "assistant", "content": "You're welcome! Let me know if you have any more questions."},
]