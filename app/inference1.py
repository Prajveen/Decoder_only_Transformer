from transformer1 import *
from transformers import GPT2Tokenizer, GPT2TokenizerFast

# Load the pre-trained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

d_model = 256
num_heads = 8
drop_prob = 0.1
batch_size = 128
ffn_hidden = 256
num_layers = 4
vocab_size = len(tokenizer)
num_epochs = 100


# Load the trained model
transformer = Transformer(d_model, ffn_hidden, num_heads, drop_prob, num_layers, vocab_size)
transformer.load_state_dict(torch.load("app/model/v1.pth", map_location=torch.device('cpu')))
transformer.eval()

def inference(transformer, tokenizer, starting_word, max_length, temperature=1.0):
    
    transformer.eval()
    # Convert starting and ending words to token IDs
    starting_token_ids = tokenizer.encode(starting_word)

    # Convert token IDs to tensor
    input_tensor = torch.tensor(starting_token_ids).unsqueeze(0).to(get_device())

    # Generate tokens until ending word is reached or maximum length is reached
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass through the model
            output = transformer(input_tensor)

            # Apply temperature scaling to the logits
            scaled_output = output / temperature

            # Get the last predicted token
            last_token = scaled_output.argmax(dim=-1)[:, -1]

            # Append the last token to the input tensor
            last_token = last_token.unsqueeze(0).to(input_tensor.device)  # Ensure last_token is on the same device as input_tensor
            input_tensor = torch.cat([input_tensor, last_token], dim=-1)

            # Check if the ending word is reached
            if (last_token == tokenizer.eos_token_id):
                break

    # Decode the generated tokens
    generated_text = tokenizer.decode(input_tensor.squeeze().tolist())

    return generated_text


# Now you can use the inference function with the loaded and evaluated model
max_length = 100
temperature = 0.8

def generate_text(start_word):
    generated_sequence = inference(transformer, tokenizer, start_word
                               , max_length, temperature)
    return generated_sequence

