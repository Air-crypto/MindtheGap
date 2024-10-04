import openai
import json
import time
from datasets import load_dataset, Dataset
from huggingface_hub import login

# Initialize API keys and clients
write_key = 'HF_KEY'
login(write_key)

client = openai.OpenAI(api_key='API_KEY')

# Load the English sentences dataset
english_sentences_dataset = load_dataset("sentence-transformers/wikipedia-en-sentences", streaming=True, split="train")

def get_french_translation(english_sentence):
    try:
        translation_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. You translate English sentences into French."},
                {"role": "user", "content": f"Translate the following sentence to French:\n{english_sentence}"}
            ]
        )
        return translation_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Create the dataset
dataset_list = []

for i, entry in enumerate(english_sentences_dataset):
    if i >= 10000:
        break
    
    if i < 2000:
        continue

    english_sentence = entry['sentence']
    french_translation = get_french_translation(english_sentence)

    if french_translation:
        dataset_list.append({
            "english": english_sentence,
            "french": french_translation,
            "label": 1  # Good translation
        })
        print(f"Added English-French pair {i + 1}")

    time.sleep(1)  # To avoid rate limiting

# Save the dataset to a JSON file
with open('english_french_translation_dataset_test.json', 'w') as f:
    json.dump(dataset_list, f, indent=4)

# Load the dataset and push to the Hugging Face Hub
dataset = Dataset.from_json("english_french_translation_dataset_test.json")
dataset.push_to_hub("aircrypto/English-French-Translations-Train-Large")

print("English-French translation dataset creation complete!")
