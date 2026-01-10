import pandas as pd
import re
import numpy as np
import json

with open("./datasets/ssa_holder_as_dict", 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame.from_dict(data, orient='index')
df = df.reset_index().rename(columns={'index': 'id'})

list_of_texts = []
list_of_positions = []
tags_tokens = []
list_of_tags = []
notfound = []
                   
def get_splitted_words_by_location(text: str, location):
    if not location or len(location) != 2:
        return []
    start = location[0]
    end = location[1]
    return text[start:end].split()
    
def map_to_tags(row):
    text = row["text"]
    annotations = row["annotations"]
    
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    
    tags = np.zeros(len(preprocessed), dtype='int')
    
    for annotation in annotations:
        polarity = annotation.get("polarity", "")
        
        for component in ["holder", "aspect", "sentiment"]:
            if component not in annotation:
                continue
                
            component_data = annotation[component]
            
            if component_data is None or component_data.get("term") == "null" or not component_data.get("term"):
                continue
            
            location = component_data.get("location")
            if not location or len(location) != 2:
                continue
            
            splited_words = get_splitted_words_by_location(text, location)
            if not splited_words:
                continue
            
            last_found_index = 0
            for idx, word in enumerate(splited_words):
                found = False
                
                cleaned_word = re.sub(r'[^\w\s]', '', word.lower())
                
                for token_idx in range(last_found_index, len(preprocessed)):
                    cleaned_token = re.sub(r'[^\w\s]', '', preprocessed[token_idx].lower())
                    
                    if cleaned_token == cleaned_word:
                    
                        if component == "sentiment":
                            tag_type = polarity
                        else:
                            tag_type = component
                        
                        if tag_type in termToLabel:
                            if idx == 0:
                                tags[token_idx] = label2id[termToLabel[tag_type][0]]
                            else:
                                tags[token_idx] = label2id[termToLabel[tag_type][1]]
                        
                        last_found_index = token_idx + 1
                        found = True
                        break
                
                if not found:
                    notfound.append(word)
    
    tags_tokens.append({
        "id": row["id"],
        "tags": tags,
        "tokens": preprocessed,
        "text": text
    })
    list_of_tags.append(tags)

id2label = {
    0: "O",
    1: "B-ASPECT",
    2: "I-ASPECT",
    3: "B-HOLDER",
    4: "I-HOLDER",
    5: "B-EXPRESSION-POS",
    6: "I-EXPRESSION-POS",
    7: "B-EXPRESSION-NEG",
    8: "I-EXPRESSION-NEG",
    9: "B-EXPRESSION-NEU",
    10: "I-EXPRESSION-NEU"
}

label2id = {v: k for k, v in id2label.items()}

termToLabel = {
    "holder": ["B-HOLDER", "I-HOLDER"],
    "aspect": ["B-ASPECT", "I-ASPECT"],
    "POS": ["B-EXPRESSION-POS", "I-EXPRESSION-POS"],
    "NEG": ["B-EXPRESSION-NEG", "I-EXPRESSION-NEG"],
    "NEU": ["B-EXPRESSION-NEU", "I-EXPRESSION-NEU"]
}

for _, row in df.iterrows():
    map_to_tags(row)

tags_tokens_df = pd.DataFrame([
    {
        "id": t["id"],
        "tags": t["tags"].tolist(),
        "tokens": t["tokens"],
        "text": t["text"]
    }
    for t in tags_tokens
])

tags_tokens_df.to_json("dataset.json", orient="records", force_ascii=False, indent=2)


