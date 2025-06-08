import json

with open('stelarquestions_generated.txt', 'r') as file:
    data = file.read().replace('\n', ' ').replace('\t', ' ')
    data = ' '.join(data.split())
    entries = data.split('@')
    
with open('stelarquestions.json', 'r') as file:
    stelarquestions = json.load(file)
    
    
# print(data)
for entry in entries:
    question, sparql = entry.split(' ! ')
    print(question, sparql)
    stelarquestions.append({
        'question': question,
        'sparql': sparql
    })
    
with open("stelarquestions_combined.json", "w") as outfile:
    json.dump(stelarquestions, outfile, indent=4, sort_keys=False)

