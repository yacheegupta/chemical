import json
import csv

f = open('synonyms.txt',"r")
lines = f.readlines()
f.close() 

length_list = len(lines)


##to extract names and synonames for every line 
index = 1
for index, line in enumerate(lines):
    string_input = eval(line)
    out_json = json.dumps(string_input)
    json_dict = json.loads(out_json)
    print(index,"   ",json_dict['name'][0])
    print(index,"   ",json_dict['synonyms'])
    

    for i in range(len(json_dict['synonyms'])):
        ab = json_dict['synonyms'][(len(json_dict['synonyms'])-i-1)]
        print(index,"   ", ab)
        
        csvData1 = [json_dict['name']]
        csvData2 = [json_dict['synonyms']]

    with open('name.csv', 'a') as csvFile1:
        writer1 = csv.writer(csvFile1)
        writer1.writerows(csvData1)
        csvFile1.close()

    with open('synonyms.csv', 'a') as csvFile2:  
        writer2 = csv.writer(csvFile2)
        writer2.writerows(csvData2)
        csvFile2.close()

    index+=1
    
            

    

