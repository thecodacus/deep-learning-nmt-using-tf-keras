import json
import os


source='Dataset/Bible/English/bible.json'
target='Dataset/Bible/Bengali/bible.json'
datadir='Dataset/Bible/Bengali/data.json'

if(os.path.isfile('Dataset/temp1.txt')): os.remove('Dataset/temp1.txt')
if(os.path.isfile('Dataset/temp2.txt')): os.remove('Dataset/temp2.txt')
if(os.path.isfile(datadir)): os.remove(datadir)

counter = 0
source_dict = {}
target_dict = {}
with open(source, 'rb') as source_data:
    source_json = json.load(source_data)
    for chapter in source_json['Book']:
        for verse in chapter['Chapter']:
            for v in verse['Verse']:
                source_dict[v['Verseid']] = v['Verse']


with open(target, 'rb') as target_data:
    target_json=json.load(target_data)
    for chapter in target_json[ 'Book' ]:
        for verse in chapter[ 'Chapter' ]:
            for v in verse[ 'Verse' ]:
                line =u''+v['Verse'].encode('utf-8','ignore').decode('utf-8')
                target_dict[ v[ 'Verseid' ] ] = line


with open(datadir, 'wb') as dest:
            for verId in source_dict:
                if verId in target_dict:
                    srcTxt = source_dict[verId]
                    tgtTxt = target_dict[verId]
                    line = "{\"id\":\""+verId+"\", \"source\":\""+srcTxt+"\", \"target\":\""+tgtTxt+"\"}";
                    dest.write((line+'\n').encode())


if(os.path.isfile('Dataset/temp1.txt')): os.remove('Dataset/temp1.txt')
if(os.path.isfile('Dataset/temp2.txt')): os.remove('Dataset/temp2.txt')
