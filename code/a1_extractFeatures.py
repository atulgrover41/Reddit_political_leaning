import numpy as np
import argparse
import json
import pandas as pd
import csv
# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}
punctations = {
    '$', '.', '#', ':', ',', '(', ')', '"', '``', '“', '”', '’', "''"
}

Bristol_Gilhooly_logie  = "/Users/jarvis/Documents/grad/sem4/csc2511/assignment1/A1/code/BristolNorms+GilhoolyLogie.csv"
Warrineretal = "/Users/jarvis/Documents/grad/sem4/csc2511/assignment1/A1/code/Ratings_Warriner_et_al.csv"
left_path  = "/Users/jarvis/Documents/grad/sem4/csc2511/assignment1/A1/feats/Left_IDs.txt"
right_path = "/Users/jarvis/Documents/grad/sem4/csc2511/assignment1/A1/feats/Right_IDs.txt"
alt_path = "/Users/jarvis/Documents/grad/sem4/csc2511/assignment1/A1/feats/Alt_IDs.txt"
center_path = "/Users/jarvis/Documents/grad/sem4/csc2511/assignment1/A1/feats/Center_IDs.txt"

left_data_path  = "/Users/jarvis/Documents/grad/sem4/csc2511/assignment1/A1/feats/Left_feats.dat.npy"
right_data_path = "/Users/jarvis/Documents/grad/sem4/csc2511/assignment1/A1/feats/Right_feats.dat.npy"
alt_data_path = "/Users/jarvis/Documents/grad/sem4/csc2511/assignment1/A1/feats/Alt_feats.dat.npy"
center_data_path = "/Users/jarvis/Documents/grad/sem4/csc2511/assignment1/A1/feats/Center_feats.dat.npy"



#Warriner = csv.DictReader(open(Warrineretal))
#BGL = csv.DictReader(open(Bristol_Gilhooly_logie))
left_id = open(left_path,"r")
left_id =[line.strip('\n') for line in left_id]
left_id = dict(zip(left_id,range(0,len(left_id))))
right_id = open(right_path,"r")
right_id =[line.strip('\n') for line in right_id]
right_id = dict(zip(right_id,range(0,len(right_id))))
center_id = open(center_path,"r")
center_id =[line.strip('\n') for line in center_id]
center_id = dict(zip(center_id,range(0,len(center_id))))
alt_id = open(alt_path,"r")
alt_id =[line.strip('\n') for line in alt_id]
alt_id = dict(zip(alt_id,range(0,len(alt_id))))

left_data = np.load(left_data_path)
right_data = np.load(right_data_path)
center_data = np.load(center_data_path)
alt_data = np.load(alt_data_path)

def readCSV(filename):
    list_read=[]
    words = dict()
    with open(filename,'r') as fp:
        list_read=fp.readlines()
    header_list= list_read[0].split(',')
    list_read.pop(0)

    list_read =[line.split(',') for line in list_read]
    for i,j in enumerate(list_read):
        words[j[1]] = i
    return list_read,words

Warriner,Warriner_words = readCSV(Warrineretal)
BGL,BGL_words = readCSV(Bristol_Gilhooly_logie)
def normsBGL(token):

    AoA=[]
    FAM=[]
    IMG=[]

    for i in token:
        if i in BGL_words and i !="":
            v, a, d = BGL[BGL_words[i]][3], BGL[BGL_words[i]][4], BGL[BGL_words[i]][5]
            AoA.append(float(v))
            FAM.append(float(a))
            IMG.append(float(d))
    #print(AoA)
    #remain = len(token) - len(AoA)
    #print(AoA)
    #AoA = np.pad(AoA,(0,remain),mode = 'constant')
    #FAM = np.pad(FAM,(0,remain),mode = 'constant')
    #IMG = np.pad(IMG,(0,remain),mode = 'constant')
    if len(AoA) == 0:
        AoA =[0]
    if len(FAM) == 0:
        FAM =[0]
    if len(IMG) == 0:
        IMG =[0]

    return [np.average(AoA),np.average(IMG),np.average(FAM), np.std(AoA),np.std(IMG),np.std(FAM)]

def normsWarriner(token):

    VMS=[]
    AMS=[]
    DMS=[]
    for i in token:
        if i in Warriner_words:
            v,a,d=Warriner[Warriner_words[i]][2], Warriner[Warriner_words[i]][5], Warriner[Warriner_words[i]][8]
            VMS.append(float(v))
            AMS.append(float(a))
            DMS.append(float(d))
    #print(VMS)
    #remain = len(token) - len(VMS)
    #VMS = np.pad(VMS,(0,remain),mode ='constant')
    #AMS = np.pad(AMS,(0,remain),mode ='constant')
    #DMS = np.pad(DMS,(0,remain),mode ='constant')
    if len(AMS) == 0:
        AMS = [0]
    if len(VMS) == 0:
        VMS = [0]
    if len(DMS) == 0:
        DMS = [0]

    return [np.average(VMS),np.average(AMS),np.average(DMS), np.std(VMS),np.std(AMS),np.std(DMS)]


'''


def normsBGL(token):
    AoA=[]
    FAM=[]
    IMG=[]
    for i in token:
        if i in BGL["WORD"].values:
            x=(BGL.loc[BGL["WORD"] == i].index.values[0])
            AoA.append(BGL.iloc[x,"AoA (100-700)"])
            FAM.append(BGL.iloc[x,"FAM"])
            IMG.append(BGL.iloc[x,"IMG"])
        else:
            AoA.append(0)
            FAM.append(0)
            IMG.append(0)
    return [np.average(AoA),np.average(IMG),np.average(FAM), np.std(AoA),np.std(IMG),np.std(FAM)]

def normsWarriner(token):
    VMS=[]
    AMS=[]
    DMS=[]

    for i in token:
        if i in Warriner["Word"].values:
            x=(Warriner.loc[Warriner["Word"] == i].index.values[0])
            VMS.append(Warriner.iloc[x,"V.Mean.Sum"])
            AMS.append(Warriner.iloc[x,"A.Mean.Sum"])
            DMS.append(Warriner.iloc[x,"D.Mean.Sum"])
        else:
            VMS.append(0)
            AMS.append(0)
            DMS.append(0)
    return [np.average(VMS),np.average(AMS),np.average(DMS), np.std(VMS),np.std(AMS),np.std(DMS)]

'''

def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''    
    # TODO: Extract features that rely on capitalization.
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    # TODO: Extract features that do not rely on capitalization.
    #print(comment)
    feats = np.zeros((29,))
    words = comment.split()
    token = []
    tags = []
    for word in words:
        split_word = word.split("/")
        tags.append(split_word[1])
        token.append(split_word[0])
    feats[0] = len([letter for letter in token if (letter.isupper() and len(letter) >= 3)])
    token_lower =[letter.lower() for letter in token]
    feats[1] = len([letter for letter in token_lower if letter in FIRST_PERSON_PRONOUNS])
    feats[2] = len([letter for letter in token_lower if letter in SECOND_PERSON_PRONOUNS])
    feats[3] = len([letter for letter in token_lower if letter in THIRD_PERSON_PRONOUNS])
    feats[4] = len([tag for tag in tags if tag == "CC"])
    feats[5] = len([tag for tag in tags if tag in ["VBD" , "VBN"]])
    feats[6] =  0 #FUTURE TENSE,complete it
    feats[7] = len([letter for letter in token_lower if letter == ","])
    feats[8] = len([tag for tag in tags if tag in punctations])     #Check this, it may be incorrect
    feats[9] = len([tag for tag in tags if tag in ["NN" , "NNS"]])
    feats[10] = len([tag for tag in tags if tag in ["NNP", "NNPS"]])
    feats[11] = len([tag for tag in tags if tag in ["RB" , "RBR" , "RBS"]])
    feats[12] = len([tag for tag in tags if tag in ["WP", "WDT", "WP$", "WRB" ]])
    feats[13] = len([letter for letter in token_lower if letter in SLANG])
    sentences = comment.split("\n")
    #print(sentences)
    feats[14] = np.average([len(sentence.split()) for sentence in sentences if sentence != []])
    feats[15] = np.average([len(word) if tags[number] not in punctations else 0 for number,word in enumerate(token) ])
    feats[16] = len(sentences) - 1

    feats[17:23] = normsBGL(token_lower)
    feats[23:29] = normsWarriner(token_lower)
    #print(feats[17:29])
    return feats


def extract2(feats, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feats: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''    
    #We can also use a switcher but i think that will be a overkill for this
    if comment_class == 'Left':
        feats[29:173] = left_data[left_id[comment_id]]
        feats[173] = 0
    elif comment_class == 'Right':
        feats[29:173] = right_data[right_id[comment_id]]
        feats[173] = 1
    elif comment_class == 'Alt':
        feats[29:173] = alt_data[alt_id[comment_id]]
        feats[173] = 2
    elif comment_class == 'Center':
        feats[29:173] = center_data[center_id[comment_id]]
        feats[173] = 3

    return feats

def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))
    print(len(data))
    # TODO: Use extract1 to find the first 29 features for each
    # data point. Add these to feats.
    for i,line in enumerate(data):
        line = json.loads(line)
        #print(line['body'])
        comment = line['body']
        if len(comment) > 0:
            feats[i][0:29] =  extract1(comment)
        else:
            feats[i][0:29] = [0]*29

        feats[i] = extract2(feats[i],line['cat'],line['id'])


    # TODO: Use extract2 to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).
    print('TODO')

    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

