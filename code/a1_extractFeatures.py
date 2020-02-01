import numpy as np
import argparse
import json
import math
import pandas as pd
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

Bristol_Gilhooly_logie  = "~/Documents/grad/sem4/csc2511/assignment1/A1/code/BristolNorms+GilhoolyLogie.csv"
Warrineretal = "~/Documents/grad/sem4/csc2511/assignment1/A1/code/Ratings_Warriner_et_al.csv"

def normsBGL(task,token, bin):
    BGL =pd.read_csv(Bristol_Gilhooly_logie)
    mean=[]

    for i in token:
        if i in BGL["WORD"].values:
            mean.append(BGL.at[BGL.loc[BGL["WORD"] == i].index.values[0],task])
        else:
            mean.append(0)
    return np.average(mean) if bin == 1 else np.std(mean)

def normsWarriner(task,token, bin):
    Warriner =pd.read_csv(Warrineretal)
    mean=[]

    for i in token:
        if i in Warriner["Word"].values:
            mean.append(Warriner.at[Warriner.loc[Warriner["Word"] == i].index.values[0],task])
        else:
            mean.append(0)
    return np.average(mean) if bin == 1 else np.std(mean)

def retword(path,name):

    path_csv = pd.read_csv(path)
    print(len(path_csv[name]))







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
    feats[6] =  0 #FUTURE TENSE
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
    #retword(Bristol_Gilhooly_logie,"WORD")
    '''
    feats[17] = normsBGL("AoA (100-700)",token_lower,1)
    feats[18] = normsBGL("IMG",token_lower,1)
    feats[19] = normsBGL("FAM",token_lower,1)
    feats[20] = normsBGL("AoA (100-700)",token_lower,0)
    feats[21] = normsBGL("IMG", token_lower, 0)
    feats[22] = normsBGL("FAM", token_lower, 0)
    feats[23] = normsWarriner("V.Mean.Sum" ,token_lower, 0)
    feats[24] = normsWarriner("A.Mean.Sum", token_lower, 0)
    feats[25] = normsWarriner("D.Mean.Sum", token_lower, 0)
    feats[26] = normsWarriner("V.Mean.Sum", token_lower, 1)
    feats[27] = normsWarriner("A.Mean.Sum", token_lower, 1)
    feats[28] = normsWarriner("D.Mean.Sum", token_lower, 1)
    '''


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
    print('TODO')


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    # TODO: Use extract1 to find the first 29 features for each 
    # data point. Add these to feats.
    for i,line in enumerate(data[:]):
        line = json.loads(line)
        #print(line['body'])
        comment = line['body']
        if len(comment) > 0:
            feats[i][0:28] =  extract1(comment)
        #feats[i] = extract2(feats,line['cat'],line['id'])
        else:
            print("pangaaaa \n \n \n \n \n \n \n")

    # TODO: Use extract2 to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).
    print('TODO')

    #np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

