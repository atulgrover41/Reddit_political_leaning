import sys
import argparse
import os
import json
import re
import spacy
import html

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def preproc1(comment , steps=range(1, 8)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment

    if modComm == "[deleted]":
        return ""
    if 1 in steps:  # replace newlines with spaces
        modComm = re.sub(r"\n{1,}", " ", modComm)
    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)
    if 3 in steps:  # remove URLs
        modComm = re.sub(r"(http|www)\S+", "", modComm)
    if 4 in steps:  # remove duplicate spaces
        modComm = re.sub(r" {1,}", " ", modComm)
        #https://stackoverflow.com/questions/1546226/simple-way-to-remove-multiple-spaces-in-a-string
    if 5 in steps:
        doc = nlp(modComm)
        newline = ""
        #for words in doc.sents:
            # Token Text + slash required + tag required + one space
            #if word.is_stop == False:             # check if this makes a difference
        for words in doc.sents:
            for word in words:
                if word.lemma_.startswith("-") and not word.text.startswith("-"):
                    newline += word.text + "/" + word.tag_ + " "
                elif len(re.findall(r'\s{1,}',word.lemma_)) > 0:
                    newline+=word.text + "/" +word.tag_ + " "
                else:
                    newline += word.lemma_ + "/" + word.tag_ + " "
            newline += '\n'
        #modComm = newline


    if 7 in steps:
        '''
        newline = ""
        doc = nlp (modComm)
        for sent in doc.sents:
            newline+= sent.text
            newline+= "\n"
        '''
        modComm = re.sub(r'\n\s*\n',"\n",newline)
    #print(re.sub(r'\n\s*\n',"\n",modComm))


    # TODO: get Spacy document for modComm
    
    # TODO: use Spacy document for modComm to create a string.
    # Make sure to:
    #    * Insert "\n" between sentences.
    #    * Split tokens with spaces.
    #    * Write "/POS" after each token.
        
    return modComm

def main(args):
    allOutput = []
    print(preproc1("This is terrible     and going to \n \n \n \n Yes it is"))
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            ndata = json.load(open(fullFile))
            print(args.ID)
            start = args.ID[0] % len(ndata)
            if start + args.max <len(ndata):
                data = ndata[start:start+args.max]
            else:
                data = ndata[:args.max - len(ndata) + start] + ndata[start:]
            storedata =[]
            for i in range(len(data)):
                templine = {}
                line = json.loads(data[i])
                preprocess = preproc1(line['body'])
                templine['id'] = line['id']
                templine['body'] = preprocess
                templine['cat'] = file
                storedata.append(json.JSONEncoder().encode(templine))
            allOutput = allOutput + storedata
            # TODO: select appropriate args.max lines
            # TODO: read those lines with something like `j = json.loads(line)`
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            # TODO: append the result to 'allOutput'
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()
    #print(preproc1('well thats too bad!! i have never been to. There is some work going on here'))
#have to check step 3 of this

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max",type = int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')
    
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
    
    indir = os.path.join(args.a1_dir, 'data')
    main(args)
