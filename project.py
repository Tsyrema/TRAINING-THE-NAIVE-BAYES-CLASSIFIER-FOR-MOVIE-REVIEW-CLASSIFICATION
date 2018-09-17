# Naive Bayes Classifier

import math
import copy


inputData = []
classes = []
dict = {}
dict_likelihood = {}
docs = {}

def getNumberOfDocs (textFile):   # count how many documents - lines in text file
    count = 0
    with open(textFile, 'r') as read_input:
        for line in read_input:
            count += 1
    read_input.close()
    return count

def processFile (inputText):    #list of lists that are lines from file
    out_list = []
    read_input = open(inputText, 'r')
    for line in read_input:
        temp = line.strip().split()
        out_list.append(temp)
    read_input.close()
    return out_list


def createVocab (inList):   # vocabulary - list of unique words
    vocab = []
    new_list = []
    for line in inList:
        for words in line:
            if words != line[len(line)-1]:
                new_list.append(words)
    for w in new_list:
        if w in vocab:
            continue
        else:
            vocab.append(w)
    return vocab

def getClasses (inList):    #list of classes
    out_list = []
    for line in inList:
        if line[len(line)-1] in out_list:
            continue
        else:
            out_list.append(line[len(line)-1])
    return out_list

def numberOfDocsGivenClass(cl, inList):    # Number of documents, given a class
    count = 0
    for line in inList:
        if line[len(line)-1] != cl:
            continue
        else:
            count += 1
    return count


def main():
    number_of_docs = getNumberOfDocs('data.txt')
    print 'Total number of documents/lines in input', number_of_docs

    inputData = processFile('data.txt')

    vocab_unique = createVocab(inputData)
    print 'Vocabulary of unique words: ' , vocab_unique

    classes = getClasses(inputData)
    print 'Classes:' , classes

    # dictionary with keys = classes, values = vocabulary of uniques words
    for c in classes:
        temp_dict = {}
        for word in vocab_unique:
            temp_dict[word] = 0
        dict[c] = temp_dict


    # keys = classes, values = count of words in document given class
    for key in dict:
        for line in inputData:
            if line[len(line)-1] != key:
                continue
            else:
                for word in line:
                    if word not in dict[key]:
                        continue
                    (dict[key])[word] +=1

    print 'dict :', dict

    # count of words given class
    words = {}
    for cl in classes:
        count = 0
        for k in dict[cl]:
            count += dict[cl][k]
        words[cl] = count
    print 'Count of words, given class: ', words

    # Number of documents, given a class
    for cl in classes:
        d = numberOfDocsGivenClass(cl, inputData)
        docs[cl] = d
        print 'Number of documents, given class ', cl, ' : ', d


    #create a dict of word likelihoods with add-1 smoothing with respect to a class
    dict_likelihood = dict.copy()

    for cl in classes:
        for k in dict_likelihood[cl]:
            dict_likelihood[cl][k] = (dict_likelihood[cl][k] + 1)/float(words[cl] + len(vocab_unique))

    print 'Word likelihoods with add-1 smoothing with respect to class: ', dict_likelihood

    # prior probabilities
    priors = {}
    for cl in classes:
        if cl not in priors:
            priors[cl] = docs[cl] / float(number_of_docs)
    print 'Prior probabilities: ', priors

    # Test you classifier on the new document {fast, couple, shoot, fly}
    test_data = ['fast', 'couple', 'shoot', 'fly']
    testing = {}

    #add classes as keys to testing dict, values = prior probabilities
    for cl in classes:
        if cl not in testing:
            testing[cl] = priors[cl]

    for cl in classes:
        for w in test_data:
            for key in dict_likelihood[cl]:
                if key != w:
                    continue
                elif key == w:
                    testing[cl] *= dict_likelihood[cl][w]
    print 'Probabilities of test data: ', testing

    #Compute the most likely class
    result = max(testing, key = testing.get)
    print 'The most likely class for the test document: ', result


if __name__ == "__main__": main()
