#------------------------------------------------------------------

#
#   Bayes Optimal Classifier
#
#   In this quiz we will compute the optimal label for a second missing word in a row
#   based on the possible words that could be in the first blank
#
#   Finish the procedurce, LaterWords(), below
#
#   You may want to import your code from the previous programming exercise!
#

sample_memo = '''
Milt, we're gonna need to go ahead and move you downstairs into storage B. We have some new people coming in, and we need all the space we can get. So if you could just go ahead and pack up your stuff and move it down there, that would be terrific, OK?
Oh, and remember: next Friday... is Hawaiian shirt day. So, you know, if you want to, go ahead and wear a Hawaiian shirt and jeans.
Oh, oh, and I almost forgot. Ahh, I'm also gonna need you to go ahead and come in on Sunday, too...
Hello Peter, whats happening? Ummm, I'm gonna need you to go ahead and come in tomorrow. So if you could be here around 9 that would be great, mmmk... oh oh! and I almost forgot ahh, I'm also gonna need you to go ahead and come in on Sunday too, kay. We ahh lost some people this week and ah, we sorta need to play catch up.
'''

corrupted_memo = '''
Yeah, I'm gonna --- you to go ahead --- --- complain about this. Oh, and if you could --- --- and sit at the kids' table, that'd be --- 
'''

data_list = sample_memo.strip().split()

words_to_guess = ["ahead","could"]

def NextWordProbability(sampletext,word):
    
    result = {}
    split = sampletext.split();
    for i, s in enumerate(split):
        if s == word and i < len(split):
            next_word = split[i+1]
            if next_word in result:
                count = result[next_word]+ 1
            else:
                count = 1
            result[next_word] = count
    
    sum_all = 0
    for v in result.values():
        sum_all = sum_all + v
    
    print ""
    print word
    print result
    
    for k in result:
        result[k] = (float)(result[k])/sum_all
    
    return result

def LaterWords(sample,word,distance):
    '''@param sample: a sample of text to draw from
    @param word: a word occuring before a corrupted sequence
    @param distance: how many words later to estimate (i.e. 1 for the next word, 2 for the word after that)
    @returns: a single word which is the most likely possibility
    '''
    
    # TODO: Given a word, collect the relative probabilities of possible following words
    # from @sample. You may want to import your code from the maximum likelihood exercise.
    relative_probabilities = NextWordProbability(sample, word)
    
    # TODO: Repeat the above process--for each distance beyond 1, evaluate the words that
    # might come after each word, and combine them weighting by relative probability
    # into an estimate of what might appear next.
    for i in range(2,distance + 1):
        full_dict = {}
        for wd in relative_probabilities:
            temp_dict = NextWordProbability(sample,wd)
            for wd2 in temp_dict:
                full_dict[wd2] =  temp_dict[wd2] * relative_probabilities[wd]
        relative_probabilities = full_dict

    srt_words = sorted(relative_probabilities.items(), key = lambda x:x[1],reverse = True)
    return srt_words[0][0]

print LaterWords(sample_memo,"ahead",2)