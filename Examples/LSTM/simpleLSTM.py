import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

"""
Takes a sequence and converts it to a sequence of integers
@param toID is a dictionary
@param sequence is the sequence to convert
@return a tensor of the sequence of ids
"""


def prepareSequence(sequence, toID):
    ids = [toID[w] for w in sequence]
    return torch.tensor(ids, dtype=torch.long)


# list of tuples, each tuple has a list of words and its tags
trainingData = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("dog read that book".split(), ["NN", "V", "DET", "NN"])
]

# build a dictionary with all the sentences
wordToIdDictionary = {}
for sentence, tags in trainingData:
    for word in sentence:
        if word not in wordToIdDictionary:
            # a new id for each new word
            wordToIdDictionary[word] = len(wordToIdDictionary)

print("Word to id dictionary")
print(wordToIdDictionary)
# 3 classes to use
tagsIds = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
INPUT_DIMENSION = 6
HIDDEN_DIMENSION = 6


class LSTM_simple(nn.Module):
    """
    @param inputEmbeddingDimension, the dimension of the input vector, used by word2vec
    @param hiddenDimension, number of hidden units
    @param vocabularySize, number of different words in the training set of sentences
    @param numberClasses, number of different classes to
    """

    def __init__(self, inputEmbeddingDimension, hiddenDimension, vocabularySize, numberClasses):
        super(LSTM_simple, self).__init__()
        # Number of hidden units
        self.hiddenDimension = hiddenDimension
        # Create word embedding using Word2vec or skipgram
        # we can specify the desired dimensions of the input
        self.wordEmbeddings = nn.Embedding(vocabularySize, inputEmbeddingDimension)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hiddenDimension.
        # creates an LSTM cell
        # We can feed the whole sequence at once to the LSTM
        self.lstm = nn.LSTM(inputEmbeddingDimension, hiddenDimension)

        # The linear layer that maps from hidden state space to the output, with the number of tags
        self.output = nn.Linear(hiddenDimension, numberClasses)

    """
    Forward received a whole sentence first
    @param sentence, takes the whole sentence of words
    @return classScores, higher the better
    """

    def forward(self, sentence):
        # we can feed the whole sequence at once
        inputSentenceEmbedding = self.wordEmbeddings(sentence)

        # deflattens the whole sentence
        # Pytorch’s LSTM expects all of its inputs to be 3D tensors. The semantics of the axes of these tensors is important.
        # The first axis is the sequence itself, the second indexes instances in the mini-batch (number of samples), and the third indexes elements
        # of the input (INPUT_DIMENSION)
        deflattenedInput = inputSentenceEmbedding.view(len(sentence), 1, -1)

        lstmOutput, _ = self.lstm(deflattenedInput)
        classNetOutput = self.output(lstmOutput.view(len(sentence), -1))
        classScores = F.log_softmax(classNetOutput, dim=1)
        return classScores

model = LSTM_simple(INPUT_DIMENSION, HIDDEN_DIMENSION, len(wordToIdDictionary), len(tagsIds))
# negative log likelihood loss
lossFunction = nn.NLLLoss()
# stochastic gradient descent with learning rate 0.1
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepareSequence(trainingData[0][0], wordToIdDictionary)
    classScores = model(inputs)
    print("Scores with no training")
    print(classScores)

for epoch in range(200):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in trainingData:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentenceEmbedding = prepareSequence(sentence, wordToIdDictionary)
        targets = prepareSequence(tags, tagsIds)

        # Step 3. Run our forward pass.
        classScores = model(sentenceEmbedding)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = lossFunction(classScores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = prepareSequence(trainingData[0][0], wordToIdDictionary)
    classScores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print("Scores after training")
    print(classScores)
    print("For sentence ")
    print(trainingData[0][0])