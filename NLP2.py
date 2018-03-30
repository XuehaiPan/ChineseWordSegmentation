import numpy as np
from time import time


class MultiClassPerceptron(object):
    def __init__(self, vecDim: int, classNum: int, learningRate: float):
        self.vecDim = vecDim
        self.classNum = classNum
        self.weights = np.zeros(shape = (classNum, vecDim))
        self.learningRate = learningRate

    def __str__(self):
        return str(self.weights)

    def __repr__(self):
        return repr(self.weights)

    def copy(self):
        copy = MultiClassPerceptron(vecDim = self.vecDim, classNum = self.classNum, learningRate = self.learningRate)
        copy.weights = self.weights.copy()
        return copy

    def predict(self, x: np.ndarray):
        f = np.dot(a = self.weights, b = x)
        return max(range(self.classNum), key = lambda c: f[c, 0])

    def update(self, x: np.ndarray, y: int):
        c = self.predict(x = x)
        if c != y:
            x = x[:, 0]
            self.weights[c, :] -= self.learningRate * x
            self.weights[y, :] += self.learningRate * x


class dictionary(set):
    def __init__(self, seq = ()):
        super(dictionary, self).__init__(seq)

    def make(self, file: str, encoding = 'UTF-8'):
        with open(file = file, mode = 'rt', encoding = encoding) as trainingData:
            for line in trainingData.readlines():
                words = line.strip().split()
                for word in words:
                    if word not in dictionary:
                        self.add(word)

    def load(self, file: str, encoding = 'UTF-8'):
        with open(file = file, mode = 'rt', encoding = encoding) as load:
            self.update(eval(load.readline()))

    def save(self, file: str, encoding = 'UTF-8'):
        with open(file = file, mode = 'w', encoding = encoding) as save:
            save.write('%r\n' % self)


class ChineseWordSegmenter(object):
    tagToLabel = {'S': 0, 'B': 1, 'M': 2, 'E': 3}
    labelToTag = {0: 'S', 1: 'B', 2: 'M', 3: 'E'}

    def __init__(self, learningRate: float):
        self.dict = dictionary()
        self.perceptron = MultiClassPerceptron(vecDim = 16, classNum = len(CWS.tagToLabel),
                                               learningRate = learningRate)

    def makeDict(self, file: str, encoding = 'UTF-8'):
        self.dict.make(file = file, encoding = encoding)

    def loadDict(self, file: str, encoding = 'UTF-8'):
        self.dict.load(file = file, encoding = encoding)

    def saveDict(self, file: str, encoding = 'UTF-8'):
        self.dict.save(file = file, encoding = encoding)

    @staticmethod
    def characterCount(file: str, encoding = 'UTF-8'):
        with open(file = file, mode = 'rt', encoding = encoding) as data:
            return sum(map(lambda line: len(CWS.getSentence(line = line)), data.readlines()))

    @staticmethod
    def getWords(line: str) -> [str]:
        return line.strip().split()

    @staticmethod
    def getSentence(line: str) -> str:
        return ''.join(CWS.getWords(line = line))

    @staticmethod
    def getTags(word: str) -> str:
        if len(word) == 0:
            return ''
        elif len(word) == 1:
            return 'S'
        else:
            return 'B' + 'M' * (len(word) - 2) + 'E'

    @staticmethod
    def getLabels(word: str) -> [int]:
        if len(word) == 0:
            return []
        elif len(word) == 1:
            return [0]
        else:
            return [1] + [2] * (len(word) - 2) + [3]

    def getVectors(self, sentence: str) -> [np.ndarray]:
        sentenceLen = len(sentence)
        vectors = [np.zeros(shape = (self.perceptron.vecDim, 1)) for i in range(sentenceLen)]
        for i, vec in enumerate(vectors):
            vec[0, 0] = 1

            if sentence[i] in self.dict:
                vec[1, 0] = 1

            if i > 0 and sentence[i - 1:i + 1] in self.dict:
                vec[2, 0] = 1
            if i < sentenceLen - 1 and sentence[i:i + 2] in self.dict:
                vec[3, 0] = 1

            if i > 1 and sentence[i - 2:i + 1] in self.dict:
                vec[4, 0] = 1
            if 0 < i < sentenceLen - 1 and sentence[i - 1:i + 2] in self.dict:
                vec[5, 0] = 1
            if i < sentenceLen - 2 and sentence[i:i + 3] in self.dict:
                vec[6, 0] = 1

            if i > 2 and sentence[i - 3:i + 1] in self.dict:
                vec[7, 0] = 1
            if 1 < i < sentenceLen - 1 and sentence[i - 2:i + 2] in self.dict:
                vec[8, 0] = 1
            if 0 < i < sentenceLen - 2 and sentence[i - 1:i + 3] in self.dict:
                vec[9, 0] = 1
            if i < sentenceLen - 3 and sentence[i:i + 4] in self.dict:
                vec[10, 0] = 1

            if i > 3 and sentence[i - 4:i + 1] in self.dict:
                vec[11, 0] = 1
            if 2 < i < sentenceLen - 1 and sentence[i - 3:i + 2] in self.dict:
                vec[12, 0] = 1
            if 1 < i < sentenceLen - 2 and sentence[i - 2:i + 3] in self.dict:
                vec[13, 0] = 1
            if 0 < i < sentenceLen - 3 and sentence[i - 1:i + 4] in self.dict:
                vec[14, 0] = 1
            if i < sentenceLen - 4 and sentence[i:i + 5] in self.dict:
                vec[15, 0] = 1

        return vectors

    def getSentenceTagsVectors(self, line: str) -> (str, str, [np.ndarray]):
        words = CWS.getWords(line = line)
        sentence = ''.join(words)
        tags = ''.join(map(CWS.getTags, words))
        vectors = self.getVectors(sentence = sentence)
        return sentence, tags, vectors

    def getSentenceLabelsVectors(self, line: str) -> (str, [int], [np.ndarray]):
        words = CWS.getWords(line = line)
        sentence = ''.join(words)
        labels = sum(map(CWS.getLabels, words), [])
        vectors = self.getVectors(sentence = sentence)
        return sentence, labels, vectors

    def train(self, file: str, encoding = 'UTF-8'):
        with open(file = file, mode = 'rt', encoding = encoding) as trainingData:
            updateTime = 0
            weightsSum = np.zeros(shape = self.perceptron.weights.shape)
            for line in trainingData.readlines():
                sentence, tags, vectors = self.getSentenceTagsVectors(line)
                updateTime += len(sentence)
                for vector, tag in zip(vectors, tags):
                    self.perceptron.update(x = vector, y = CWS.tagToLabel[tag])
                    weightsSum += self.perceptron.weights
            self.perceptron.learningRate *= 0.9
            print('#' * 80)
            self.perceptron.weights = weightsSum / updateTime

    def segment(self, sentence: str) -> str:
        vectors = self.getVectors(sentence = sentence)
        segmented = '  '
        for c, vec in zip(sentence, vectors):
            tag = CWS.labelToTag[self.perceptron.predict(x = vec)]
            if (tag == 'S' or tag == 'B') and segmented[-1] != ' ':
                segmented += '  '
            segmented += c
            if tag == 'S' or tag == 'E':
                segmented += '  '
        return segmented.strip()

    def test(self, inputFile: str, outputFile: str, encoding = 'UTF-8'):
        with open(file = inputFile, mode = 'rt', encoding = encoding) as testData:
            with open(file = outputFile, mode = 'w', encoding = encoding) as testResult:
                for line in testData.readlines():
                    sentence = CWS.getSentence(line)
                    segmented = self.segment(sentence)
                    testResult.write('%s\n' % segmented)

    @staticmethod
    def evaluate(resultFile: str, answerFile: str, encoding = 'UTF-8'):
        with open(file = resultFile, mode = 'rt', encoding = encoding) as result:
            with open(file = answerFile, mode = 'rt', encoding = encoding) as answer:
                resultNum, answerNum, correctNum = 0, 0, 0
                for lineResult, lineAnswer in zip(result.readlines(), answer.readlines()):
                    wordsResult = set(CWS.getWords(line = lineResult))
                    wordsAnswer = set(CWS.getWords(line = lineAnswer))
                    resultNum += len(wordsResult)
                    answerNum += len(wordsAnswer)
                    correctNum += len(wordsAnswer.intersection(wordsResult))
                P = correctNum / resultNum
                R = correctNum / answerNum
                F = 2 * P * R / (P + R)
                return P, R, F


CWS = ChineseWordSegmenter

if __name__ == '__main__':
    CWSDemo = CWS(learningRate = 1)
    try:
        CWSDemo.loadDict(file = 'dictionary.txt', encoding = 'UTF-8')
    except FileNotFoundError:
        CWSDemo.makeDict(file = 'train.txt', encoding = 'UTF-8')
        CWSDemo.saveDict(file = 'dictionary.txt', encoding = 'UTF-8')

    for i in range(5):
        t0 = time()
        CWSDemo.train(file = 'train.txt', encoding = 'UTF-8')
        print(time() - t0)
        CWSDemo.test(inputFile = 'test.txt', outputFile = 'testResult.txt', encoding = 'UTF-8')
        print(CWS.evaluate(resultFile = 'testResult.txt', answerFile = 'test.answer.txt', encoding = 'UTF-8'))
    pass
