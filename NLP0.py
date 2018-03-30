import numpy as np
from matplotlib import pyplot as plt
from time import time


class ProgressBar(object):
    def __init__(self):
        self.progress = -1

    def display(self, progress):
        if progress > 100 or progress < self.progress:
            raise ValueError
        if self.progress < 0:
            print('[', end = '')
        for percent in range(self.progress + 1, progress + 1):
            ones = percent % 10
            if percent == 100:
                print(']')
            elif ones >= 5:
                print('>', end = '', flush = True)
            elif ones == 0 or ones == 4:
                print(' ', end = '', flush = True)
            elif ones == 1:
                print(percent // 10, end = '', flush = True)
            elif ones == 2:
                print(0, end = '', flush = True)
            else:
                print('%', end = '', flush = True)
        self.progress = progress
        return


class Perceptron(object):
    def __init__(self, vecDim: int, learningRate: float):
        self.vecDim = vecDim
        self.weights = np.zeros(shape = vecDim)
        self.learningRate = learningRate

    def __str__(self):
        return str(self.weights)

    def __repr__(self):
        return repr(self.weights)

    def copy(self):
        copy = Perceptron(vecDim = self.vecDim, learningRate = self.learningRate)
        copy.weights = self.weights.copy()
        return copy

    def predict(self, x: np.ndarray) -> float:
        return sum(self.weights * x)

    def update(self, x: np.ndarray, y: float):
        if y >= 0:
            self.weights += self.learningRate * x
        else:
            self.weights -= self.learningRate * x


class ChineseWordSegmenter(object):
    tagToLabel = {'S': 0, 'B': 1, 'M': 2, 'E': 3}
    labelToTag = {0: 'S', 1: 'B', 2: 'M', 3: 'E'}

    def __init__(self):
        self.unigramFeatures = set()
        self.bigramFeatures = set()
        self.trigramFeatures = set()
        self.perceptron = Perceptron(vecDim = 37, learningRate = 1)
        self.perceptron.weights = np.random.random(size = 37)

    def getFeatures(self, file: str, encoding = 'UTF-8'):
        with open(file = file, mode = 'rt', encoding = encoding) as data:
            print('getting features ...')
            startTime = time()
            lines = data.readlines()
            lineNum = len(lines)
            progressBar = ProgressBar()
            for k, line in enumerate(lines):
                sentence, tags = CWS.getSentenceTags(line = line)
                sentenceLen = len(sentence)
                for i, (mid, tag) in enumerate(zip(sentence, tags)):
                    left2 = sentence[i - 2] if i >= 2 else '#'
                    left1 = sentence[i - 1] if i >= 1 else '#'
                    right1 = sentence[i + 1] if i + 1 < sentenceLen else '#'
                    right2 = sentence[i + 2] if i + 2 < sentenceLen else '#'
                    self.unigramFeatures.add('1' + mid + tag)
                    self.unigramFeatures.add('2' + left1 + tag)
                    self.unigramFeatures.add('3' + right1 + tag)
                    self.bigramFeatures.add('4' + left1 + mid + tag)
                    self.bigramFeatures.add('5' + left1 + right1 + tag)
                    self.bigramFeatures.add('6' + mid + right1 + tag)
                    self.trigramFeatures.add('7' + left2 + left1 + mid + tag)
                    self.trigramFeatures.add('8' + left1 + mid + right1 + tag)
                    self.trigramFeatures.add('9' + mid + right1 + right2 + tag)
                progressBar.display(progress = int(100 * k / lineNum))
            progressBar.display(progress = 100)
            print('getting time: %f s' % (time() - startTime))

    def loadFeatures(self, file: str, encoding = 'UTF-8'):
        with open(file = file, mode = 'rt', encoding = encoding) as data:
            print('loading features ...')
            self.unigramFeatures.update(eval(data.readline()))
            self.bigramFeatures.update(eval(data.readline()))
            self.trigramFeatures.update(eval(data.readline()))

    def saveFeatures(self, file: str, encoding = 'UTF-8'):
        with open(file = file, mode = 'w', encoding = encoding) as data:
            print('saving features ...')
            data.write('%r\n' % self.unigramFeatures)
            data.write('%r\n' % self.bigramFeatures)
            data.write('%r\n' % self.trigramFeatures)

    def loadModel(self, file: str, encoding = 'UTF-8'):
        with open(file = file, mode = 'rt', encoding = encoding) as data:
            print('loading the model ...')
            self.perceptron.weights = np.array(eval(data.readline()))
            self.unigramFeatures.update(eval(data.readline()))
            self.bigramFeatures.update(eval(data.readline()))
            self.trigramFeatures.update(eval(data.readline()))

    def saveModel(self, file: str, encoding = 'UTF-8'):
        with open(file = file, mode = 'w', encoding = encoding) as data:
            print('saving the model ...')
            data.write('%r\n' % list(self.perceptron.weights))
            data.write('%r\n' % self.unigramFeatures)
            data.write('%r\n' % self.bigramFeatures)
            data.write('%r\n' % self.trigramFeatures)

    @staticmethod
    def getWords(line: str) -> [str]:
        return line.strip().split()

    @staticmethod
    def getTags(word: str) -> str:
        if len(word) == 0:
            return ''
        elif len(word) == 1:
            return 'S'
        else:
            return 'B' + 'M' * (len(word) - 2) + 'E'

    @staticmethod
    def getSentence(line: str) -> str:
        return ''.join(CWS.getWords(line = line))

    @staticmethod
    def getSentenceTags(line: str) -> (str, str):
        words = CWS.getWords(line = line)
        return ''.join(words), ''.join(map(CWS.getTags, words))

    def getVector(self, sentence: str, index: int, tag: str = None, label: int = None) -> np.ndarray:
        sentenceLen = len(sentence)
        if tag is None:
            tag = CWS.labelToTag[label]
        if label is None:
            label = CWS.tagToLabel[tag]
        mid = sentence[index]
        left2 = sentence[index - 2] if index >= 2 else '#'
        left1 = sentence[index - 1] if index >= 1 else '#'
        right1 = sentence[index + 1] if index + 1 < sentenceLen else '#'
        right2 = sentence[index + 2] if index + 2 < sentenceLen else '#'
        vector = np.zeros(self.perceptron.vecDim)
        vector[0] = 1
        vector[label] = int(('1' + mid + tag) in self.unigramFeatures)
        vector[4 + label] = int(('2' + left1 + tag) in self.unigramFeatures)
        vector[8 + label] = int(('3' + right1 + tag) in self.unigramFeatures)
        vector[12 + label] = int(('4' + left1 + mid + tag) in self.bigramFeatures)
        vector[16 + label] = int(('5' + left1 + right1 + tag) in self.bigramFeatures)
        vector[20 + label] = int(('6' + mid + right1 + tag) in self.bigramFeatures)
        vector[24 + label] = int(('7' + left2 + left1 + mid + tag) in self.trigramFeatures)
        vector[28 + label] = int(('8' + left1 + mid + right1 + tag) in self.trigramFeatures)
        vector[32 + label] = int(('9' + mid + right1 + right2 + tag) in self.trigramFeatures)
        return vector

    def predictTag(self, sentence: str, index: int) -> str:
        return max(CWS.tagToLabel.keys(),
                   key = lambda tag: self.perceptron.predict(
                           x = self.getVector(sentence = sentence, index = index, tag = tag)))

    def train(self, file: str, encoding = 'UTF-8', trainingTimes: int = None, totalTrainingTimes: int = None):
        with open(file = file, mode = 'rt', encoding = encoding) as trainingData:
            if trainingTimes is None or totalTrainingTimes is None:
                print('training ...')
            else:
                print('training (%d of %d) ...' % (trainingTimes, totalTrainingTimes))
            startTime = time()
            updateTimes = 0
            weightsSum = np.zeros(shape = self.perceptron.weights.shape)
            lines = trainingData.readlines()
            lineNum = len(lines)
            progressBar = ProgressBar()
            for k, line in enumerate(lines):
                sentence, trueTags = self.getSentenceTags(line)
                for i, trueTag in enumerate(trueTags):
                    trueLabel = CWS.tagToLabel[trueTag]
                    vectors = [self.getVector(sentence = sentence, index = i, label = label)
                               for label in CWS.labelToTag.keys()]
                    predictLabel = max(CWS.labelToTag.keys(),
                                       key = lambda label: self.perceptron.predict(x = vectors[label]))
                    if predictLabel != trueLabel:
                        self.perceptron.update(x = vectors[trueLabel], y = 1)
                        self.perceptron.update(x = vectors[predictLabel], y = -1)
                    updateTimes += 1
                    weightsSum += self.perceptron.weights
                progressBar.display(progress = int(100 * k / lineNum))
            progressBar.display(progress = 100)
            self.perceptron.weights = weightsSum / updateTimes
            print('training time: %f s' % (time() - startTime))

    def segment(self, sentence: str) -> str:
        segmented = '  '
        for i, c in enumerate(sentence):
            tag = self.predictTag(sentence = sentence, index = i)
            if (tag == 'S' or tag == 'B') and segmented[-1] != ' ':
                segmented += '  '
            segmented += c
            if tag == 'S' or tag == 'E':
                segmented += '  '
        return segmented.strip()

    def test(self, testFile: str, resultFile: str, encoding = 'UTF-8'):
        with open(file = testFile, mode = 'rt', encoding = encoding) as testData:
            with open(file = resultFile, mode = 'w', encoding = encoding) as testResult:
                print('testing ...')
                startTime = time()
                lines = testData.readlines()
                lineNum = len(lines)
                progressBar = ProgressBar()
                for k, line in enumerate(lines):
                    segmented = self.segment(sentence = CWS.getSentence(line))
                    testResult.write('%s\n' % segmented)
                    progressBar.display(progress = int(100 * k / lineNum))
                progressBar.display(progress = 100)
                print('testing time: %f s' % (time() - startTime))

    @staticmethod
    def evaluate(resultFile: str, answerFile: str, encoding = 'UTF-8'):
        with open(file = resultFile, mode = 'rt', encoding = encoding) as resultData:
            with open(file = answerFile, mode = 'rt', encoding = encoding) as answerData:
                print('evaluating ...')
                startTime = time()
                resultSegNum, answerSegNum, correctSegNum = 0, 0, 0
                resultLines = resultData.readlines()
                lineNum = len(resultLines)
                progressBar = ProgressBar()
                for k, (lineResult, lineAnswer) in enumerate(zip(resultLines, answerData.readlines())):
                    wordsResult = set(CWS.getWords(line = lineResult))
                    wordsAnswer = set(CWS.getWords(line = lineAnswer))
                    resultSegNum += len(wordsResult)
                    answerSegNum += len(wordsAnswer)
                    correctSegNum += len(wordsAnswer.intersection(wordsResult))
                    progressBar.display(progress = int(100 * k / lineNum))
                progressBar.display(progress = 100)
                P = correctSegNum / resultSegNum
                R = correctSegNum / answerSegNum
                F = 2 * P * R / (P + R)
                print('evaluation time: %f s' % (time() - startTime))
                return P, R, F


CWS = ChineseWordSegmenter

if __name__ == '__main__':
    np.random.seed(0)
    CWSDemo = CWS()
    try:
        CWSDemo.loadModel(file = 'model0.txt', encoding = 'UTF-8')
    except FileNotFoundError:
        try:
            CWSDemo.loadFeatures(file = 'features0.txt', encoding = 'UTF-8')
        except FileNotFoundError:
            CWSDemo.getFeatures(file = 'train.txt', encoding = 'UTF-8')
            CWSDemo.saveFeatures(file = 'features0.txt', encoding = 'UTF-8')
        CWSDemo.saveModel(file = 'model0.txt', encoding = 'UTF-8')

    print('number of unigram features: %d' % len(CWSDemo.unigramFeatures))
    print('number of bigram features: %d' % len(CWSDemo.bigramFeatures))
    print('number of trigram features: %d\n' % len(CWSDemo.trigramFeatures))

    with open(file = 'testScores0.txt', mode = 'a', encoding = 'UTF-8') as testScores:

        CWSDemo.test(testFile = 'test.txt', resultFile = 'testResult.txt', encoding = 'UTF-8')
        P, R, F = CWS.evaluate(resultFile = 'testResult.txt', answerFile = 'test.answer.txt', encoding = 'UTF-8')
        precision, recall, FScore = [P], [R], [F]
        print('starting score:\nPrecision = %f   Recall = %f   F-Score = %f\n' % (P, R, F))
        testScores.write('starting score:\nPrecision = %f   Recall = %f   F-Score = %f\n\n' % (P, R, F))

        totalTrainingTimes = 20
        for trainingTimes in range(totalTrainingTimes):
            CWSDemo.train(file = 'train.txt', encoding = 'UTF-8',
                          trainingTimes = trainingTimes + 1, totalTrainingTimes = totalTrainingTimes)
            CWSDemo.test(testFile = 'test.txt', resultFile = 'testResult.txt', encoding = 'UTF-8')
            P, R, F = CWS.evaluate(resultFile = 'testResult.txt', answerFile = 'test.answer.txt', encoding = 'UTF-8')

            precision.append(P)
            recall.append(R)
            FScore.append(F)
            print('score:\nPrecision = %f   Recall = %f   F-Score = %f' % (P, R, F))
            testScores.write('score:\nPrecision = %f   Recall = %f   F-Score = %f\n\n' % (P, R, F))
            testScores.flush()
            CWSDemo.saveModel(file = 'model0.txt', encoding = 'UTF-8')
            print()
        plt.plot(range(totalTrainingTimes + 1), precision, label = 'Precision')
        plt.plot(range(totalTrainingTimes + 1), recall, label = 'Recall')
        plt.plot(range(totalTrainingTimes + 1), FScore, label = 'F-Score')
        plt.legend(loc = 'lower right', frameon = True)
        plt.xlabel(s = 'training times')
        plt.ylabel(s = 'score')
        plt.xlim(xmin = 0)
        plt.savefig(fname = 'score.png')
        plt.show()
