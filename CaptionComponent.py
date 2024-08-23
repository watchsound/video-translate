from typing import List
from abc import ABC, abstractmethod
import numpy as np

class Actor:
    nextId = 0 
    def __init__(self):
        self.id = Actor.getNextId()
        self.hasHintAudio = False

    def getId(self):
        return self.id

    def setId(self, id):
        self.id = id

    def isHasHintAudio(self):
        return self.hasHintAudio

    def setHasHintAudio(self, hasHintAudio):
        self.hasHintAudio = hasHintAudio


    @classmethod
    def getNextId(cls):
        Actor.nextId += 1
        return Actor.nextId

class HasTimeRange(ABC):

    @abstractmethod
    def getStart(self):
        pass

    @abstractmethod
    def getEnd(self):
        pass


class TimeRangeUtils:
    @staticmethod
    def overlap(r1, r2, margin):
        s1 = r1.getStart() - margin
        e1 = r1.getEnd() + margin
        s2 = r2.getStart()
        e2 = r2.getEnd()

        return max(s1, s2) < min(e1, e2)


class Caption(HasTimeRange):

    def __init__(self):
        self.order = 0
        self.start = 0.0
        self.end = 0.0
        self.text = ""
        self.actor = ""
        self.emotion = ""

    def getOrder(self):
        return self.order

    def setOrder(self, order):
        self.order = order

    def getStart(self):
        return self.start

    def setStart(self, start):
        self.start = start

    def getEnd(self):
        return self.end

    def setEnd(self, end):
        self.end = end

    def getActor(self):
        return self.actor

    def setActor(self, actor):
        self.actor = actor

    def getText(self):
        return self.text

    def setText(self, text):
        self.text = text
 
    def getEmotion(self):
        return self.emotion

    def setEmotion(self, emotion):
        self.emotion = emotion
