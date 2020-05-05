#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from typing import Any, Text, Dict, List

from farm.infer import Inferencer
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
import spacy
import pyphen

class ActionGiveFeedback(Action):

    def __init__(self):
        self.model = Inferencer.load("/app/actions/ModelTob")

    def name(self) -> Text:
        return "action_give_feedback"

    def predict_components(self, text :str):
        text_to_analyze = [{'text': '{}'.format(text)}]
        result = self.model.inference_from_dicts(dicts=text_to_analyze)

        annotated_text = [[i['label'], i['start'], i['end']] for i in result[0]['predictions'] if i['probability'] > 0.75]

        count = 0
        count_claim = 0
        count_premise = 0
        elements = []
        for ann in annotated_text:
            if ann[0] != 'O':
                elements.append({
                    'id': count,
                    'label': ann[0].lower(),
                    'start': ann[1],
                    'end': ann[2]
                })
                if ann[0].lower() == 'claim':
                    count_claim += 1
                else:
                    count_premise += 1
            else:
                continue
            count += 1

        return elements, count_claim, count_premise


    def prepare_feedback(self, text: str, elements: tuple):
        feedback_text = ""
        before = 0
        for e in elements[0]:
            start = e['start']
            end = e['end']
            marker = '__' if e['label'] == 'claim' else '_'
            feedback_text += text[before:start]
            feedback_text += marker
            feedback_text += text[start:end]
            feedback_text += marker
            before = end
        if before == 0:
            feedback_text += text

        if elements[1] > elements[2] or elements[1] < 2:
            if elements[1] < 2:
                feedback_text += "\n\n\n\n" \
                                 "[ Ich würde dir empfehlen, deinen Text noch argumentativer zu gestalten. ]() " \
                                 "[ Versuche mindestens zwei Claims mit relevanten Prämissen zu stützen. ]() \n"
            else:
                feedback_text += "\n\n\n\n" \
                                 "[ Ich würde dir empfehlen, deinen Text noch argumentativer zu gestalten. ]()" \
                                 "[ Versuche Deine Claims besser mit relevanten Prämissen zu stützen. ]() \n"
        else:
            feedback_text += "\n\n\n\n" \
                             "[ Ich empfinde Deine Argumentation als gelungen!]() " \
                             "[ Du hast mehrere Aussagen gemacht und diese mit relevanten Prämissen gestützt. Weiter so! ]()\n"

        return feedback_text

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        last_utterance = tracker.latest_message["text"]
        elements = self.predict_components(last_utterance) #Todo text muss noch geholt werden
        feedback = self.prepare_feedback(last_utterance, elements) # Todo text muss noch geholt werden
        dispatcher.utter_message(feedback)
        return []

class ActionGiveScore(Action):

    def name(self) -> Text:
        return "action_give_score"

    def get_number_of_sentences(self, text):
        nlp = spacy.load('de_core_news_sm')
        doc = nlp(text)
        sentences = [sent.string.strip() for sent in doc.sents]

        return len(sentences)

    def count_syllables(self, token:str):
        dic = pyphen.Pyphen(lang='de')
        split_token = dic.inserted(token)
        syllables = split_token.split("-")
        return len(syllables)

    def fre_german(self, text):
        count_tok = 0
        count_s = 0
        for token in text:
            count_tok +=1
            count_s = count_s + self.count_syllables(token)

        number_of_sentences = self.get_number_of_sentences(text)


        asw = count_s / count_tok
        asl = count_tok / number_of_sentences

        fre = (206.835 - (1.015 * asl) - (84.6 * asw))
        fre_1= round(fre,2)
        return fre_1

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        last_utterance = tracker.latest_message["text"]
        score=self.fre_german(last_utterance)
        dispatcher.utter_message(" Dein Readability Score beträgt: {}".format(score))
        return []