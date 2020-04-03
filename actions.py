#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from typing import Any, Text, Dict, List

from farm.infer import Inferencer
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

class ActionGiveFeedback(Action):

    def __init__(self):
        self.model = Inferencer.load("https://github.com/TKueng/argubotV1/tree/master/ModelTob")

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
        feedback_text = "Hier kommt das Feedback zu Deiner Argumentation, " \
                        "Claims werden *fett* und Premises _kursiv_ dargestellt:\n\n\n"
        before = 0
        for e in elements[0]:
            start = e['start']
            end = e['end']
            marker = '*' if e['label'] == 'claim' else '_'
            feedback_text += text[before:start]
            feedback_text += marker
            feedback_text += text[start:end]
            feedback_text += marker
            before = end
        if before == 0:
            feedback_text += text

        if elements[1] > elements[2] or elements[1] < 2:
            if elements[1] < 2:
                feedback_text += "\n\nIch würde dir empfehlen, deinen Text noch argumentativer zu gestalten. " \
                                 "Versuche mindestens zwei Claims mit relevanten Premises zu stützen. \n"
            else:
                feedback_text += "\n\nIch würde dir empfehlen, deinen Text noch argumentativer zu gestalten. " \
                                 "Versuche Deine Claims besser mit relevanten Premises zu stützen. \n"
        else:
            feedback_text += "\n\nIch empfinde Deine Argumentation als gelungen! " \
                             "Du hast mehrere Aussagen gemacht und diese mit relevanten Premises gestützt. Weiter so! \n"

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