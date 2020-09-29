import unittest
import numpy as np
from sportsbot.inference import _top_softmax, _calculate_accuracy
from sportsbot.inference import few_shot_train, download_model_tokenizer
from sportsbot.conversations import get_conversations

class TestInferenceFunctions(unittest.TestCase):
    model, tokenizer = download_model_tokenizer()

    def test_top_softmax(self, tokenizer=tokenizer):
        text_array = np.array([.1, .06, .46, .007, .09])
        result = [
                    {tokenizer(2): str(.46)},
                    {tokenizer(0): str(.1)},
                    {tokenizer(4): str(.09)},
                    {tokenizer(1): str(.06)},
                    {tokenizer(3): str(.007)}
                    ]
        self.assertEqual(_top_softmax(text_array, tokenizer), result)

    def test_calc_accuracy(self):
        labels = [" no", " yes", " yes"]
        model_ans = [" no", " !", " I"]
        self.assertAlmostEqual(_calculate_accuracy(labels, model_ans), 0.33)

    def test_few_shot_train(self, model=model, tokenizer=tokenizer):
        topic = "lakers"
        test_convos = get_conversations('"the lakers suck"', ['racist', 'china'])
        len_data = len(test_convos)
        test_data = test_convos[:len_data-3]
        test_labels = [ "Yes"]*len(test_data)
        train_data = test_convos[len_data-3:]
        train_labels = ["Yes"]*len(train_data)
        few_shot_train(
                        test_data,
                        test_labels,
                        topic,
                        train_data,
                        train_labels,
                        tokenizer,
                        model
                        )
