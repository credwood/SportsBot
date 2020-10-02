import unittest
import numpy as np
from sportsbot.inference import _top_softmax, _calculate_accuracy
from sportsbot.inference import few_shot_test, download_model_tokenizer
from sportsbot.conversations import get_conversations
from sportsbot.datasets import _read_data

class TestInferenceFunctions(unittest.TestCase):
    model, tokenizer = download_model_tokenizer()

    def test_top_softmax(self, tokenizer=tokenizer):
        text_array = np.array([.1, .06, .46, .007, .09])
        result = [
                    {tokenizer.decode(2): str(.46)},
                    {tokenizer.decode(0): str(.1)},
                    {tokenizer.decode(4): str(.09)},
                    {tokenizer.decode(1): str(.06)},
                    {tokenizer.decode(3): str(.007)}
                ]
        self.assertEqual(_top_softmax(text_array, tokenizer), result)

    def test_calc_accuracy(self):
        labels = [" no", " yes", " yes"]
        model_ans = [" no", " !", " I"]
        self.assertAlmostEqual(_calculate_accuracy(labels, model_ans), 0.3333333333333333333)

    def test_few_shot_test(self, model=model, tokenizer=tokenizer):
        topic = "lakers"
        test_convos = get_conversations('"the lakers suck"', ['racist', 'china'])
        len_data = len(test_convos)
        test_data = test_convos[:len_data-3]
        train_data = test_convos[len_data-3:]
        train_labels = ["Yes"]*len(train_data)
        test = few_shot_test(
                        test_data,
                        topic,
                        train_data,
                        train_labels,
                        tokenizer,
                        model,
                        jsonlines_file_out='stats_output.jsonl'
                        )
        print(test)
        saved_data = _read_data('stats_output.jsonl')
        for conv in saved_data:
            print(conv)
