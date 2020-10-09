import unittest
from sportsbot.datasets import Tweet, Conversation
from sportsbot.datasets import _save_data, read_data, _prepare_few_shot_testing_set
from sportsbot.conversations import get_conversations

class TestJsonlinesFunctions(unittest.TestCase):
    def test_encode_decode_data(self):
        empty_list = []
        test_conv_obj = Conversation(
                                [Tweet(12338473830332,
                                        "dog",
                                        "dooog",
                                        "@TheHoopGenius I heard someone say AD"\
                                             "better than Giannis 😂😂😂",
                                        "en",
                                        "Tue Mar 29 08:11:25 +0000 2011",
                                        122,
                                        1,
                                        "dummy profile"
                                        )],"label","template",empty_list)
        _save_data([test_conv_obj], "read_write_test.jsonl")

        self.assertEqual(test_conv_obj, read_data("read_write_test.jsonl")[0])

class TestTemplateFunctions(unittest.TestCase):
    conversations = get_conversations('"lakers suck"', ["china", "racist"], "lakers")
    training_convs = conversations[:3]
    training_labels = ["Yes"]*3
    testing_convs = conversations[3:]
    full_templates,_= _prepare_few_shot_testing_set(training_convs, testing_convs, "lakers", training_labels)
    for conv in full_templates:
        print(conv)
