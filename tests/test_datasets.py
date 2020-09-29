import unittest
from sportsbot.datasets import Tweet, ConversationPrompt
from sportsbot.datasets import _save_data, _read_data

class TestJsonlinesFunctions(unittest.TestCase):
    def test_encode_decode_data(self):
        empty_list = []
        test_conv_obj = ConversationPrompt(
                                [Tweet(12338473830332,
                                        "dog",
                                        "dooog",
                                        "@TheHoopGenius I heard someone say AD better than Giannis 😂😂😂",
                                        "en",
                                        "Tue Mar 29 08:11:25 +0000 2011"
                                        )],empty_list)
        _save_data([test_conv_obj], "read_write_test.jsonl")

        self.assertEqual(test_conv_obj, _read_data("read_write_test.jsonl")[0])
