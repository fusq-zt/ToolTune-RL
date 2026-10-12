"""CPU-only data and normalization contracts; no model weights are loaded."""

import copy
import unittest

from tooltune.efficiency import AccumulationCheck, encode_reviewed_rows, validate_pairs


def row(key, kind="matched"):
    return {
        "task_id": key,
        "kind": kind,
        "training_split": "train",
        "reviewed_correct": True,
        "messages": [
            {"role": "system", "content": "ordinary prompt"},
            {"role": "user", "content": key},
            {"role": "assistant", "content": "answer"},
        ],
    }


class TemplateFixture:
    """Small explicit template to test role provenance, not Qwen compatibility."""

    def encode(self, text, add_special_tokens=False):
        return [10] if text == "\n" else [99]

    def apply_chat_template(self, messages, **kwargs):
        ids = [1, 2]
        for message in messages[2:]:
            ids += [3, 4, 10] if message["role"] == "assistant" else [90, 91]
        return ids


class EfficiencyTests(unittest.TestCase):
    def test_identical_matched_pairs_are_kept(self):
        data = [row("q"), row("r", "replay")]
        validate_pairs(data, copy.deepcopy(data), 1, 1)

    def test_pair_order_and_shared_replay_are_enforced(self):
        data = [row("q"), row("r", "replay")]
        bad = copy.deepcopy(data)
        bad[1]["messages"][-1]["content"] = "different replay"
        with self.assertRaises(ValueError):
            validate_pairs(data, bad, 1, 1)
        with self.assertRaises(ValueError):
            validate_pairs(data, list(reversed(data)), 1, 1)

    def test_evaluation_and_unreviewed_rows_are_rejected(self):
        for key, value in [("training_split", "test"), ("reviewed_correct", False)]:
            data = [row("q"), row("r", "replay")]
            data[0][key] = value
            with self.assertRaises(ValueError):
                validate_pairs(data, copy.deepcopy(data), 1, 1)

    def test_assistant_only_masks_preserve_tool_attention(self):
        data = row("q")
        data["messages"] += [
            {"role": "tool", "content": "feedback"},
            {"role": "assistant", "content": "final"},
        ]
        result = encode_reviewed_rows(TemplateFixture(), [data])[0]
        self.assertEqual(result["input_ids"], [1, 2, 3, 4, 10, 90, 91, 3, 4, 10])
        self.assertEqual(
            result["labels"], [-100, -100, 3, 4, -100, -100, -100, 3, 4, -100]
        )
        self.assertEqual(result["attention_mask"], [1] * 10)
        data["labels"] = result["input_ids"]
        with self.assertRaises(ValueError):
            encode_reviewed_rows(TemplateFixture(), [data])
        with self.assertRaises(ValueError):
            encode_reviewed_rows(TemplateFixture(), [row("q")], max_length=3)

    def test_266_rows_give_34_updates_and_two_tail_groups_of_ten(self):
        check = AccumulationCheck(266)
        exposures = 0
        for step in range(1, 35):
            size = 10 if step in (17, 34) else 16
            counts = list(range(1, size + 1))
            for count in counts:
                check.add(count, sum(counts), size)
            result = check.finish(step)
            exposures += result["examples"]
        self.assertEqual(exposures, 532)
        self.assertEqual(check.step, 34)

    def test_wrong_tail_or_microbatch_token_mean_is_rejected(self):
        for denominator, microbatches in [(16, 16), (1, 10)]:
            check = AccumulationCheck(10, epochs=1)
            for _ in range(10):
                check.add(1, denominator, microbatches)
            with self.assertRaises(RuntimeError):
                check.finish(1)


if __name__ == "__main__":
    unittest.main()
