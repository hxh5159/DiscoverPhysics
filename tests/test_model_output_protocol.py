"""Model transport format/diagnostics only; no physics or model inference."""
from io import BytesIO
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from agent.models import ModelClient, InferenceError, action_output_schema


class Opener:
    def __init__(self, *results):
        self.results = list(results)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return BytesIO(json.dumps(result).encode())


def response(content, finish='stop', **more):
    return {'choices': [{'message': {'content': content}, 'finish_reason': finish}], **more}


class ModelOutputProtocolTests(unittest.TestCase):
    def client(self, *results, mode='off'):
        client = ModelClient('http://127.0.0.1:8000', structured_output=mode)
        client.model = 'Qwen3.8-27B'
        client._opener = Opener(*results)
        return client

    def test_default_protocol_and_redacted_diagnostics(self):
        content = '{"action":"submit","law":"sensitive candidate"}'
        client = self.client(response(content, usage={'prompt_tokens': 30,
            'completion_tokens': 15, 'total_tokens': 45, 'secret': 'not a token count'}))
        self.assertEqual(client.complete([{'role': 'user', 'content': 'sensitive observation'}]), content)
        body = json.loads(client._opener.requests[0][0].data)
        self.assertNotIn('structured_outputs', body)
        report = client.last_generation_report
        self.assertEqual(report['usage'], {'prompt_tokens': 30, 'completion_tokens': 15, 'total_tokens': 45})
        self.assertEqual(report['finish_reason'], 'stop')
        self.assertEqual(report['output_characters'], len(content))
        self.assertEqual(report['status'], 'received')
        self.assertGreaterEqual(report['elapsed_seconds'], 0)
        self.assertNotIn('sensitive', json.dumps(report))
        self.assertNotIn('secret', json.dumps(report))

    def test_opt_in_schema_allows_saved_law_and_batch_without_answers(self):
        client = self.client(response('{"action":"archive"}'), mode='json_schema')
        client.complete([{'role': 'user', 'content': 'request'}], max_tokens=2048)
        body = json.loads(client._opener.requests[0][0].data)
        schema = body['structured_outputs']['json']
        self.assertEqual(schema, action_output_schema())
        self.assertEqual(body['max_tokens'], 2048)
        self.assertEqual(schema['required'], ['action'])
        self.assertEqual(schema['properties']['law_ref']['enum'], ['saved'])
        self.assertEqual(schema['properties']['actions']['maxItems'], 8)
        self.assertNotIn('batch', schema['properties']['actions']['items']['properties']['action']['enum'])
        self.assertNotIn('default', json.dumps(schema))
        self.assertNotIn('const', json.dumps(schema))

    def test_schema_accepts_both_claim_values_and_real_batch_shapes(self):
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            self.skipTest('Optional schema-validator package is not installed.')
        validator = Draft202012Validator(action_output_schema())
        for order in (1, 2):
            for hidden in (False, True):
                for timed in (False, True):
                    validator.validate({'action': 'draft', 'law': 'generated code', 'claims': {
                        'dynamics_order': order, 'hidden_matter': hidden, 'time_dependence': timed}})
        validator.validate({'action': 'batch', 'actions': [
            {'action': 'launch', 't_launch': 0, 'readout': [0, .1], 'source_properties': {},
             'probes': [{'pos': [1, 2], 'vel': [0, 0], 'properties': {}}]},
            {'action': 'fit', 'law_ref': 'saved'},
            {'action': 'submit', 'law_ref': 'saved'}]})

    def test_environment_opt_in_is_explicit_and_overridable(self):
        with patch.dict(os.environ, {'SHORE_STRUCTURED_OUTPUT': 'json_schema'}):
            self.assertEqual(ModelClient('http://127.0.0.1:8000').structured_output, 'json_schema')
            self.assertEqual(ModelClient('http://127.0.0.1:8000', structured_output='off').structured_output, 'off')
        with patch.dict(os.environ, {'SHORE_STRUCTURED_OUTPUT': 'misspelled'}):
            with self.assertRaises(InferenceError):
                ModelClient('http://127.0.0.1:8000')

    def test_schema_rejection_is_one_call_without_silent_retry(self):
        client = self.client(HTTPError('http://127.0.0.1:8000', 400, 'unsupported', {}, None), mode='json_schema')
        with self.assertRaisesRegex(InferenceError, 'HTTP 400'):
            client.complete([{'role': 'user', 'content': 'request'}])
        self.assertEqual(len(client._opener.requests), 1)
        self.assertEqual(client.last_generation_report['status'], 'error')
        self.assertEqual(client.last_generation_report['error_type'], 'InferenceError')
        self.assertIsNone(client.last_response)

    def test_timeout_keeps_failure_elapsed_and_clears_previous_response(self):
        client = self.client(response('{"action":"archive"}'), TimeoutError())
        client.complete([{'role': 'user', 'content': 'request'}])
        with patch('agent.models.time.monotonic', side_effect=[10., 310.]):
            with self.assertRaises(InferenceError):
                client.complete([{'role': 'user', 'content': 'request'}])
        self.assertEqual(len(client.generation_reports), 2)
        self.assertEqual(client.generation_reports[0]['status'], 'received')
        self.assertEqual(client.last_generation_report['status'], 'error')
        self.assertEqual(client.last_generation_report['elapsed_seconds'], 300.)
        self.assertIsNone(client.last_response)

    def test_length_finish_is_recorded_without_fabricating_or_retrying_output(self):
        content = '{"action":"draft","law":'
        client = self.client(response(content, finish='length'))
        self.assertEqual(client.complete([{'role': 'user', 'content': 'request'}]), content)
        self.assertEqual(client.last_generation_report['finish_reason'], 'length')
        self.assertEqual(len(client._opener.requests), 1)

    def test_thinking_only_is_distinguished_without_saving_reasoning(self):
        result = {'choices': [{'message': {'content': None, 'reasoning_content': 'private chain'},
                               'finish_reason': 'length'}]}
        client = self.client(result)
        self.assertEqual(client.complete([{'role': 'user', 'content': 'request'}]), '')
        self.assertEqual(client.last_generation_report['status'], 'thinking_only')
        self.assertNotIn('private chain', json.dumps(client.last_generation_report))


if __name__ == '__main__':
    unittest.main()
