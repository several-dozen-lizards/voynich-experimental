import hashlib
import json
from pathlib import Path
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from server import DEFAULT_LAB, Voynich, handler_for, request_options


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Voynich(DEFAULT_LAB)

    def options(self, prompt='qokedy chedy', **kwargs):
        body = dict(model='voynich-experimental', seed=42, max_tokens=32,
                    messages=[dict(role='user', content=prompt)])
        body.update(kwargs)
        return request_options(body)

    def test_checkpoint_identity(self):
        checkpoint = Path(DEFAULT_LAB) / 'weights/model.pt'
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            '9342df1f11f874676dc0dd3e8ba8409ca158d1e8474077aa42644302182b6651')
        self.assertEqual(self.engine.info['parameters'], 3189035)
        self.assertEqual(self.engine.info['context_tokens'], 64)

    def test_reference_generation_and_reproducibility(self):
        result = self.engine.complete(self.options())
        self.assertEqual(result, self.engine.complete(self.options()))
        self.assertEqual(result[0], ' or ol al dchy qokar ytodar chol da')

    def test_long_prompt_and_output_limit(self):
        options = self.options('qokedy ' * 1000, max_tokens=10000)
        self.assertEqual(options[1], 256)
        _, _, usage = self.engine.complete(self.options('qokedy ' * 1000, max_tokens=2))
        self.assertEqual(usage['prompt_tokens'], 64)
        self.assertEqual(usage['completion_tokens'], 2)

    def test_http_json_and_sse_match(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(self.engine))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            base = f'http://127.0.0.1:{server.server_port}'
            with urlopen(base + '/health', timeout=10) as response:
                self.assertEqual(json.load(response)['parameters'], 3189035)
            body = dict(model='voynich-experimental', seed=42, max_tokens=8,
                        messages=[dict(role='user', content='qokedy chedy')])
            def post(payload):
                request = Request(base + '/v1/chat/completions',
                    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
                with urlopen(request, timeout=30) as response:
                    return response.read().decode()
            expected = json.loads(post(body))['choices'][0]['message']['content']
            events = post(dict(body, stream=True))
            chunks = [json.loads(line[6:]) for line in events.splitlines()
                      if line.startswith('data: ') and line != 'data: [DONE]']
            actual = ''.join(c['choices'][0]['delta'].get('content', '')
                             for c in chunks if c.get('choices'))
            self.assertEqual(actual, expected)
            self.assertIn('data: [DONE]', events)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
