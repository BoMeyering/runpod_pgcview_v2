# runpod serverless handler for PGCView v2
## Local testing
Run the following command in terminal to test locally:
```bash
python runpod_handler.py
```
That should pull in the data from `test_input.json` and run the handler function.

Or set up a test deployment server by the following command:
```bash
python runpod_handler.py --rp_serve_api
```

and then in a separate terminal, send a test request using the test_input.json:
```
curl -X POST http://localhost:8000/runsync
-H "Content-Type: application/json"
-d @test_input.json
```

This should return a JSON response with the segmentation and marker detection results.
