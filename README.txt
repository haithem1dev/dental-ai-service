Dental AI Service - Render Ready

1) Put your trained model here:
   models/best.onnx

2) Render Build Command:
   pip install -r requirements.txt

3) Render Start Command:
   uvicorn app:app --host 0.0.0.0 --port $PORT

4) Test URLs:
   /health
   /docs
   /analyze
