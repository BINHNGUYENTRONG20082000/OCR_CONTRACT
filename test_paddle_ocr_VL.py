from paddleocr import PaddleOCRVL
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
pipeline = PaddleOCRVL(pipeline_version="v1")
start_time = time.time()
output = pipeline.predict("/home/chatbot/AIPT_OCR/Screenshot 2026-06-03 133248.jpg")
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
