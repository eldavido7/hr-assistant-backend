# cache_models.py
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

ef = ONNXMiniLM_L6_V2()
# Trigger download by doing a tiny embed call:
_ = ef(["warmup"])
print("✅ ONNX MiniLM model cached inside the image")
