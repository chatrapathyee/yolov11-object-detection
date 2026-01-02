from ultralytics import YOLO

try:
    model = YOLO('best.pt')
    print("✅ Model loaded successfully!")
    print(f"Model classes: {model.names}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
