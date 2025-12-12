from ultralytics import YOLO
import os

def train_model():
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    # data: path to data.yaml
    # epochs: number of training epochs
    # imgsz: image size
    # batch: batch size (auto if -1)
    # name: name of the experiment
    results = model.train(
        data=os.path.abspath('yolo_workspace/dataset/data.yaml'),
        epochs=50,
        imgsz=640,
        batch=4, # Small batch size for safety on CPU/small GPU
        name='microalgae_yolov8',
        project='yolo_workspace/runs'
    )
    
    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_model()
