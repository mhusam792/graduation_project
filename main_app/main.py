from my_app import MyApp
import uvicorn

if __name__ == "__main__":
    model_full_path = "yolov8_model/result/weights/last.pt"##### change dir
    upload_folder = ""
    source_folder_path = 'folders/runs/detect/prediction/predict' ####### change dir
    destination_folder_path = 'folders/runs/detect/prediction' ####### change dir
    host = "127.0.0.1"
    port = 8000

    my_app = MyApp(model_full_path, upload_folder, source_folder_path, destination_folder_path, host, port)
    my_app.setup_routes()

    uvicorn.run(my_app, host=my_app.host, port=my_app.port)
