# from my_app import MyApp
# import uvicorn

# if __name__ == "__main__":
#     model_full_path = "yolov8_model/result/weights/last.pt"##### change dir
#     upload_folder = ""
#     source_folder_path = 'folders/runs/detect/prediction/predict' ####### change dir
#     destination_folder_path = 'folders/runs/detect/prediction' ####### change dir
#     host = "127.0.0.1"
#     port = 8000

#     my_app = MyApp(model_full_path, upload_folder, source_folder_path, destination_folder_path, host, port)
#     my_app.setup_routes()

#     uvicorn.run(my_app, host=my_app.host, port=my_app.port)

from pyngrok import ngrok
import uvicorn

from my_app import MyApp

def start_ngrok_and_uvicorn(app, port):
    # Create a public URL using ngrok
    public_url = ngrok.connect(port)

    # Print ngrok public URL
    print('Ngrok Tunnel URL:', public_url)

    # Start the uvicorn server
    uvicorn.run(app, host='0.0.0.0', port=port)

if __name__ == "__main__":
    # Replace 'your_authtoken' with the actual authtoken you obtained from ngrok
    ngrok.set_auth_token('2cWNJ6RxJ7ytZQE5vH6PUin2VW6_63TubrJiwfQNvbrzS6eWS')

    model_full_path = "yolov8_model/result/weights/last.pt"
    upload_folder = ""
    source_folder_path = 'folders/runs/detect/prediction/predict'
    destination_folder_path = 'folders/runs/detect/prediction'
    host = "127.0.0.1"
    port = 8000

    my_app = MyApp(model_full_path, upload_folder, source_folder_path, destination_folder_path, host, port)
    my_app.setup_routes()

    start_ngrok_and_uvicorn(my_app, port)

# The rest of your code will be blocked until you manually stop the execution.
