import gradio as gr
from exercise_monitor import start_exercise_monitor, start_bridging_exercise_monitor, start_leg_raise_monitor
from model.user import User

# Import modules from other files
from qdrant import model_inference, chatbot

# current user
current_user = None


# Function to fetch and print query parameters
def query_params(request: gr.Request):
    global current_user
    if request:
        query_params = dict(request.query_params)
        print(query_params)
        current_user = User(
            query_params["_id"], query_params["name"], query_params["email"])

    return query_params  # Optional: return the params if needed


# Chat interface block
with gr.Blocks(
        css="""
        .gradio-container .avatar-container {height: 40px; width: 40px !important;}
        #duplicate-button {margin: auto; color: white; background: #f1a139; border-radius: 100vh; margin-top: 2px; margin-bottom: 2px;}
        """,
) as chat:
    gr.ChatInterface(
        fn=model_inference,
        chatbot=chatbot,
        multimodal=True,
        autofocus=True,
        concurrency_limit=10,
    )

# Function to handle exercise clicks


def start_exercise(exercise_name, mode="liveCam", path=None):
    print(f"Starting exercise: {exercise_name} with mode: {mode}")
    if (exercise_name == "Bridging"):
        start_bridging_exercise_monitor(user_id=current_user.get_id()) if mode == "liveCam" else start_bridging_exercise_monitor(
            video_path=path, user_id=current_user.get_id())

    if (exercise_name == "Exercise 2"):
        start_leg_raise_monitor(user_id=current_user.get_id()) if mode == "liveCam" else start_leg_raise_monitor(
            video_path=path, user_id=current_user.get_id())

    else:
        start_exercise_monitor()
    return f"Started monitoring: {exercise_name}"

# Show Upload Document Module


def show_uploader(file_uploader):
    return gr.update(visible=True, interactive=True)

# Return file name after upload


def handle_upload(file):
    if file is None:
        return "No file selected"
    return file.name


# Exercise interface block with embedded video previews and clickable buttons
with gr.Blocks() as exercise:
    gr.Markdown("### Exercise Gallery")

    with gr.Row():
        with gr.Column():
            gr.Video("data/video/exercise1.mp4",
                     label="Bridging", autoplay=True, loop=True)
            with gr.Row():
                ex1_liveCam_button = gr.Button("Live Camera")
                ex1_upload_button = gr.Button("Upload Video")
            ex1_file_input = gr.File(
                label="Select Video", visible=False, interactive=True)

        with gr.Column():
            gr.Video("data/video/Lying leg lift_2.mp4",
                     label="Exercise 2", autoplay=True, loop=True)
            with gr.Row():
                ex2_liveCam_button = gr.Button("Live Camera")
                ex2_upload_button = gr.Button("Upload Video")
            ex2_file_input = gr.File(
                label="Select Video", visible=False, interactive=True)

    with gr.Row():
        with gr.Column():
            gr.Video("data/video/Ankle plantar Flexion.mp4",
                     label="Exercise 3", autoplay=True, loop=True)
            with gr.Row():
                ex3_liveCam_button = gr.Button("Live Camera")
                ex3_upload_button = gr.Button("Upload Video")
            ex3_file_input = gr.File(
                label="Select Video", visible=False, interactive=True)

        with gr.Column():
            gr.Video("data/video/Seated calf stretch.mp4",
                     label="Exercise 4", autoplay=True, loop=True)
            with gr.Row():
                ex4_liveCam_button = gr.Button("Live Camera")
                ex4_upload_button = gr.Button("Upload Video")
            ex4_file_input = gr.File(
                label="Select Video", visible=False, interactive=True)

    def upload_handler(file, exercise_name):
        start_exercise(exercise_name, mode="upload", path=file.name)
        # Hide the file uploader after file is selected
        return gr.update(visible=False)

   # Link buttons to their corresponding exercises
    ex1_liveCam_button.click(
        fn=lambda: start_exercise("Bridging", mode="liveCam"))
    ex1_upload_button.click(fn=lambda: show_uploader(
        ex1_file_input), outputs=ex1_file_input)
    ex1_file_input.change(fn=lambda file: upload_handler(
        file, "Bridging"), inputs=ex1_file_input)

    ex2_liveCam_button.click(
        fn=lambda: start_exercise("Exercise 2", mode="liveCam"))
    ex2_upload_button.click(fn=lambda: show_uploader(
        ex2_file_input), outputs=ex2_file_input)
    ex2_file_input.change(fn=lambda file: upload_handler(
        file, "Exercise 2"), inputs=ex2_file_input)

    ex3_liveCam_button.click(
        fn=lambda: start_exercise("Exercise 3", mode="liveCam"))
    ex3_upload_button.click(fn=lambda: show_uploader(
        ex3_file_input), outputs=ex3_file_input)
    ex3_file_input.change(fn=lambda file: upload_handler(
        file, "Exercise 3"), inputs=ex3_file_input)

    ex4_liveCam_button.click(
        fn=lambda: start_exercise("Exercise 4", mode="liveCam"))
    ex4_upload_button.click(fn=lambda: show_uploader(
        ex4_file_input), outputs=ex4_file_input)
    ex4_file_input.change(fn=lambda file: upload_handler(
        file, "Exercise 4"), inputs=ex4_file_input)


# Main application block
with gr.Blocks() as demo:
    demo.load(fn=query_params, inputs=[], outputs=[])
    gr.TabbedInterface([chat, exercise], ['💬 SuperChat', '🏋️‍♂️ Exercises'])


demo.queue(max_size=300)
demo.launch(share=True)
