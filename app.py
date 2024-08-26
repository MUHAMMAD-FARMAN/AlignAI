import gradio as gr
from exercise_monitor import start_exercise_monitor, start_bridging_exercise_monitor

# Import modules from other files
from qdrant import model_inference, chatbot

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
        start_bridging_exercise_monitor() if mode == "liveCam" else start_bridging_exercise_monitor(
            video_path=path)
    else:
        start_exercise_monitor()
    return f"Started monitoring: {exercise_name}"


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
            ex1_file_input = gr.File(label="Select Video", visible=False)

        with gr.Column():
            gr.Video("data/video/Lying leg lift_2.mp4",
                     label="Exercise 2", autoplay=True, loop=True)
            with gr.Row():
                ex2_liveCam_button = gr.Button("Live Camera")
                ex2_upload_button = gr.Button("Upload Video")
            ex2_file_input = gr.File(label="Select Video", visible=False)

    with gr.Row():
        with gr.Column():
            gr.Video("data/video/Ankle plantar Flexion.mp4",
                     label="Exercise 3", autoplay=True, loop=True)
            with gr.Row():
                ex3_liveCam_button = gr.Button("Live Camera")
                ex3_upload_button = gr.Button("Upload Video")
            ex3_file_input = gr.File(label="Select Video", visible=False)

        with gr.Column():
            gr.Video("data/video/Seated calf stretch.mp4",
                     label="Exercise 4", autoplay=True, loop=True)
            with gr.Row():
                ex4_liveCam_button = gr.Button("Live Camera")
                ex4_upload_button = gr.Button("Upload Video")
            ex4_file_input = gr.File(label="Select Video", visible=False)

   # Link buttons to their corresponding exercises
    ex1_liveCam_button.click(
        fn=lambda: start_exercise("Bridging", mode="liveCam"))
    ex1_upload_button.click(lambda: ex1_file_input.update(visible=True))
    ex1_file_input.change(fn=lambda file: start_exercise(
        "Bridging", mode="upload", path=file.name), inputs=ex1_file_input)

    ex2_liveCam_button.click(
        fn=lambda: start_exercise("Exercise 2", mode="liveCam"))
    ex2_upload_button.click(lambda: ex2_file_input.update(visible=True))
    ex2_file_input.change(fn=lambda file: start_exercise(
        "Exercise 2", mode="upload", path=file.name), inputs=ex2_file_input)

    ex3_liveCam_button.click(
        fn=lambda: start_exercise("Exercise 3", mode="liveCam"))
    ex3_upload_button.click(lambda: ex3_file_input.update(visible=True))
    ex3_file_input.change(fn=lambda file: start_exercise(
        "Exercise 3", mode="upload", path=file.name), inputs=ex3_file_input)

    ex4_liveCam_button.click(
        fn=lambda: start_exercise("Exercise 4", mode="liveCam"))
    ex4_upload_button.click(lambda: ex4_file_input.update(visible=True))
    ex4_file_input.change(fn=lambda file: start_exercise(
        "Exercise 4", mode="upload", path=file.name), inputs=ex4_file_input)


# Main application block
with gr.Blocks() as demo:
    gr.TabbedInterface([chat, exercise], ['💬 SuperChat', '🏋️‍♂️ Exercises'])


demo.queue(max_size=300)
demo.launch(share=True)
