import gradio as gr
from exercise_monitor import start_exercise_monitor

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


def start_exercise(exercise_name):
    print(f"Starting exercise: {exercise_name}")
    start_exercise_monitor()
    return f"Started monitoring: {exercise_name}"


# Exercise interface block with embedded video previews and clickable buttons
with gr.Blocks() as exercise:
    gr.Markdown("### Exercise Gallery")

    with gr.Row():
        with gr.Column():
            gr.Video("data/video/exercise1.mp4",
                     label="Exercise 1", autoplay=True, loop=True)
            ex1_button = gr.Button("Start Exercise 1")

        with gr.Column():
            gr.Video("data/video/lying_leg_lift.mp4",
                     label="Exercise 2", autoplay=True, loop=True)
            ex2_button = gr.Button("Start Exercise 2")

    with gr.Row():
        with gr.Column():
            gr.Video("data/video/ankle_plantar_flexion.mp4",
                     label="Exercise 3", autoplay=True, loop=True)
            ex3_button = gr.Button("Start Exercise 3")

        with gr.Column():
            gr.Video("data/video/seated_calf_stretch.mp4",
                     label="Exercise 4", autoplay=True, loop=True)
            ex4_button = gr.Button("Start Exercise 4")

    # Link buttons to their corresponding exercises
    ex1_button.click(fn=lambda: start_exercise(
        "Exercise 1"))
    ex2_button.click(fn=lambda: start_exercise(
        "Exercise 2"))
    ex3_button.click(fn=lambda: start_exercise(
        "Exercise 3"))
    ex4_button.click(fn=lambda: start_exercise(
        "Exercise 4"))

# Main application block
with gr.Blocks() as demo:
    gr.TabbedInterface([chat, exercise], ['💬 SuperChat', '🏋️‍♂️ Exercises'])

demo.queue(max_size=300)
demo.launch(share=True)
