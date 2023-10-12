import gradio as gr
import os

def imageSelection(x):
    return x

with gr.Blocks() as demo:

    with gr.Row():
        im = gr.Image()
        im_2 = gr.Image()

    btn = gr.Button(value="Select Image")
    btn.click(imageSelection, inputs=[im], outputs=[im_2])

if __name__ == "__main__":
    demo.launch()



