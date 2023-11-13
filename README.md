# Mixcraft-Riffusion

Mixcraft-Riffusion is a powerful tool that provides several functionalities. Here is a brief explanation of the available functions in `mixcraft.py` and how to call them:

1. **Inference Pipeline**: This function is used to run the inference pipeline. It can be called with the `--inference` flag. The `--model` flag is used to specify the model name and the `--prompt` flag is used to provide a prompt.

    Example: `python mixcraft.py --inference --model "riffusion/riffusion-model-v1" --prompt "Your prompt"`

2. **Dataset Pipeline**: This function is used to create a dataset. It can be called with the `--create_dataset` flag. The `--hf_dataset` flag is used to specify the HuggingFace dataset name and the `--hf_token` flag is used to provide the HuggingFace token.

    Example: `python mixcraft.py --create_dataset --hf_dataset "Your dataset" --hf_token "Your token"`

3. **Image to Audio Conversion**: This function is used to convert an image to audio. It can be called with the `--image_to_audio` flag. The `--input` flag is used to specify the input image and the `--output` flag is used to specify the output audio file.

    Example: `python mixcraft.py --image_to_audio --input "input.jpg" --output "output.wav"`

4. **Audio to Image Conversion**: This function is used to convert audio to an image. It can be called with the `--audio_to_image` flag. The `--input` flag is used to specify the input audio file and the `--output` flag is used to specify the output image file.

    Example: `python mixcraft.py --audio_to_image --input "input.wav" --output "output.jpg"`
