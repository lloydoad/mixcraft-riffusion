import argparse
from mixcraft_pipeline import MixcraftPipelineParams, MixcraftPipeline
from dataset_processor import DatasetProcessorParams, DatasetProcessor
from riffusion.cli import audio_to_image, image_to_audio

# Define the command line arguments
parser = argparse.ArgumentParser()

# INFERENCE PIPELINE
parser.add_argument("--inference", action="store_true")
parser.add_argument("--model", type=str, default="riffusion/riffusion-model-v1")
parser.add_argument("--prompt", type=str, default="")

# DATASET PIPELINE
parser.add_argument("--create_dataset", action="store_true")
parser.add_argument("--hf_dataset", type=str, default="")
parser.add_argument("--hf_token", type=str, default="")

# CONVERSION
parser.add_argument("--image_to_audio", action="store_true")
parser.add_argument("--audio_to_image", action="store_true")

# INPUT / OUTPUT
parser.add_argument("--input", type=str, default="")
parser.add_argument("--output", type=str, default="")

args = parser.parse_args()

if args.inference:
    params = MixcraftPipelineParams(model_name=args.model)
    pipe = MixcraftPipeline(params=params)
    pipe.run(prompt=args.prompt)
elif args.create_dataset:
    params = DatasetProcessorParams()
    params.input_directory = args.input
    params.output_directory = args.output
    params.huggingface_dataset_name = args.hf_dataset
    params.huggingface_token = args.hf_token
    pipe = DatasetProcessor(params=params)
    pipe.run()
elif args.image_to_audio:
    image_to_audio(image=args.input, audio=args.output)
elif args.audio_to_image:
    audio_to_image(audio=args.input, image=args.output)