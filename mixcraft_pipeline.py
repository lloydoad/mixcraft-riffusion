import torch
from diffusers import StableDiffusionPipeline
from riffusion.cli import image_to_audio
from typing import List
import uuid

class MixcraftPipelineParams:
    """
    This class is used to define the parameters for the MixcraftPipeline.
    """
    def __init__(self, model_name: str, device: str = 'cuda'):
        """
        Initializes the MixcraftPipelineParams class.
        
        :param model_name: The name of the model to be used.
        :param device: The device on which the model will be run.
        """
        self.model_name = model_name
        self.device = device

cached_model = None
cached_params = None

class MixcraftPipeline:
    def __init__(self, params: MixcraftPipelineParams) -> None:
        self.params = params
        self.pipeline = StableDiffusionPipeline.from_pretrained(self.params.model_name, 
                                                                torch_dtype=torch.float16)
        self.pipeline.to(self.params.device)

    def run(self, prompt: str) -> List[str]:
        images = self.pipeline(prompt=prompt).images
        for image in images:
            identifier = str(uuid.uuid4())
            image_file_name = f'{identifier}.png'
            audio_file_name = f'{identifier}.wav'
            image.save(image_file_name)
            image_to_audio(image=image_file_name,
                           audio=audio_file_name,
                           device=self.params.device)