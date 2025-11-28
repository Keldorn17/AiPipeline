import requests
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline


def image_to_text_model(image_url: str) -> str:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to("cpu")

    raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')

    text = "a detailed description of this image: "
    inputs = processor(raw_image, text, return_tensors="pt").to("cpu")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))

    inputs = processor(raw_image, return_tensors="pt").to("cpu")

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


def text_to_text_model(message: str) -> str:
    generator = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct")

    raw = generator(
        f"Rewrite this as a detailed prompt: {message}",
        max_new_tokens=200,
    )[0]["generated_text"]

    if "Rewrite this as a detailed prompt:" in raw:
        raw = raw.split("Rewrite this as a detailed prompt:")[1].strip()

    if "Certainly!" in raw:
        raw = raw.split("Certainly!")[1].strip()

    if "---" in raw:
        raw = raw.split("---")[-1].strip()

    return raw



def text_to_image_model(prompt: str | list[dict[str, str]]) -> None:
    if not isinstance(prompt, str):
        prompt = str(prompt)

    model_id = "sd-legacy/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")

    image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
    image.save("result.jpg")
