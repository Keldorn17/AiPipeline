from Models import *


def main() -> None:
    link: str = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    text_response: str = image_to_text_model(link)
    print(text_response)
    prompt_response: list[dict[str, str]] = text_to_text_model(text_response)
    print(prompt_response)
    text_to_image_model(prompt_response)


if __name__ == '__main__':
    main()