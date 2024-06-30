import pymupdf
from PIL import Image
import io
import gradio as gr
import base64
import pandas as pd
import pymupdf


def image_to_bytes(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")


def extract_pdfs(docs, doc_collection):
    if docs:
        doc_collection = []
        doc_collection.extend(docs)
    return (
        doc_collection,
        gr.Tabs(selected=1),
        pd.DataFrame([i.split("/")[-1] for i in list(docs)], columns=["Filename"]),
    )


def extract_images(docs):
    images = []
    for doc_path in docs:
        doc = pymupdf.open(doc_path)

        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images()

            for _, img in enumerate(image_list, start=1):
                xref = img[0]
                pix = pymupdf.Pixmap(doc, xref)

                if pix.n - pix.alpha > 3:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

                images.append(Image.open(io.BytesIO(pix.pil_tobytes("JPEG"))))
    return images


def clean_text(text):
    text = text.strip()
    cleaned_text = text.replace("\n", " ")
    cleaned_text = cleaned_text.replace("\t", " ")
    cleaned_text = cleaned_text.replace("  ", " ")
    cleaned_text = cleaned_text.strip()
    return cleaned_text
