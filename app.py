import base64
import chromadb
import gc
import gradio as gr
import io
import numpy as np
import os
import pandas as pd
import pymupdf
from pypdf import PdfReader
# import spaces
import torch
from PIL import Image
from chromadb.utils import embedding_functions
from chromadb.utils.data_loaders import ImageLoader
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from gradio.themes.utils import sizes
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceEndpoint
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from utils import *


def result_to_text(result, as_text=False) -> str or list:
    full_doc = []
    for _, page in enumerate(result.pages, start=1):
        text = ""
        for block in page.blocks:
            text += "\n\t"
            for line in block.lines:
                for word in line.words:
                    text += word.value + " "

        full_doc.append(clean_text(text) + "\n\n")

    return "\n".join(full_doc) if as_text else full_doc


ocr_model = ocr_predictor(
    "db_resnet50",
    "crnn_mobilenet_v3_large",
    pretrained=True,
    assume_straight_pages=True,
)


if torch.cuda.is_available():
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    vision_model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
    )


# @spaces.GPU()
def get_image_description(image):
    torch.cuda.empty_cache()
    gc.collect()

    # n = len(prompt)
    prompt = "[INST] <image>\nDescribe the image in a sentence [/INST]"

    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    output = vision_model.generate(**inputs, max_new_tokens=100)
    return processor.decode(output[0], skip_special_tokens=True)


CSS = """
#table_col {background-color: rgb(33, 41, 54);}
"""


# def get_vectordb(text, images, tables):
def get_vectordb(text, images, img_doc_files):
    client = chromadb.EphemeralClient()
    loader = ImageLoader()
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="multi-qa-mpnet-base-dot-v1"
    )
    if "text_db" in [i.name for i in client.list_collections()]:
        client.delete_collection("text_db")
    if "image_db" in [i.name for i in client.list_collections()]:
        client.delete_collection("image_db")

    text_collection = client.get_or_create_collection(
        name="text_db",
        embedding_function=sentence_transformer_ef,
        data_loader=loader,
    )
    image_collection = client.get_or_create_collection(
        name="image_db",
        embedding_function=sentence_transformer_ef,
        data_loader=loader,
        metadata={"hnsw:space": "cosine"},
    )
    descs = []
    for i in range(len(images)):
        try:
            descs.append(img_doc_files[i] + "\n" + get_image_description(images[i]))
        except:
            descs.append("Could not generate image description due to some error")
        print(descs[-1])
        print()

    # image_descriptions = get_image_descriptions(images)
    image_dict = [{"image": image_to_bytes(img)} for img in images]

    if len(images) > 0:
        image_collection.add(
            ids=[str(i) for i in range(len(images))],
            documents=descs,
            metadatas=image_dict,
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=10,
    )

    if len(text.replace(" ", "").replace("\n", "")) == 0:
        gr.Error("No text found in documents")
    else:
        docs = splitter.create_documents([text])
        doc_texts = [i.page_content for i in docs]
        text_collection.add(
            ids=[str(i) for i in list(range(len(doc_texts)))], documents=doc_texts
        )
    return client


def extract_only_text(reader):
    text = ""
    for _, page in enumerate(reader.pages):
        text = page.extract_text()
    return text.strip()


def extract_data_from_pdfs(
    docs, session, include_images, do_ocr, progress=gr.Progress()
):
    if len(docs) == 0:
        raise gr.Error("No documents to process")
    progress(0, "Extracting Images")

    # images = extract_images(docs)

    progress(0.25, "Extracting Text")

    all_text = ""

    images = []
    img_docs = []
    for doc in docs:
        if do_ocr == "Get Text With OCR":
            pdf_doc = DocumentFile.from_pdf(doc)
            result = ocr_model(pdf_doc)
            all_text += result_to_text(result, as_text=True) + "\n\n"
        else:
            reader = PdfReader(doc)
            all_text += extract_only_text(reader) + "\n\n"

        if include_images == "Include Images":
            imgs = extract_images([doc])
            images.extend(imgs)
            img_docs.extend([doc.split("/")[-1] for _ in range(len(imgs))])

    progress(
        0.6, "Generating image descriptions and inserting everything into vectorDB"
    )
    vectordb = get_vectordb(all_text, images, img_docs)

    progress(1, "Completed")
    session["processed"] = True
    return (
        vectordb,
        session,
        gr.Row(visible=True),
        all_text[:2000] + "...",
        # display,
        images[:2],
        "<h1 style='text-align: center'>Completed<h1>",
        # image_descriptions
    )


sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="multi-qa-mpnet-base-dot-v1"
)


def conversation(
    vectordb_client,
    msg,
    num_context,
    img_context,
    history,
    temperature,
    max_new_tokens,
    hf_token,
    model_path,
):
    if hf_token.strip() != "" and model_path.strip() != "":
        llm = HuggingFaceEndpoint(
            repo_id=model_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            huggingfacehub_api_token=hf_token,
        )
    else:
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            huggingfacehub_api_token=os.getenv("P_HF_TOKEN", "None"),
        )

    text_collection = vectordb_client.get_collection(
        "text_db", embedding_function=sentence_transformer_ef
    )
    image_collection = vectordb_client.get_collection(
        "image_db", embedding_function=sentence_transformer_ef
    )

    results = text_collection.query(
        query_texts=[msg], include=["documents"], n_results=num_context
    )["documents"][0]
    similar_images = image_collection.query(
        query_texts=[msg],
        include=["metadatas", "distances", "documents"],
        n_results=img_context,
    )
    img_links = [i["image"] for i in similar_images["metadatas"][0]]

    images_and_locs = [
        Image.open(io.BytesIO(base64.b64decode(i[1])))
        for i in zip(similar_images["distances"][0], img_links)
    ]
    img_desc = "\n".join(similar_images["documents"][0])
    if len(img_links) == 0:
        img_desc = "No Images Are Provided"
    template = """
    Context:
    {context}

    Included Images:
    {images}
    
    Question:
    {question}

    Answer:

    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    context = "\n\n".join(results)
    # references = [gr.Textbox(i, visible=True, interactive=False) for i in results]
    response = llm(prompt.format(context=context, question=msg, images=img_desc))
    return history + [(msg, response)], results, images_and_locs


def check_validity_and_llm(session_states):
    if session_states.get("processed", False) == True:
        return gr.Tabs(selected=2)
    raise gr.Error("Please extract data first")


with gr.Blocks(css=CSS, theme=gr.themes.Soft(text_size=sizes.text_md)) as demo:
    vectordb = gr.State()
    doc_collection = gr.State(value=[])
    session_states = gr.State(value={})
    references = gr.State(value=[])

    gr.Markdown(
        """<h2><center>Multimodal PDF Chatbot</center></h2>
    <h3><center><b>Interact With Your PDF Documents</b></center></h3>"""
    )
    gr.Markdown(
        """<center><h3><b>Note: </b> This application leverages advanced Retrieval-Augmented Generation (RAG) techniques to provide context-aware responses from your PDF documents</center><h3><br>
    <center>Utilizing multimodal capabilities, this chatbot can interpret and answer queries based on both textual and visual information within your PDFs.</center>"""
    )
    gr.Markdown(
        """
    <center><b>Warning: </b> Extracting text and images from your document and generating embeddings may take some time due to the use of OCR and multimodal LLMs for image description<center>
    """
    )
    with gr.Tabs() as tabs:
        with gr.TabItem("Upload PDFs", id=0) as pdf_tab:
            with gr.Row():
                with gr.Column():
                    documents = gr.File(
                        file_count="multiple",
                        file_types=["pdf"],
                        interactive=True,
                        label="Upload your PDF file/s",
                    )
                    pdf_btn = gr.Button(value="Next", elem_id="button1")

        with gr.TabItem("Extract Data", id=1) as preprocess:
            with gr.Row():
                with gr.Column():
                    back_p1 = gr.Button(value="Back")
                with gr.Column():
                    embed = gr.Button(value="Extract Data")
                with gr.Column():
                    next_p1 = gr.Button(value="Next")
            with gr.Row():
                include_images = gr.Radio(
                    ["Include Images", "Exclude Images"],
                    value="Include Images",
                    label="Include/ Exclude Images",
                    interactive=True,
                )
                do_ocr = gr.Radio(
                    ["Get Text With OCR", "Get Available Text Only"],
                    value="Get Text With OCR",
                    label="OCR/ No OCR",
                    interactive=True,
                )

            with gr.Row(equal_height=True, variant="panel") as row:
                selected = gr.Dataframe(
                    interactive=False,
                    col_count=(1, "fixed"),
                    headers=["Selected Files"],
                )
                prog = gr.HTML(
                    value="<h1 style='text-align: center'>Click the 'Extract' button to extract data from PDFs<h1>"
                )

            with gr.Accordion("See Parts of Extracted Data", open=False):
                with gr.Column(visible=True) as sample_data:
                    with gr.Row():
                        with gr.Column():
                            ext_text = gr.Textbox(
                                label="Sample Extracted Text", lines=15
                            )
                        with gr.Column():
                            images = gr.Gallery(
                                label="Sample Extracted Images", columns=1, rows=2
                            )

        with gr.TabItem("Chat", id=2) as chat_tab:
            with gr.Accordion("Config (Advanced) (Optional)", open=False):
                with gr.Row(variant="panel", equal_height=True):
                    choice = gr.Radio(
                        ["chromaDB"],
                        value="chromaDB",
                        label="Vector Database",
                        interactive=True,
                    )
                    with gr.Accordion("Use your own model (optional)", open=False):
                        hf_token = gr.Textbox(
                            label="HuggingFace Token", interactive=True
                        )
                        model_path = gr.Textbox(label="Model Path", interactive=True)
                with gr.Row(variant="panel", equal_height=True):
                    num_context = gr.Slider(
                        label="Number of text context elements",
                        minimum=1,
                        maximum=20,
                        step=1,
                        interactive=True,
                        value=3,
                    )
                    img_context = gr.Slider(
                        label="Number of image context elements",
                        minimum=1,
                        maximum=10,
                        step=1,
                        interactive=True,
                        value=2,
                    )
                with gr.Row(variant="panel", equal_height=True):
                    temp = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=1,
                        step=0.1,
                        interactive=True,
                        value=0.4,
                    )
                    max_tokens = gr.Slider(
                        label="Max Tokens",
                        minimum=10,
                        maximum=2000,
                        step=10,
                        interactive=True,
                        value=500,
                    )
            with gr.Row():
                with gr.Column():
                    ret_images = gr.Gallery("Similar Images", columns=1, rows=2)
                with gr.Column():
                    chatbot = gr.Chatbot(height=400)
            with gr.Accordion("Text References", open=False):
                # text_context = gr.Row()

                @gr.render(inputs=references)
                def gen_refs(references):
                    # print(references)
                    n = len(references)
                    for i in range(n):
                        gr.Textbox(
                            label=f"Reference-{i+1}", value=references[i], lines=3
                        )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your question here (e.g. 'What is this document about?')",
                    interactive=True,
                    container=True,
                )
            with gr.Row():
                submit_btn = gr.Button("Submit message")
                clear_btn = gr.ClearButton([msg, chatbot], value="Clear conversation")

    pdf_btn.click(
        fn=extract_pdfs,
        inputs=[documents, doc_collection],
        outputs=[doc_collection, tabs, selected],
    )
    embed.click(
        extract_data_from_pdfs,
        inputs=[doc_collection, session_states, include_images, do_ocr],
        outputs=[
            vectordb,
            session_states,
            sample_data,
            ext_text,
            images,
            prog,
        ],
    )

    submit_btn.click(
        conversation,
        [
            vectordb,
            msg,
            num_context,
            img_context,
            chatbot,
            temp,
            max_tokens,
            hf_token,
            model_path,
        ],
        [chatbot, references, ret_images],
    )
    msg.submit(
        conversation,
        [
            vectordb,
            msg,
            num_context,
            img_context,
            chatbot,
            temp,
            max_tokens,
            hf_token,
            model_path,
        ],
        [chatbot, references, ret_images],
    )

    documents.change(
        lambda: "<h1 style='text-align: center'>Click the 'Extract' button to extract data from PDFs<h1>",
        None,
        prog,
    )

    back_p1.click(lambda: gr.Tabs(selected=0), None, tabs)

    next_p1.click(check_validity_and_llm, session_states, tabs)
if __name__ == "__main__":
    demo.launch()
