from flask import Flask, request, render_template, redirect, url_for, session
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from werkzeug.utils import secure_filename
from langchain.schema import HumanMessage
import os
import uuid
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from PIL import Image
import pytesseract


app = Flask(__name__)
#add your secret key here
app.secret_key = 'your_secret_key'
CORS(app)

load_dotenv()
#add your openai key in env
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

conversation_store = {}

def process_image(img_file):
    img = Image.open(img_file)

    # Rotate the image based on its EXIF orientation data
    try:
        exif = img._getexif()
        if exif is not None:
            orientation = exif.get(0x0112, 1)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass  # If the image doesn't have EXIF data, just continue

    # Convert the image to RGB mode if it's not already
    if img.mode != 'RGB':
        img = img.convert('RGB')

    text = pytesseract.image_to_string(img, lang='eng')
    return text

def create_conversation_key():
    return str(uuid.uuid4())

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4-turbo-2024-04-09",openai_api_key=OPENAI_API_KEY)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        pdf_files = request.files.getlist('pdf_files')
        image_files = request.files.getlist('image_files')

        if pdf_files or image_files:
            text = ""
            for pdf in pdf_files:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()

            for img_file in image_files:
                img_text = process_image(img_file)
                text += img_text

            text_chunks = get_text_chunks(text)
            vectorstore = get_vectorstore(text_chunks)
            conv_key = create_conversation_key()
            conversation_store[conv_key] = get_conversation_chain(vectorstore)
            return jsonify({'conversation_key': conv_key})

    return jsonify({'error': 'Invalid request'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    conv_key = request.json.get('conversation_key')
    if conv_key not in conversation_store:
        return jsonify({'error': 'Conversation not found'}), 404
    
    conversation = conversation_store[conv_key]
    
    user_input = request.json.get('user_input')
    if user_input:
        response = conversation({'question': user_input})
        chat_history = response['chat_history']
        
        # Convert HumanMessage objects to a JSON-serializable format
        serialized_chat_history = [
            {'type': 'human', 'content': message.content}
            if isinstance(message, HumanMessage)
            else {'type': 'ai', 'content': message.content}
            for message in chat_history
        ]
        
        return jsonify({'chat_history': serialized_chat_history})
    
    return jsonify({'error': 'Invalid request'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
