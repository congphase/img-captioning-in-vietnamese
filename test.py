import sample
import cv2
from build_vocab import Vocabulary
import json
import requests
import urllib
from playsound import playsound
import os

url = "https://viettelgroup.ai/voice/api/tts/v1/rest/syn"
headers = {'Content-type': 'application/json', 'token': '-wGcnpMasnRrNWs3Grdc5vWV9l0FkIQrQ6BHbiPBB0UY1h2eI89NBD-O-mpaCvAl'}


try:
 
    from enum import Enum
    from io import BytesIO, StringIO
    from typing import Union
 
    import pandas as pd
    import streamlit as st
except Exception as e:
    print(e)
 
STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""
image = ''
 
class FileUpload(object):
    def __init__(self):
        self.fileTypes = ["csv", "png", "jpg", "jpeg"]
 
    def run(self):
        """
        Upload File on Streamlit Code
        :return:
        """
        file_uploaded = 0

        st.info(__doc__)
        st.markdown(STYLE, unsafe_allow_html=True)
        file = st.file_uploader("Upload file", type=self.fileTypes)
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " + ", ".join(["csv", "png", "jpg"]))
            return
        content = file.getvalue()
        file_uploaded = 1

        if isinstance(file, BytesIO):
            show_file.image(file)
        else:
            data = pd.read_csv(file)
            st.dataframe(data.head(10))
        file.close()


        image_path = str(file.name)
        encoder_path = "encoder-50-20.ckpt"
        decoder_path = "decoder-50-20.ckpt"
        vocab_path = "preprocessed_vocab_.pkl"


        if(file_uploaded==1):
            returned_sentence = sample.run_inference(image_path, encoder_path, decoder_path, vocab_path)
            st.write(returned_sentence)

            data = {"text": f'{returned_sentence}', "voice": "phamtienquan", "id": "2", "without_filter": True, "speed": 0.9, "tts_return_option": 2}
            response = requests.post(url, data=json.dumps(data), headers=headers)
            print(f'DEBUG: response.headers: {response.headers}')
            print(f'DEBUG: response: {response}')
            print(f'DEBUG: response.status_code: {response.status_code}')

            data = response.content
            
            save_name = image_path.rstrip('.jpg')
            save_name = save_name.rstrip('.png')
            save_name = save_name.rstrip('.jpeg')
            path_to_save = f'audio/{save_name}.wav'
            print(f'debug: COUNT = {save_name}')

            f = open(path_to_save, "wb")
            f.write(data)
            f.close()

            playsound(path_to_save)
            file_uploaded = 0
        # else:
        #     print(f'debug: Insert an image')
        #     cv2.waitKey(1)
 
if __name__ ==  "__main__":
    helper = FileUpload()
    helper.run()