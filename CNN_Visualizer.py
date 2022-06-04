from ctypes import alignment
from email.policy import default
from operator import mod
from unicodedata import name
import streamlit as st
from streamlit_option_menu import option_menu
import cv2 # import opencv - open images with python
import numpy as np # numpy - expand_dims 
import requests  # pip install requests
from streamlit_lottie import st_lottie
import pickle
from tensorflow.keras.models import load_model, Sequential # Modelling

def model_call(model,input):
    input_image = cv2.imdecode(input, 1)
    resize = cv2.resize(input_image, (256,256))
    resize = cv2.cvtColor(resize,cv2.COLOR_BGR2RGB)
    st.image(resize, caption='Input Image.', use_column_width=True)

    yhat = model.predict(np.expand_dims(resize,0))

    shape_1 = yhat.shape

    st.image([yhat[0,:,:,i] for i in range(shape_1[3])],clamp=True,width=300)
    st.write('Output images from all Neurons in that layer.')
                        
    st.success('However, We Highly encourage you to guess what features are CNN is trying to extract/Highlight.')


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def main():
    
    st.set_page_config(layout="wide")

    option_type = option_menu(
        menu_title = None,
        options = ["Home","Default Model","Custom Model","Contact"],
        icons = ["house","book","award-fill","envelope"],
        menu_icon = "cast",
        default_index = 0,
        orientation = "horizontal",
    )

    
    st.markdown("<h1 style='text-align: center; color: #08e08a;'>CNN Visualizer</h1>",unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #dae5e0;'>Let's see how cnn see the pictures and extract features from it.</h4>",unsafe_allow_html=True)

    if option_type == 'Home':
        lottie_home = load_lottieurl("https://assets4.lottiefiles.com/private_files/lf30_GjhcdO.json")

        st_lottie(
            lottie_home,
            speed=1,
            reverse=False,
            loop=True,
            quality="high", # medium ; high
            #renderer="svg", # canvas
            height=400,
            width=-400,
            key=None,
        )
        st.markdown("<h4 style='text-align: center; color: #dae5e0;'>Let's go through each and every layer and find out how it see the world.</h4>",unsafe_allow_html=True)
        

    elif option_type == 'Default Model':
        st.subheader("Upload Image")
        image = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

        model = load_model('./imageclassifier.h5')
        st.markdown("<h2 style='text-align: center; color: #c9b42c;'>Default Model Architecture</h2>",unsafe_allow_html=True)
        model.summary(print_fn=lambda x: st.text(x))
        
        #print(model.get_layer('max_pooling2d'))

        if image != None:
            file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)

            layers = model.layers
            list = []
            for layer in layers:
                list.append(layer.name)
                #print(layer.output.type_spec.shape[-1])
            

            list = filter(lambda x: x.startswith(('conv2d','max_pooling2d')), list)

            layer_name = st.sidebar.selectbox('Layers',list)

            if len(layer_name) > 0:
                index = None
                for idx, layer in enumerate(model.layers):  
                    if layer.name == layer_name:
                        index = idx
                        break
            
            selected_layer = model.layers[:idx+1]
            next_model = Sequential(selected_layer)

            model_call(next_model,file_bytes)

            

        else:
            st.warning("Upload image to start Visualizing.")
        
    elif option_type == "Custom Model":
        st.subheader("Upload Your Model")
        model_custom = st.file_uploader("Upload Model", type=['h5'])

        if model_custom != None:
            image_upload = st.file_uploader("Upload Image",type=['jpg', 'png', 'jpeg'])
            if image_upload != None:
                file_bytes_1 = np.asarray(bytearray(image_upload.read()), dtype=np.uint8)
                st.markdown("<h2 style='color: #c9b42c;'>Input Model Architecture</h2>",unsafe_allow_html=True)
                if model_custom.type == 'application/x-hdf':
                    model_custom_1 = load_model(model_custom.name)
                    model_custom_1.summary(print_fn=lambda x: st.text(x))

                layers = model_custom_1.layers
                list = []
                for layer in layers:
                    list.append(layer.name)
                
                list = filter(lambda x: x.startswith(('conv2d','max_pooling2d')), list)

                layer_name = st.sidebar.selectbox('Layers',list)

                if len(layer_name) > 0:
                    index = None
                    for idx, layer in enumerate(model_custom_1.layers):  
                        if layer.name == layer_name:
                            index = idx
                            break

                    selected_layer = model_custom_1.layers[:idx+1]
                    next_model = Sequential(selected_layer)

                    model_call(next_model,file_bytes_1)
                

        
    
    else:
        st.subheader("About")
        st.text("This is an CNN Visualizer, Made by a student of the IIIT-Kalyani, India.")
        st.text("Sai Dileep Kumar Mukkamala")
        st.text("Senior CSE Student @ IIITKalyani")
        st.text("Data Science Enthusiasist.")

        lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_in4cufsz.json")

        st_lottie(
            lottie_hello,
            speed=1,
            reverse=False,
            loop=True,
            quality="high", # medium ; high
            #renderer="svg", # canvas
            height=300,
            width=-900,
            key=None,
        )

        
        st.header(":mailbox: Get In Touch With Me!")


        contact_form = """
        <form action="https://formsubmit.co/msaidileepkumar2002@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here"></textarea>
            <button type="submit">Send</button>
        </form>
        """

        st.markdown(contact_form, unsafe_allow_html=True)

        # Use Local CSS File
        def local_css(file_name):
            with open(file_name) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


        local_css("style/style.css")



if __name__ == '__main__':
    main()
