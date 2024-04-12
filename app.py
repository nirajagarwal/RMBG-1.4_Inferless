from transformers import pipeline
from io import BytesIO
import base64, requests
from rembg import remove
 
class InferlessPythonModel:
    def initialize(self):
        self.pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

    def infer(self, inputs):
        image_url = inputs["image_url"]
        #pillow_image = self.pipe(image_url) # applies mask on input and returns a pillow image        
        #buff = BytesIO()
        #pillow_image.save(buff, format="PNG")
        #image_base64 = base64.b64encode(buff.getvalue()).decode()
        image_output = remove(requests.get(image_url).content)
        image_base64 = base64.b64encode(image_output).decode() 
        return { "generated_image_base64" : image_base64  }

    def finalize(self):
        self.pipe = None
        