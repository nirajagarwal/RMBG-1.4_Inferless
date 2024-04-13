import base64, requests, torch
from rembg import remove
 
class InferlessPythonModel:
    def initialize(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def infer(self, inputs):
        image_url = inputs["image_url"]
        image_output = remove(requests.get(image_url).content)
        image_base64 = base64.b64encode(image_output).decode() 
        return { "generated_image_base64" : image_base64  }
        
    def finalize(self):
        self.pipe = None
        