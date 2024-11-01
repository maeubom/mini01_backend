from fastapi import APIRouter, WebSocket, Request, Form
from fastapi.responses import StreamingResponse
import websocket
import uuid
import json
import urllib.request
import urllib.parse
import random
import io
from PIL import Image

router = APIRouter()
# server_address = "127.0.0.1:8188"
server_address = "192.168.0.61:8188"  # 임시 서버 IP
client_id = str(uuid.uuid4())
ws = websocket.WebSocket()

# 웹소켓 연결을 설정하는 함수
async def setup_websocket():
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}")

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"http://{server_address}/view?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
        return json.loads(response.read())

def get_images(prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break # Execution is done
        else:
            continue # previews are binary data
    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                images_output.append(image_data)
        output_images[node_id] = images_output
    return output_images

# =====================================================================
# 웹소켓 세팅, 라우터 설정


@router.post("/v1/api/text-to-image")
async def process_request(query: str = Form(...)):
    try:
        if not ws.connected:
            await setup_websocket()
            print("comfy connected!")
        prompt_text = """
        {
        "3": {
            "inputs": {
            "seed": 51727030489353,
            "steps": 10,
            "cfg": 3,
            "sampler_name": "euler_cfg_pp",
            "scheduler": "sgm_uniform",
            "denoise": 1,
            "model": [
                "10",
                0
            ],
            "positive": [
                "6",
                0
            ],
            "negative": [
                "7",
                0
            ],
            "latent_image": [
                "5",
                0
            ]
            },
            "class_type": "KSampler",
            "_meta": {
            "title": "KSampler"
            }
        },
        "4": {
            "inputs": {
            "ckpt_name": "juggernautXL_juggXIByRundiffusion.safetensors"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {
            "title": "Load Checkpoint"
            }
        },
        "5": {
            "inputs": {
            "width": 1024,
            "height": 1024,
            "batch_size": 1
            },
            "class_type": "EmptyLatentImage",
            "_meta": {
            "title": "Empty Latent Image"
            }
        },
        "6": {
            "inputs": {
            "text": "cartoon, pastel, cat character, negative, annoying",
            "clip": [
                "10",
                1
            ]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
            "title": "Positive CLIP Text Encode (Prompt)"
            }
        },
        "7": {
            "inputs": {
            "text": "text, watermark",
            "clip": [
                "10",
                1
            ]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
            "title": "Negative CLIP Text Encode (Prompt)"
            }
        },
        "8": {
            "inputs": {
            "samples": [
                "3",
                0
            ],
            "vae": [
                "4",
                2
            ]
            },
            "class_type": "VAEDecode",
            "_meta": {
            "title": "VAE Decode"
            }
        },
        "9": {
            "inputs": {
            "filename_prefix": "ComfyUI",
            "images": [
                "8",
                0
            ]
            },
            "class_type": "SaveImage",
            "_meta": {
            "title": "Save Image"
            }
        },
        "10": {
            "inputs": {
            "lora_name": "simple_drawing_xl_b1-000012.safetensors",
            "strength_model": 1,
            "strength_clip": 1,
            "model": [
                "4",
                0
            ],
            "clip": [
                "4",
                1
            ]
            },
            "class_type": "LoraLoader",
            "_meta": {
            "title": "Load LoRA"
            }
        }
        }
        """
        prompt = json.loads(prompt_text)
        prompt["3"]["inputs"]["seed"] = int(random.randint(0, 2**32 - 1))
        prompt["6"]["inputs"]["text"] = "cartoon, pastel, cat character" + query
        prompt["5"]["inputs"]["width"] = 512
        prompt["5"]["inputs"]["height"] = 512
        prompt["5"]["inputs"]["batch_size"] = 1
        images = get_images(prompt)
        
        node_id = next(iter(images))
        image_data = images[node_id][0]
        image = Image.open(io.BytesIO(image_data))


        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)


        return StreamingResponse(img_byte_arr, media_type="image/png")
    except Exception as e:
        return {"error": str(e)}

