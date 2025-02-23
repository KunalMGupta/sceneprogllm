def text_to_3d(text):
    import os
    import trimesh
    import requests

    response = requests.post("http://brahmastra.ucsd.edu:3001/generate_glb", json={"text": text})
    os.makedirs("tmp", exist_ok=True)
    if response.status_code == 200:
        with open("tmp/received_model.glb", "wb") as f:
            f.write(response.content)
    else:
        raise Exception("Failed to generate 3D model")
    
    mesh = trimesh.load("tmp/received_model.glb", force="mesh", process=False)
    os.system("rm -rf tmp/received_model.glb")
    return mesh