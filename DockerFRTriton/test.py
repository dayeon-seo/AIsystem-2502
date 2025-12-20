import onnx
model = onnx.load(r"C:\Users\sdyha\OneDrive\문서\GitHub\AIsystem-2502\DockerFRTriton\model_repository\face_detector\1\model.onnx")
for o in model.graph.output:
    print(o.name)
