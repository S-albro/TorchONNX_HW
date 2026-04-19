let session;

async function loadModel() {
    try {
        session = await ort.InferenceSession.create("/TorchONNX_HW/model.onnx");
        console.log("Model loaded");
        document.getElementById("result").innerText = "Model loaded";
    } catch (e) {
        console.error("Model failed to load", e);
        document.getElementById("result").innerText = "Model failed to load";
    }
}

loadModel();

async function predict() {

    if (!session) {
        alert("Model not loaded yet");
        return;
    }

    const input = new Float32Array([
        parseFloat(document.getElementById("f1").value || 0),
        parseFloat(document.getElementById("f2").value || 0),
        parseFloat(document.getElementById("f3").value || 0),
        parseFloat(document.getElementById("f4").value || 0),
        parseFloat(document.getElementById("f5").value || 0),
        parseFloat(document.getElementById("f6").value || 0),
        parseFloat(document.getElementById("f7").value || 0)
    ]);

    const tensor = new ort.Tensor("float32", input, [1, 7]);

    const results = await session.run({
        input1: tensor
    });

    const output = results.output1.data[0];

    document.getElementById("result").innerText =
        "Prediction: " + output.toFixed(2);
}
