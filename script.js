let session;

async function loadModel() {
    try {
        session = await ort.InferenceSession.create("model.onnx");
        console.log("Model loaded");
    } catch (e) {
        console.error("Model load failed", e);
    }
}

loadModel();

async function predict() {

    if (!session) {
        alert("Model not loaded yet");
        return;
    }

    const input = new Float32Array([
        parseFloat(document.getElementById("f1").value),
        parseFloat(document.getElementById("f2").value),
        parseFloat(document.getElementById("f3").value),
        parseFloat(document.getElementById("f4").value),
        parseFloat(document.getElementById("f5").value),
        parseFloat(document.getElementById("f6").value),
        parseFloat(document.getElementById("f7").value)
    ]);

    const tensor = new ort.Tensor("float32", input, [1, 7]);

    const results = await session.run({
        input1: tensor
    });

    const output = results.output1.data[0];

    document.getElementById("result").innerText =
        "Prediction: " + output.toFixed(2);
}
