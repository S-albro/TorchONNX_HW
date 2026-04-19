let session;

async function loadModel() {
    try {
        console.log("Loading model...");

        session = await ort.InferenceSession.create(
            "./model.onnx",
            {
                executionProviders: ["wasm"]
            }
        );

        console.log("Model loaded successfully");

        document.getElementById("result").innerText = "Model loaded";

        console.log("Inputs:", session.inputNames);
        console.log("Outputs:", session.outputNames);

    } catch (e) {
        console.error("MODEL LOAD FAILED:", e);
        document.getElementById("result").innerText = "Model failed to load";
    }
}

loadModel();

async function predict() {

    if (!session) {
        alert("Model not ready yet");
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

    const inputName = session.inputNames[0];

    const feeds = {};
    feeds[inputName] = tensor;

    const results = await session.run(feeds);

    const outputName = session.outputNames[0];

    const output = results[outputName].data[0];

    document.getElementById("result").innerText =
        "Prediction: " + output.toFixed(2);
}
