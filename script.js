let session;

async function loadModel(){

    session = await ort.InferenceSession.create("model.onnx");

}

loadModel();

async function predict(){

    const input = new Float32Array([

        parseFloat(document.getElementById("f1").value),
        parseFloat(document.getElementById("f2").value),
        parseFloat(document.getElementById("f3").value),
        parseFloat(document.getElementById("f4").value),
        parseFloat(document.getElementById("f5").value),
        parseFloat(document.getElementById("f6").value),
        parseFloat(document.getElementById("f7").value),
        parseFloat(document.getElementById("f8").value),
        parseFloat(document.getElementById("f9").value),
        parseFloat(document.getElementById("f10").value)

    ]);

    const tensor = new ort.Tensor("float32", input, [1,10]);

    const results = await session.run({input:tensor});

    const prediction = results.output.data[0];

    document.getElementById("result").innerText =
        "Predicted progression score: " + prediction.toFixed(2);
}