$("#image-selector").change(function() {
    let reader = new FileReader();
    reader.onload = function() {
        let dataURL = reader.result;
        $("#selected-image").attr("src", dataURL);
        $("#prediction-list").empty();
    }
    let file =$("#image-selector").prop('files')[0];
    reader.readAsDataURL(file);
});

let model;
(async function (){
    $(".progress-bar").show();
    model = undefined;
    model = await tf.loadModel('http://localhost:81/tfjs-models/predict-strength-model/model.json');
    $('.progress-bar').hide();
})();


$("#predict-button").click(async function() {
    let image = $("#selected-image").get(0);
    let tensor = tf.fromPixels(image)
        .resizeNearestNeighbor([512, 512])
        .toFloat()
        .reshape([1,512,512,1]);

    let prediction = await model.predict(tensor).data();
    $("#prediction-list").empty();
    $("#prediction-list").append(`<li>${prediction.toFixed(4)}</li>`);
});
