let video;
let poseNet;
let pose;
let skeleton;

let brain;
let poseLabel = 'Y';

/* let state = 'waiting';
let targetLabel; */

function keyPressed(){
  /* if(key == 's'){
    brain.saveData();
  }else {
    if(key == 'y'){
      targetLabel = 'Dab';
    }else{
      targetLabel = key;
      console.log(targetLabel);
    }
    setTimeout(() => {
      console.log('collecting')
      state = 'collecting';
      setTimeout(() => {
        console.log('not collecting');
        state = 'waiting';
      }, 10000);
    }, 10000);
  } */

}

function setup(){
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.hide();
  poseNet = ml5.poseNet(video, modelLoaded);
  poseNet.on('pose', gotPoses);


  let options = {
    input: 34,
    output: 4,
    task: 'classification',
    debug: true
  }
  brain = ml5.neuralNetwork(options);
  const modelInfo = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin',
  }
  brain.load(modelInfo, brainLoaded);
  //brain.loadData('dab_&_y.json', dataReady);
}

function brainLoaded(){
  console.log('brain Loaded! :)!')
  classifyPose();
}

function classifyPose(){
  if(pose){
    let inputs = [];

    for(let i = 0; i < pose.keypoints.length; i++){
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    brain.classify(inputs, gotResult);
  }else {
    setTimeout(classifyPose, 100);
  }
}

function gotResult(error, results){
  //console.log(results)
  if(results[0].confidence > 0.75){
    poseLabel = results[0].label;
  }
  console.log(results[0].confidence);
  classifyPose();
}

function dataReady(){
  brain.normalizeData();
  brain.train({epochs: 50}, finished);
}

function finished(){
  console.log('model trained');
  brain.save();
}

function gotPoses(poses){
  //console.log(poses);
  if(poses.length > 0){
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
    /* let inputs = [];
    if(state === 'collecting'){
  
      for(let i = 0; i < pose.keypoints.length; i++){
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;
        inputs.push(x);
        inputs.push(y);
      }
    }

    let target = [targetLabel]; 

    brain.addData(inputs, target) */
  }
}

function modelLoaded(){
  console.log('poseNet ready! :)')
}

function draw(){
  translate(video.width, 0);
  scale(-1, 1);
  image(video, 0 ,0, video.width, video.height);

  if(pose){

    /* let eyeR = pose.rightEye;
    let eyeL = pose.leftEye;
    let d = dist(eyeR.x, eyeR.y, eyeL.x, eyeL.y); 

    fill(255, 0, 0);
    ellipse(pose.nose.x, pose.nose.y, d);
    fill(0, 0, 255); */
    ellipse(pose.rightWrist.x, pose.rightWrist.y, 32);
    ellipse(pose.leftWrist.x, pose.leftWrist.y, 32);

    for(let i = 0; i < pose.keypoints.length; i++){
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(0,255,0);
      ellipse(x, y, 16, 16);
    }

    for(let i = 0; i < skeleton.length; i++){
      let a = skeleton[i][0];
      let b = skeleton[i][1];
      strokeWeight(2);
      stroke(255);
      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }
  }
  fill(255,0,255);
  noStroke();
  textSize(156);
  textAlign(CENTER, CENTER);
  text(poseLabel, width / 2, height / 2);

}