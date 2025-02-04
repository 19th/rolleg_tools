document.addEventListener("DOMContentLoaded", async function () {
    const video = document.getElementById("myVideo");
    const frameCountInput = document.getElementById("frameCountInput");
    const imageSizeInput = document.getElementById("imageSizeInput");
    const poseProcessingSwitch = document.getElementById("poseProcessingSwitch");
    const fileInput = document.getElementById("fileInput");
    const framePreviewContainer = document.getElementById("framePreviewContainer");

    poseProcessingSwitch.addEventListener("change", extractFrames);
    frameCountInput.addEventListener("change", extractFrames);
    imageSizeInput.addEventListener("change", () => {
        const aspectRatio = video.videoWidth / video.videoHeight;
        const imageSize = parseInt(imageSizeInput.value);
        document.getElementById("imageSize").innerHTML = imageSize;
        const width = parseInt(imageSize * aspectRatio);
        const height = parseInt(imageSize);

        Array.from(document.getElementsByClassName("frame-img")).forEach((frameCanvas) => {
            frameCanvas.style.width = `${width}px`;
            frameCanvas.style.height = `${height}px`;
        });
    });

    document.getElementById('saveTimelineBtn').addEventListener('click', saveTimeline);

    function saveTimeline() {
        const timelineCanvas = document.createElement('canvas');
        const ctx = timelineCanvas.getContext('2d');
        const imageSize = parseInt(imageSizeInput.value);
        const aspectRatio = video.videoWidth / video.videoHeight;
        const frameWidth = parseInt(imageSize * aspectRatio);
        const frameHeight = parseInt(imageSize);

        const frames = Array.from(document.getElementsByClassName("frame-img"));
        timelineCanvas.width = frameWidth * frames.length;
        timelineCanvas.height = frameHeight;

        frames.forEach((frameCanvas, index) => {
            ctx.drawImage(frameCanvas, index * frameWidth, 0, frameWidth, frameHeight);
        });

        timelineCanvas.toBlob(blob => {
            saveAs(blob, 'timeline.png');
        });
    }

    document.getElementById('saveGifBtn').addEventListener('click', saveGif);
    async function saveGif() {
        const gif = new GIF({
            workers: 2,
            quality: 10,
            width: document.getElementsByClassName("frame-img")[0].width,
            height: document.getElementsByClassName("frame-img")[0].height,
        });

        const frames = Array.from(document.getElementsByClassName("frame-img"));

        for (const frameCanvas of frames) {
            gif.addFrame(frameCanvas, { delay: 200 });
        }

        gif.on('finished', function(blob) {
            saveAs(blob, 'animation.gif');
        });

        gif.render();
    }

    let bodyPoseModel;
    let videoFrames = [];

    // Load the BodyPose model
    async function loadModel() {
        try {
            bodyPoseModel = await ml5.bodyPose("MoveNet"); // Use "MoveNet" for fast detection
            console.log("BodyPose model loaded.");
        } catch (err) {
            console.error("Error loading BodyPose model:", err);
        }
    }

    loadModel(); // Initialize model

    // Handle video file upload
    fileInput.addEventListener("change", function (event) {
        const selectedFile = event.target.files[0];
        if (selectedFile) {
            const objectURL = URL.createObjectURL(selectedFile);
            video.src = objectURL;
            video.style.display = "block";
            video.load();
        } else {
            video.style.display = "none";
        }
    });

    // When video metadata is loaded
    video.addEventListener("loadedmetadata", function () {
        video.currentTime = 0;
        extractFrames();
    });

    // Extract frames from the video
    async function extractFrames() {
        framePreviewContainer.innerHTML = ""; // Clear previous frames
        videoFrames = [];
        const frameCount = parseInt(frameCountInput.value);
        document.getElementById("frameCount").innerHTML = frameCount;

        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        for (let i = 0; i < frameCount; i++) {
            video.currentTime = (i / frameCount) * video.duration;

            // Properly wait for video to seek before capturing frame
            await new Promise((resolve) => video.addEventListener("seeked", resolve, { once: true }));

            const aspectRatio = video.videoWidth / video.videoHeight;
            const imageSize = parseInt(imageSizeInput.value);

            const frameContainer = document.createElement("div");
            frameContainer.style.position = "relative";

            framePreviewContainer.appendChild(frameContainer);

            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const frameCanvas = document.createElement("canvas");
            frameCanvas.className = "frame-img";
            frameCanvas.width = canvas.width;
            frameCanvas.height = canvas.height;
            frameCanvas.style.width = `${parseInt(imageSize * aspectRatio)}px`;
            frameCanvas.style.height = `${parseInt(imageSize)}px`;
            frameCanvas.getContext("2d").drawImage(canvas, 0, 0);

            // Create save button
            const saveButton = document.createElement("button");
            saveButton.className = "btn btn-sm btn-light save-btn";
            saveButton.innerHTML = '<i class="fas fa-save"></i>';
            saveButton.addEventListener("click", () => {
                html2canvas(frameCanvas).then(canvas => {
                    canvas.toBlob(blob => {
                        saveAs(blob, `frame_${i + 1}.png`);
                    });
                });
            });

            const deleteButton = document.createElement("button");
            deleteButton.innerHTML = '<i class="fas fa-trash-alt"></i>';
            deleteButton.className = "btn btn-sm btn-danger delete-btn";

            deleteButton.addEventListener("click", () => {
                frameCanvas.remove();
                saveButton.remove();
                deleteButton.remove();
                frameContainer.remove();
            });

            // Append buttons to the frame container
            frameContainer.appendChild(saveButton);
            frameContainer.appendChild(deleteButton);

            if (poseProcessingSwitch.checked) {
                // Process frame for pose detection
                await processFrame(frameCanvas);
            }

            videoFrames.push(frameCanvas);
            frameContainer.appendChild(frameCanvas);
        }
    }

    // Process a frame for pose estimation
    async function processFrame(canvas) {
        if (!bodyPoseModel) {
            console.warn("BodyPose model is not loaded yet.");
            return;
        }

        const ctx = canvas.getContext("2d");

        try {
            const results = await bodyPoseModel.detect(canvas);
            if (results && results[0] && results[0].keypoints) {
                drawPose(results[0], ctx);
            } else {
                console.warn("No pose detected in frame.");
            }
        } catch (error) {
            console.error("Error estimating pose:", error);
        }
    }

    // Draw detected pose keypoints and skeleton
    function drawPose(pose, ctx) {
        ctx.strokeStyle = "white";
        ctx.lineWidth = 5;

        if (!pose.keypoints || pose.keypoints.length === 0) {
            console.warn("Pose data missing keypoints.");
            return;
        }

        pose.keypoints.forEach((keypoint) => {
            if (keypoint && keypoint.x && keypoint.y) {
                ctx.beginPath();
                ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
                ctx.fillStyle = "blue";
                ctx.fill();
                ctx.stroke();
            }
        });

        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4], // Right arm
            [0, 5], [5, 6], [6, 7], [7, 8], // Left arm
            [0, 9], [9, 10], [10, 11], [11, 12], // Torso
        ];

        connections.forEach(([start, end]) => {
            const kp1 = pose.keypoints[start];
            const kp2 = pose.keypoints[end];
            if (kp1 && kp2) {
                ctx.beginPath();
                ctx.moveTo(kp1.x, kp1.y);
                ctx.lineTo(kp2.x, kp2.y);
                ctx.stroke();
            }
        });
    }
});
