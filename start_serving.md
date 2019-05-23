# Start deoldify serving on Kibernetika.AI

## Ensure model exists

The serving requires PyTorch model. To make it available, need to upload it to
the Kibernetika.AI using `kdataset` CLI tool. 

The archive containing `.pth` file can be downloaded using this URL: 

```bash
https://dev.kibernetika.io/api/v0.2/workspace/kuberlab-demo/mlmodel/deoldify/versions/1.0.0-stable-pytorch/download/model-deoldify-1.0.0-stable-pytorch.tar
```

After downloading and unpacking the `.tar` file, you are ready for uploading.
In directory containing the `.pth` file:

```bash
kdataset --type model push <workspace> deoldify:1.0.0-stable-pytorch
```

## Starting and configuring serving

* Open Kibernetika.AI web UI, change to appropriate workspace
* Click to the Catalog -> Models, find just uploaded model
* Click to Versions tab, click "Serve"
* Configure serving such as the following:

![](imgs/serving.png)

> **Note**: Pick up the appropriate workspace, cluster and name.

> If you don't have an available GPU, then change **GPU** parameter in **Resources** section to 0.

* Click **Serve**
* Wait for container is up and running (look at the **Status** Tab periodically and click "Refresh")
* Go to **Parameters** Tab
* Choose an image at bottom and click **Apply**

## Start streaming

Configuring streaming is almost the same as serving. Just need to slightly change execution command in config:

* **kserving** -> **kstreaming**
* Add new arguments:
    * **--input-name input** - this specifies the input key name for streaming (streaming needs to know how to interact with serving hook/driver)
    * **--output-name output** - this specifies the output key name as well
    * **--input <your input>** - Need to specify video source. It can be rtmp URL, rtsp URL, udp camera, path to video file etc.
    * **--output <rtmp-output>** - RTMP output URL (e.g. stream to twitch.tv or youtube)

Full execution command will look like (in case of streaming from rtsp camera to twitch.tv): 

```bash
kstreaming --driver pytorch --model-path $MODEL_DIR/model.pth --hooks hook.py -o output_type=image \
 --input-name input --output-name output --input rtsp://IP \
 --output rtmp://live-waw.twitch.tv/app/mykey
```

**Note:** You can start intermediate rtmp server and then stream on it. For example, start streaming with arguments
```bash
--input server --initial-stream my-stream --output rtmp://live-waw.twitch.tv/app/mykey
```

leads to starting a rtmp server and listening stream **/live/my-stream**. After launching the kstreaming you can 
push your stream using ffmpeg:
```bash
ffmpeg -re -i video-file.mp4 -vcodec libx264 -acodec aac -r 30 -f flv rtmp://<kstreaming-IP>/live/my-stream
```